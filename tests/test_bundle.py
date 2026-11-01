"""Debug bundles: allowlisted, redacted, footage opt-in, verifiable, tamper-evident (plan P2 step 7, D10)."""

import json
import zipfile
from pathlib import Path

import pytest

from motionman import bundle, cli, doctor, pipeline, records, store
from motionman.types import RunConfig

from .test_pipeline import FakeTracker

BASE = {"manifest.json", "runspec.json", "report.json", "tracks.jsonl", "timings.jsonl", "telemetry.jsonl",
        "doctor.json", "README.txt", "bundle_manifest.json"}  # fmt: skip


@pytest.fixture
def published(clip, tmp_path, monkeypatch):
    monkeypatch.setenv("MOTIONMAN_TEST_SECRET", "s3cr3t-do-not-leak")
    monkeypatch.setattr(doctor, "_probe_devices", lambda d: doctor.Check("device", doctor.PASS, "fake"))
    root = tmp_path / "data"
    report = pipeline.run(RunConfig(clip, None, data_root=root, telemetry=True), make_tracker=FakeTracker,
                          resolve=lambda d: "cpu")  # fmt: skip
    return store.data_root(root), report


def entries(path):
    with zipfile.ZipFile(path) as z:
        return {n: z.read(n) for n in z.namelist()}


def make(published, **kw):
    root, report = published
    return bundle.make_bundle(report["run_id"], data_root=root.path, **kw)


def rewrite(src, dst, change):
    with zipfile.ZipFile(src) as z:
        items = {n: z.read(n) for n in z.namelist()}
    change(items)
    with zipfile.ZipFile(dst, "w") as z:
        for n, data in items.items():
            z.writestr(n, data)
    return dst


def test_a_default_bundle_holds_only_the_allowlist_and_verifies(published):
    path = make(published)
    got = entries(path)
    assert set(got) == BASE and all("/" not in n for n in got)
    result = bundle.verify(path)
    assert result["ok"] and result["entries"] == len(BASE) - 1 and result["run_id"] == published[1]["run_id"]
    manifest = json.loads(got["bundle_manifest.json"])
    assert any("footage" in e for e in manifest["excluded"]) and manifest["options"]["include_footage"] is False


def test_it_reopens_as_usable_evidence(published):
    got = entries(make(published))
    for name in ("manifest.json", "runspec.json", "report.json", "doctor.json"):
        json.loads(got[name])
    lines = got["tracks.jsonl"].decode().splitlines()
    assert [records.parse_line(line, i)["source_frame_index"] for i, line in enumerate(lines)] == list(range(20))
    readme = got["README.txt"].decode()
    assert "bundle --verify" in readme and "--device cpu --imgsz 640" in readme


def test_no_paths_home_folder_or_environment_leak(published, tmp_path):
    blob = b"".join(entries(make(published)).values())
    assert str(tmp_path).encode() not in blob and str(Path.home()).encode() not in blob
    assert b"s3cr3t-do-not-leak" not in blob and b"<redacted>/" in blob


def test_full_paths_only_when_asked(published, tmp_path):
    got = entries(make(published, include_paths=True))
    assert str(tmp_path).encode() in got["report.json"]


def test_footage_only_when_asked(published):
    root, report = published
    got = entries(make(published, include_footage=True))
    assert set(got) == BASE | {"annotated.mp4", "source.mp4"}
    run_dir = root.runs / report["run_id"]
    assert got["annotated.mp4"] == (run_dir / "annotated.mp4").read_bytes()
    assert store.sha256_bytes(got["source.mp4"]) == report["input_id"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda e: e.__setitem__("report.json", b"{}"), "report.json does not match its hash"),
        (lambda e: e.__setitem__("evil.txt", b"x"), "unexpected entry 'evil.txt'"),
        (lambda e: e.__setitem__("../escape.json", b"x"), "unexpected entry"),
        (lambda e: e.pop("bundle_manifest.json"), "bundle_manifest.json is missing"),
        (lambda e: e.pop("tracks.jsonl"), "entries and bundle_manifest.json disagree"),
    ],
)
def test_tampering_is_refused(published, tmp_path, change, message):
    bad = rewrite(make(published), tmp_path / "bad.zip", change)
    with pytest.raises(bundle.BundleError, match=message):
        bundle.verify(bad)


def test_unreadable_unknown_and_damaged_inputs_are_refused(published, tmp_path):
    root, report = published
    junk = tmp_path / "junk.zip"
    junk.write_bytes(b"not a zip")
    with pytest.raises(bundle.BundleError, match="not a readable ZIP"):
        bundle.verify(junk)
    with pytest.raises(bundle.BundleError, match="no published run"):
        bundle.make_bundle("r_nope", data_root=root.path)
    with (root.runs / report["run_id"] / "tracks.jsonl").open("ab") as f:
        f.write(b"x")
    with pytest.raises(bundle.BundleError, match="fails verification"):
        make(published)


def test_redaction_rules():
    home = "/Users/someone"
    assert bundle.redact("/Users/someone/clips/crosswalk.mp4", home) == "<redacted>/clips/crosswalk.mp4"
    assert bundle.redact("/opt/data/runs/r_1", home) == "<redacted>/r_1"
    assert bundle.redact("data root /var/tmp/x/data is writable", home) == "data root <redacted>/data is writable"
    assert bundle.redact({"a": ["encode/decode", "sha256:ab/cd"], "n": 3}, home) == {
        "a": ["encode/decode", "sha256:ab/cd"], "n": 3}  # fmt: skip
    assert bundle.redact("~/x and /Users/someone", home) == "~/x and <redacted>"


def test_cli_bundle_and_verify(published, tmp_path, capsys):
    root, report = published
    assert cli.main(["bundle", "--run", report["run_id"], "--data-root", str(root.path)]) == cli.EXIT_OK
    path = Path(capsys.readouterr().out.split("bundle written: ")[1].split(" (")[0])
    assert cli.main(["bundle", "--verify", str(path)]) == cli.EXIT_OK
    bad = rewrite(path, tmp_path / "bad.zip", lambda e: e.__setitem__("report.json", b"{}"))
    assert cli.main(["bundle", "--verify", str(bad)]) == cli.EXIT_FAILURE
    assert cli.main(["bundle", "--run", "r_nope", "--data-root", str(root.path)]) == 2

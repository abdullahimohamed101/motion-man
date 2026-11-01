"""Regression tests for the P0 review findings: truncated sources, publication failure, input reuse, record
ownership, runtime limits, error wording, and a lease held by a separate process."""

import json
import shutil
import subprocess
import sys

import pytest

from motionman import cli, pipeline, records, store, video_io
from motionman.types import RunConfig
from motionman.ui.runner import describe_error

from .test_pipeline import FakeTracker


def canonical(clip, root, **kw):
    return RunConfig(clip, None, data_root=root, **kw)


def run(config, tracker=FakeTracker):
    return pipeline.run(config, make_tracker=tracker, resolve=lambda d: "cpu")


def staged_manifest(root):
    (staged,) = list((root / "staging").iterdir())
    return json.loads((staged / store.MANIFEST).read_text())


# --- P1: a decoder that stops early is not end of file -----------------------------------------------------


def test_early_end_of_decoding_fails_against_the_container_count(clip, tmp_path, monkeypatch):
    real = video_io.read_frames

    def stops_after_eight(cap, fps, *args, **kw):
        for frame in real(cap, fps, *args, **kw):
            if frame.index == 8:
                return  # what OpenCV does on a damaged packet: read() just returns False
            yield frame

    monkeypatch.setattr(video_io, "read_frames", stops_after_eight)
    root = tmp_path / "data"
    with pytest.raises(video_io.VideoError, match="stopped at frame 8 of the 20 the container reports"):
        run(canonical(clip, root))
    m = staged_manifest(root)
    assert m["status"] == "FAILED" and m["validation"]["source_complete"] == "FAIL"
    assert list((root / "runs").iterdir()) == []


def test_requested_max_frames_is_complete_not_truncated(clip, tmp_path):
    report = run(canonical(clip, tmp_path / "data", max_frames=7))
    m = json.loads((tmp_path / "data" / "runs" / report["run_id"] / store.MANIFEST).read_text())
    assert m["validation"]["source_complete"] == "PASS" and m["counts"]["decoded_frames"] == 7


def test_unknown_container_count_is_recorded_as_unverified(clip, tmp_path, monkeypatch):
    real = video_io.open_reader

    def no_count(path, limits=None):
        cap, info = real(path, *([limits] if limits else []))
        return cap, type(info)(info.path, info.width, info.height, info.fps, None)

    monkeypatch.setattr(video_io, "open_reader", no_count)
    report = run(canonical(clip, tmp_path / "data"))
    m = json.loads((tmp_path / "data" / "runs" / report["run_id"] / store.MANIFEST).read_text())
    assert m["validation"]["source_complete"] == "UNVERIFIED" and m["counts"]["source_frames_reported"] is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed; real truncated clip not generated")
def test_real_truncated_faststart_mp4_is_not_published(tmp_path):
    full = tmp_path / "full.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30", "-frames:v", "60",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(full)],
        check=True, timeout=60,
    )  # fmt: skip
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(full.read_bytes()[: full.stat().st_size // 4])
    root = tmp_path / "data"
    with pytest.raises(video_io.VideoError, match="truncated or damaged"):
        run(canonical(cut, root))
    assert list((root / "runs").iterdir()) == []


# --- publication and reporting --------------------------------------------------------------------------------


def test_publish_failure_leaves_no_succeeded_manifest(clip, tmp_path, monkeypatch):
    def refuse(staging, root, run_id):
        raise OSError("rename failed")

    monkeypatch.setattr(store, "publish", refuse)
    root = tmp_path / "data"
    with pytest.raises(OSError, match="rename failed"):
        run(canonical(clip, root))
    m = staged_manifest(root)
    assert m["status"] == "FAILED" and m["error"] == "PublishError: rename failed"


def test_failed_canonical_report_points_at_staging_not_runs(clip, tmp_path):
    root = tmp_path / "data"
    with pytest.raises(RuntimeError):
        run(canonical(clip, root), tracker=lambda c: FakeTracker(c, fail_at=3, exc=RuntimeError("boom")))
    (staged,) = list((root / "staging").iterdir())
    report = json.loads((staged / pipeline.REPORT).read_text())
    assert report["run_dir"] == str(staged) and not (root / "runs" / staged.name).exists()


# --- input import -----------------------------------------------------------------------------------------


def test_corrupted_managed_copy_is_replaced_on_reuse(clip, tmp_path):
    root = store.data_root(tmp_path / "data").ensure()
    input_id, managed = store.import_input(clip, root)
    managed.write_bytes(b"corrupted")
    assert store.import_input(clip, root) == (input_id, managed)
    assert store.sha256_file(managed) == input_id


def test_missing_meta_is_rewritten_and_names_do_not_duplicate_bytes(clip, tmp_path):
    root = store.data_root(tmp_path / "data").ensure()
    other_name = tmp_path / "same-bytes.MOV"
    other_name.write_bytes(clip.read_bytes())
    input_id, managed = store.import_input(clip, root)
    (managed.parent / "meta.json").unlink()
    assert store.import_input(other_name, root, display_name="Shown.mov") == (input_id, managed)
    assert sorted(p.name for p in managed.parent.iterdir()) == ["meta.json", "source"]  # one copy per hash
    assert json.loads((managed.parent / "meta.json").read_text())["display_name"] == "Shown.mov"


def test_canonical_source_changed_mid_run_fails_and_is_not_published(clip, tmp_path):
    root = tmp_path / "data"

    class TamperManaged(FakeTracker):
        def update(self, image):
            if self.calls == 5:
                (managed,) = list((root / "inputs").glob("*/source"))
                with managed.open("ab") as f:
                    f.write(b"tampered")
            return super().update(image)

    with pytest.raises(store.SourceChangedError, match="changed during the run"):
        run(canonical(clip, root), tracker=TamperManaged)
    assert list((root / "runs").iterdir()) == []


# --- record ownership -------------------------------------------------------------------------------------


def lines(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return path


def test_records_check_rejects_legacy_lines_other_runs_and_other_sources(tmp_path):
    good = [records.frame_record("r_a", i, 30.0, 64, 48, []) for i in range(3)]
    assert store.validate_records_file(lines(tmp_path / "ok.jsonl", good), 3, "r_a", 64, 48).startswith("sha256:")
    with pytest.raises(store.StoreError, match="another run or source"):
        store.validate_records_file(lines(tmp_path / "id.jsonl", good), 3, "r_b", 64, 48)
    with pytest.raises(store.StoreError, match="another run or source"):
        store.validate_records_file(lines(tmp_path / "dims.jsonl", good), 3, "r_a", 128, 48)
    legacy = [{"frame_index": i, "timestamp_s": i / 30, "objects": []} for i in range(3)]
    with pytest.raises(store.StoreError, match="pre-v1.0"):
        store.validate_records_file(lines(tmp_path / "old.jsonl", legacy), 3, "r_a", 64, 48)


# --- runtime limits, wording, CLI ---------------------------------------------------------------------------


def test_duration_limit_applies_at_runtime_when_the_count_is_unknown():
    class Cap:
        def __init__(self):
            self.i = -1

        def read(self):
            self.i += 1
            return True, None

        def get(self, prop):
            return self.i * 1000 / 30

    with pytest.raises(video_io.VideoError, match="0.5-minute limit"):
        for _frame in video_io.read_frames(Cap(), 30.0, max_duration_s=30.0):
            pass


def test_source_change_gets_its_own_next_step():
    message, step = describe_error(store.SourceChangedError("x changed during the run"))
    assert step == "Choose the video again; it changed while it was being processed."


def test_cli_rejects_data_root_with_output_dir(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(["--source", "a.mp4", "--output-dir", "out", "--data-root", "d"])
    assert exc.value.code == 2 and "--data-root applies to canonical runs" in capsys.readouterr().err


# --- a lease held by another process ------------------------------------------------------------------------


def test_lease_held_by_another_process_is_refused(clip, tmp_path):
    holder = subprocess.Popen(
        [sys.executable, "-c",
         "import sys, time; from pathlib import Path; from motionman import store\n"
         "with store.execution_lease(store.data_root(Path(sys.argv[1]))):\n"
         "    print('held', flush=True); time.sleep(30)",
         str(tmp_path / "data")],
        stdout=subprocess.PIPE, text=True,
    )  # fmt: skip
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(store.LeaseError):
            run(canonical(clip, tmp_path / "data"))
    finally:
        holder.kill()
        holder.wait(10)
    assert run(canonical(clip, tmp_path / "data"))["status"] == "success"  # freed when the process ends

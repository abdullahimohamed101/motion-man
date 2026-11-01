"""Sanitized, verifiable debug bundles for one run (plan P2 D10, MD§21 P2 step 6).

A bundle is built from an allowlist, never by sweeping a folder. Full paths and the home folder are redacted
unless asked for, footage is left out unless asked for, and no environment variables are ever read into it.
``bundle_manifest.json`` lists every entry's size and hash; ``verify`` re-hashes them and refuses anything else.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import zipfile
import zlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import motionman
from motionman import doctor, pipeline, runs, store
from motionman.types import MotionManError

SCHEMA_VERSION = "1.0"
BUNDLE_MANIFEST = "bundle_manifest.json"
README = "README.txt"
JSON_ENTRIES = (store.MANIFEST, store.RUNSPEC, pipeline.REPORT)
JSONL_ENTRIES = (pipeline.TRACKS, pipeline.TIMINGS, pipeline.TELEMETRY)
FOOTAGE_ENTRIES = (pipeline.ANNOTATED, "source.mp4", "source.m4v", "source.mov")
ALLOWED = {*JSON_ENTRIES, *JSONL_ENTRIES, "doctor.json", README, BUNDLE_MANIFEST, *FOOTAGE_ENTRIES}
REQUIRED = (*JSON_ENTRIES, pipeline.TRACKS, "doctor.json", README)  # timings/telemetry are absent on older runs
INTEGRITY_ONLY = "integrity check: entries match bundle_manifest.json; not proof of who made the bundle"
MAX_UNCOMPRESSED = 4 * 1024**3  # refuse to verify anything implausibly large (a zip bomb, or not ours)
REDACTED = "<redacted>"
# An absolute POSIX path token: starts with "/" (not inside a word such as "encode/decode"), two or more parts.
_ABS_PATH = re.compile(r"(?<![\w.~<>])/(?:[^\s/'\"]+/)+[^\s/'\"]*")


class BundleError(MotionManError):
    exit_code = 2


def redact(value, home: str = str(Path.home())):
    """Replace the home folder with ``<redacted>`` and other absolute paths with ``<redacted>/<basename>``."""
    if isinstance(value, dict):
        return {k: redact(v, home) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, home) for v in value]
    if not isinstance(value, str):
        return value
    home = home if home and home != "/" else None

    def one(path: str) -> str:
        name = Path(path).name
        # The home folder's last part is the username: never keep it as a basename.
        if home and (path.rstrip("/") == home or name == Path(home).name):
            return REDACTED
        return f"{REDACTED}/{name}" if name else REDACTED

    if value.startswith("/") and "\n" not in value:
        return one(value)  # a whole value that is a path may contain spaces ("/Volumes/My Drive/x.mp4")
    out = _ABS_PATH.sub(lambda m: one(m.group(0)), value)
    return out.replace(home, REDACTED) if home else out


def make_bundle(
    run_id: str,
    *,
    data_root: Path | None = None,
    include_footage: bool = False,
    include_paths: bool = False,
    run_doctor: Callable[..., dict] = doctor.run_doctor,
) -> Path:
    """Write ``exports/bundle-<run_id>-<hex>/bundle.zip`` for a verified published run; returns its path."""
    root = store.data_root(data_root)
    summary = next((s for s in runs.list_runs(root) if s.run_id == run_id), None)
    if summary is None:
        raise BundleError(f"no published run {run_id!r} under {root.runs}")
    try:
        run = runs.open_run(summary, root)  # every artifact re-hashed before anything is copied
    except runs.RunError as e:
        raise BundleError(f"can't bundle a run that fails verification: {e}") from None
    folder = summary.path
    clean = (lambda v: v) if include_paths else redact
    entries: dict[str, bytes | Path] = {}  # small entries in memory; footage streamed from disk
    for name in JSON_ENTRIES:
        entries[name] = _json_bytes(clean(json.loads((folder / name).read_text(encoding="utf-8"))))
    for name in JSONL_ENTRIES:
        if (folder / name).is_file():
            entries[name] = (folder / name).read_bytes()  # numbers and IDs only: no paths to redact
    entries["doctor.json"] = _json_bytes(
        clean(
            run_doctor(
                data_root=root.path, load_model=False, device=run.runspec["settings"]["inference"]["device_requested"]
            )
        )
    )
    excluded = []
    if include_footage:
        entries[pipeline.ANNOTATED] = folder / pipeline.ANNOTATED
        entries["source" + store.safe_suffix(summary.display_name or "")] = run.source_path
    else:
        excluded.append("footage (annotated.mp4 and the source video): pass --include-footage to add them")
    if not include_paths:
        excluded.append("full paths and the home folder are redacted: pass --include-paths to keep them")
    entries[README] = _readme(run, sorted(entries) + [README]).encode()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "motionman_version": motionman.__version__,
        "options": {"include_footage": include_footage, "include_paths": include_paths},
        "entries": [_entry(n, v) for n, v in sorted(entries.items())],
        "excluded": excluded,
        "never_included": "environment variables, credentials, other runs, unlisted files",
    }
    out = root.path / "exports" / f"bundle-{run_id}-{secrets.token_hex(3)}"
    out.mkdir(parents=True)
    tmp = out / ".bundle.zip.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(entries.items()):
            if isinstance(data, Path):
                z.write(data, arcname=name)
            else:
                z.writestr(name, data)
        z.writestr(BUNDLE_MANIFEST, _json_bytes(manifest))
    tmp.replace(out / "bundle.zip")
    return out / "bundle.zip"


def verify(path: Path) -> dict:
    """Re-hash every entry against ``bundle_manifest.json``; refuse unknown, unsafe, missing or changed entries."""
    try:
        z = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as e:
        raise BundleError(f"{path} is not a readable ZIP: {e}") from None
    with z:
        infos = z.infolist()
        names = [i.filename for i in infos]
        for name in names:
            if "/" in name or "\\" in name or name.startswith(".") or name not in ALLOWED:
                raise BundleError(f"unexpected entry {name!r}: bundles are flat and allowlisted")
        if len(set(names)) != len(names):
            raise BundleError("duplicate entries in the bundle")
        if sum(i.file_size for i in infos) > MAX_UNCOMPRESSED:
            raise BundleError("bundle is implausibly large when uncompressed")
        if BUNDLE_MANIFEST not in names:
            raise BundleError(f"{BUNDLE_MANIFEST} is missing")
        try:
            manifest = json.loads(z.read(BUNDLE_MANIFEST))
            listed = {e["name"]: e for e in manifest["entries"]}
        except (ValueError, KeyError, TypeError) as e:
            raise BundleError(f"{BUNDLE_MANIFEST} is unreadable: {e}") from None
        if set(listed) != set(names) - {BUNDLE_MANIFEST}:
            raise BundleError("entries and bundle_manifest.json disagree")
        missing = [n for n in REQUIRED if n not in listed]
        if missing:
            raise BundleError(f"required entries missing: {', '.join(missing)}")
        try:
            for name, entry in listed.items():
                h, size = hashlib.sha256(), 0
                with z.open(name) as f:  # streamed: footage entries can be hundreds of MB
                    while chunk := f.read(1 << 20):
                        h.update(chunk)
                        size += len(chunk)
                if size != entry["bytes"] or f"sha256:{h.hexdigest()}" != entry["sha256"]:
                    raise BundleError(f"{name} does not match its hash in {BUNDLE_MANIFEST}")
            run_id = json.loads(z.read(store.MANIFEST)).get("run_id")
        except (zipfile.BadZipFile, zlib.error, EOFError, KeyError, TypeError, ValueError, AttributeError) as e:
            raise BundleError(f"bundle is damaged: {type(e).__name__}: {e}") from None
        if run_id != manifest.get("run_id"):
            raise BundleError(f"{BUNDLE_MANIFEST} names run {manifest.get('run_id')!r} but manifest.json is {run_id!r}")
    return {"ok": True, "run_id": run_id, "entries": len(listed), "options": manifest.get("options"),
            "scope": INTEGRITY_ONLY}  # fmt: skip


def _entry(name: str, data: bytes | Path) -> dict:
    if isinstance(data, Path):
        return {"name": name, "bytes": data.stat().st_size, "sha256": store.sha256_file(data)}
    return {"name": name, "bytes": len(data), "sha256": store.sha256_bytes(data)}


def _json_bytes(obj) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode()


def _readme(run: runs.OpenRun, names: list[str]) -> str:
    s = run.runspec["settings"]
    inf, frame_range = s["inference"], s["input"]["frame_range"]
    flags = f"--device {inf['device_requested']} --imgsz {inf['imgsz']}"
    if frame_range[1] is not None:
        flags += f" --max-frames {frame_range[1]}"
    return f"""MotionMan debug bundle for run {run.summary.run_id}

Contents: {", ".join(names)}, plus {BUNDLE_MANIFEST} (size and sha256 of every entry).

Verify it:
  python -m motionman bundle --verify bundle.zip
This is an integrity check (every entry matches {BUNDLE_MANIFEST}), not proof of who made the bundle.

Reproduce it (needs the same input video and weights; check their hashes in runspec.json):
  input  {s["input"]["input_id"]}
  model  {s["model"]["weights"]} {s["model"].get("weights_sha256")}
  python -m motionman --source <your copy of the input> {flags}
Compare timings only within the same power state and hardware (see report.json "environment" and
"timing", and doctor.json). Track IDs are local to this run.
"""

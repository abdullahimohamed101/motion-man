"""Identity, provenance and storage for runs (spec A§2, A§3, A§5): content hashes, canonical JSON, run IDs, the
resolved RunSpec, the data root, the execution lease, manifests and atomic publication. Nothing here imports
Ultralytics or torch, so it stays usable in offline tests (INV-8).
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import platform
import secrets
import shutil
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from motionman import records
from motionman.types import MotionManError, RunConfig, VideoInfo

RUNSPEC_SCHEMA = "1.0"
TRACKER_CONFIG = "bytetrack.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]
_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def sha256_file(path: Path) -> str:
    """Streamed SHA-256 of a file's bytes, as ``sha256:<hex>``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def canonical_json(obj: Any) -> str:
    """Sorted keys, fixed separators, no NaN/Infinity (raises ValueError): stable bytes for hashing (A§2)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def new_run_id(now: datetime | None = None) -> str:
    """One execution attempt: ``r_<UTC timestamp>_<6 hex>`` (D7). Repeats always get a new ID."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"r_{stamp}_{secrets.token_hex(3)}"


def tracker_config() -> dict:
    """The installed ByteTrack configuration, read from the package files without importing Ultralytics."""
    spec = importlib.util.find_spec("ultralytics")
    if spec is None or spec.origin is None:
        raise FileNotFoundError("ultralytics is not installed")
    path = Path(spec.origin).parent / "cfg" / "trackers" / TRACKER_CONFIG
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def code_revision(repo: Path = REPO_ROOT) -> dict:
    """Git commit and whether the tree had uncommitted changes; unknown outside a git checkout."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return {"commit": commit, "dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None, "reason": "not a git checkout or git unavailable"}


def optional_hash(path: Path, missing_reason: str) -> dict:
    return {"sha256": sha256_file(path)} if path.is_file() else {"sha256": None, "reason": missing_reason}


def resolve_runspec(
    config: RunConfig,
    info: VideoInfo,
    *,
    input_id: str,
    device_used: str,
    weights_path: Path | None = None,
    class_names: Mapping[int, str] | None = None,
    codec: str | None = None,
    limits: Mapping[str, Any] | None = None,
) -> dict:
    """Resolve every default once and persist it (A§3). ``config_sha256`` covers the settings that determine
    the result; ``reproducibility`` (code, dependencies, machine) is recorded beside it but not hashed, so the
    same settings on a new commit can still be compared as the same configuration."""
    tracker = tracker_config()
    weights = weights_path or Path(config.model)
    names = {str(k): v for k, v in sorted(class_names.items())} if class_names is not None else None
    settings = {
        "input": {
            "input_id": input_id,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "frame_count_reported_by_container": info.frame_count,
            "frame_range": [0, config.max_frames],  # [start, stop); None = to the end of the source
        },
        "model": {
            "task": "detect",
            "weights": config.model,  # the identifier as requested; the hash says which bytes were used
            **{f"weights_{k}": v for k, v in optional_hash(weights, "weights file not present").items()},
            "class_map_sha256": sha256_json(names) if names is not None else None,
        },
        "tracking": {"type": "bytetrack", "config": tracker, "config_sha256": sha256_json(tracker)},
        "inference": {
            "device_requested": config.device,
            "device_used": device_used,
            "imgsz": config.imgsz,
            "conf": config.conf,
            "iou": config.iou,
            "class_filter": None,
            "precision": "fp32",
        },
        "sampling": {"policy": "every_frame"},
        "rendering": {
            "enabled": True,
            "codec": codec,
            "trail_points": 30,
            "trail_expiry_frames": 60,
            "audio": "dropped",
        },
        "limits": dict(limits or {}),
    }
    return {
        "schema_version": RUNSPEC_SCHEMA,
        "settings": settings,
        "config_sha256": sha256_json(settings),
        "class_names": names,
        "reproducibility": {
            "code": code_revision(),
            "requirements": optional_hash(REPO_ROOT / "requirements.txt", "requirements.txt not found"),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


# --- Data root, lease, staging and publication (spec A§2, A§5) ------------------------------------------

MANIFEST = "manifest.json"
MANAGED_SOURCE = "source"  # no extension: the bytes, not the name, identify the input
RUNSPEC = "runspec.json"
MANIFEST_SCHEMA = "1.0"
DEFAULT_DATA_ROOT = Path("data")


class StoreError(MotionManError):
    """A storage contract was violated (lease held, publication blocked, record file invalid)."""


class LeaseError(StoreError):
    exit_code = 75  # EX_TEMPFAIL: another run is active; try again later


class SourceChangedError(StoreError):
    """The source bytes changed during import or the run; its results cannot be trusted."""


@dataclass(frozen=True)
class DataRoot:
    path: Path

    @property
    def inputs(self) -> Path:
        return self.path / "inputs"

    @property
    def staging(self) -> Path:
        return self.path / "staging"

    @property
    def runs(self) -> Path:
        return self.path / "runs"

    def ensure(self) -> DataRoot:
        for d in (self.inputs, self.staging, self.runs):
            d.mkdir(parents=True, exist_ok=True)
        return self


def data_root(path: Path | None = None) -> DataRoot:
    """``--data-root``, else ``MOTIONMAN_DATA_ROOT``, else ``./data`` (D1)."""
    return DataRoot(Path(path or os.environ.get("MOTIONMAN_DATA_ROOT") or DEFAULT_DATA_ROOT))


@contextmanager
def execution_lease(root: DataRoot) -> Iterator[None]:
    """One active run per data root, across processes (D6). Released on every exit, including Ctrl-C."""
    root.path.mkdir(parents=True, exist_ok=True)
    fd = os.open(root.path / ".execution.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise LeaseError(
                f"another MotionMan run is using {root.path}; wait for it to finish, then try again"
            ) from None
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the lock


def import_input(source: Path, root: DataRoot, display_name: str | None = None) -> tuple[str, Path]:
    """Copy a source into ``inputs/<hex>/source`` once, keyed by content hash only (D5). Returns (input_id, path).

    The filename is display metadata; the managed path comes from the hash, never from user text. An existing
    copy is re-hashed before reuse, and replaced if it no longer matches; ``meta.json`` is written whenever it
    is missing, so a crash between copy and metadata heals on the next import."""
    input_id = sha256_file(source)
    folder = root.inputs / input_id.removeprefix("sha256:")
    managed = folder / MANAGED_SOURCE
    folder.mkdir(parents=True, exist_ok=True)
    if not (managed.is_file() and sha256_file(managed) == input_id):
        tmp = folder / f".{MANAGED_SOURCE}.tmp"
        shutil.copyfile(source, tmp)
        if sha256_file(tmp) != input_id:
            tmp.unlink(missing_ok=True)
            raise SourceChangedError(f"{source} changed while it was being imported")
        os.replace(tmp, managed)
    if not (folder / "meta.json").is_file():
        write_json_atomic(
            folder / "meta.json",
            {"input_id": input_id, "display_name": display_name or source.name, "bytes": managed.stat().st_size,
             "imported_at_utc": datetime.now(UTC).isoformat(timespec="seconds")},
        )  # fmt: skip
    return input_id, managed


def write_json_atomic(path: Path, obj: Any) -> None:
    """Temp file then rename, so a crash never leaves half a JSON file."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def validate_records_file(path: Path, expected: int, run_id: str, width: int, height: int) -> str:
    """Check a run's frame records before publication (A§5 step 3): native v1.0 lines (no legacy adaptation),
    this run's ID and source size, contiguous order, and the expected count. Returns the file's hash."""
    n = -1
    for n, rec in enumerate(records.read_records(path)):
        if rec.get("legacy"):
            raise StoreError(f"{path.name}: line {n + 1} is a pre-v1.0 record")
        if rec["run_id"] != run_id or (rec["source_width"], rec["source_height"]) != (width, height):
            raise StoreError(f"{path.name}: record {n} belongs to another run or source")
        if rec["source_frame_index"] != n:
            raise StoreError(f"{path.name}: record {n} has source_frame_index {rec['source_frame_index']}")
    if n + 1 != expected:
        raise StoreError(f"{path.name}: {n + 1} records, expected {expected}")
    return sha256_file(path)


def artifact_entries(folder: Path, names: list[str]) -> list[dict]:
    return [{"path": n, "bytes": (folder / n).stat().st_size, "sha256": sha256_file(folder / n)} for n in names]


def build_manifest(
    *,
    run_id: str,
    status: str,
    input_id: str | None,
    config_sha256: str | None,
    records_sha256: str | None,
    started_at: datetime,
    finished_at: datetime,
    counts: dict,
    artifacts: list[dict],
    validation: dict,
    error: str | None = None,
) -> dict:
    """Terminal manifest (A§3). status is SUCCEEDED, FAILED or INTERRUPTED; only SUCCEEDED runs publish."""
    return {
        "schema_version": MANIFEST_SCHEMA,
        "run_id": run_id,
        "status": status,
        "error": error,
        "input_id": input_id,
        "config_sha256": config_sha256,
        "records_sha256": records_sha256,
        "started_at_utc": started_at.isoformat(timespec="milliseconds"),
        "finished_at_utc": finished_at.isoformat(timespec="milliseconds"),
        "counts": counts,
        "artifacts": artifacts,
        "validation": validation,
        "capabilities": {"masks": False, "gpu_telemetry": False},
    }


def publish(staging: Path, root: DataRoot, run_id: str) -> Path:
    """Atomically move a validated staging folder into ``runs/`` (same filesystem). Never overwrites."""
    target = root.runs / run_id
    if target.exists():
        raise StoreError(f"run {run_id} is already published")
    os.rename(staging, target)
    return target


def safe_suffix(name: str) -> str:
    """A known video extension from a user-supplied name, else .mp4; never a path component."""
    suffix = Path(name).suffix.lower()
    return suffix if suffix in {".mp4", ".m4v", ".mov"} else ".mp4"

"""Identity and provenance for runs (spec A§2, A§3): content hashes, canonical JSON, run IDs, resolved RunSpec.

Data-root layout, staging, publication and the execution lease are added here in P0 step 4. Nothing in this
module imports Ultralytics or torch, so it stays usable in offline tests (INV-8).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import secrets
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from motionman.types import RunConfig, VideoInfo

RUNSPEC_SCHEMA = "1.0"
TRACKER_CONFIG = "bytetrack.yaml"
REPO_ROOT = Path(__file__).resolve().parents[1]
_CHUNK = 1 << 20


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

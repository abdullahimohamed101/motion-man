"""Explicit, saved settings (spec A§8, plan P1 D7). A preset keeps two kinds of settings apart:

- ``view``: presentation only (labels, trails, class filter). Applying it re-renders; nothing re-runs.
- ``tracking``: inference settings (device, confidence). Applying it only fills the next run's settings; it never
  relabels an existing result (UI2: "These settings apply to your next run").

A spotlight is never saved: track IDs are local to one run, so a saved ID would point at a different object
in any other run.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from motionman import store
from motionman.render import MAX_TRAIL_LENGTH, ViewOptions
from motionman.types import MotionManError

SCHEMA_VERSION = "1.0"
NAME = re.compile(r"[A-Za-z0-9_-]{1,40}")
DEVICES = ("cpu", "auto", "cuda")
CONF_RANGE = (0.05, 0.95)


class PresetError(MotionManError):
    """A preset name or file is invalid; the message says which field."""


@dataclass(frozen=True)
class TrackingSettings:
    device: str = "cpu"
    conf: float = 0.25


@dataclass(frozen=True)
class Preset:
    name: str
    view: ViewOptions | None = None  # spotlight is always None in a preset
    tracking: TrackingSettings | None = None


def folder(root: store.DataRoot) -> Path:
    return root.path / "presets"


def save(root: store.DataRoot, preset: Preset, *, overwrite: bool = False) -> Path:
    _check_name(preset.name)
    if preset.view is None and preset.tracking is None:
        raise PresetError("a preset needs view settings, tracking settings, or both")
    path = folder(root) / f"{preset.name}.json"
    if path.exists() and not overwrite:
        raise PresetError(f"a preset named {preset.name!r} already exists")
    body = {"schema_version": SCHEMA_VERSION, "name": preset.name,
            "saved_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "view": _view_dict(preset.view), "tracking": _tracking_dict(preset.tracking)}  # fmt: skip
    _parse(body, preset.name)  # never write something load() would refuse
    folder(root).mkdir(parents=True, exist_ok=True)
    store.write_json_atomic(path, body)
    return path


def load(root: store.DataRoot, name: str) -> Preset:
    _check_name(name)
    path = folder(root) / f"{name}.json"
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PresetError(f"no preset named {name!r}") from None
    except (OSError, ValueError) as e:
        raise PresetError(f"preset {name!r} is not readable JSON") from e
    return _parse(body, name)


def list_names(root: store.DataRoot) -> list[str]:
    if not folder(root).is_dir():
        return []
    return sorted(p.stem for p in folder(root).glob("*.json") if NAME.fullmatch(p.stem))


def _check_name(name: str) -> None:
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise PresetError("preset names use 1–40 letters, digits, hyphens or underscores")


def _view_dict(view: ViewOptions | None) -> dict | None:
    if view is None:
        return None
    return {"labels": view.labels, "trail_length": view.trail_length, "trail_opacity": view.trail_opacity,
            "classes": list(view.classes) if view.classes is not None else None}  # fmt: skip


def _tracking_dict(tracking: TrackingSettings | None) -> dict | None:
    return None if tracking is None else {"device": tracking.device, "conf": tracking.conf}


def _parse(body: object, name: str) -> Preset:
    """Allowlisted fields with type and range checks (A§10); unknown or out-of-range values are refused."""
    if not isinstance(body, dict) or body.get("schema_version") != SCHEMA_VERSION:
        raise PresetError(f"preset {name!r} has an unsupported format")
    extra = set(body) - {"schema_version", "name", "saved_at_utc", "view", "tracking"}
    if extra:
        raise PresetError(f"preset {name!r} has unknown fields: {', '.join(sorted(extra))}")
    view = tracking = None
    if body.get("view") is not None:
        v = body["view"]
        if not isinstance(v, dict) or set(v) - {"labels", "trail_length", "trail_opacity", "classes"}:
            raise PresetError(f"preset {name!r}: view has unknown fields (a spotlight is never saved)")
        labels, length, opacity = v.get("labels", True), v.get("trail_length", 30), v.get("trail_opacity", 1.0)
        classes = v.get("classes")
        if not isinstance(labels, bool):
            raise PresetError(f"preset {name!r}: labels must be true or false")
        if not isinstance(length, int) or isinstance(length, bool) or not 0 <= length <= MAX_TRAIL_LENGTH:
            raise PresetError(f"preset {name!r}: trail_length must be a whole number from 0 to {MAX_TRAIL_LENGTH}")
        if not isinstance(opacity, int | float) or isinstance(opacity, bool) or not 0.0 <= opacity <= 1.0:
            raise PresetError(f"preset {name!r}: trail_opacity must be between 0 and 1")
        if classes is not None and (not isinstance(classes, list) or not all(isinstance(c, str) for c in classes)):
            raise PresetError(f"preset {name!r}: classes must be a list of class names or null")
        view = ViewOptions(labels, length, float(opacity), None, tuple(classes) if classes is not None else None)
    if body.get("tracking") is not None:
        t = body["tracking"]
        if not isinstance(t, dict) or set(t) - {"device", "conf"}:
            raise PresetError(f"preset {name!r}: tracking has unknown fields")
        device, conf = t.get("device", "cpu"), t.get("conf", 0.25)
        if device not in DEVICES:
            raise PresetError(f"preset {name!r}: device must be one of {', '.join(DEVICES)}")
        if not isinstance(conf, int | float) or isinstance(conf, bool) or not CONF_RANGE[0] <= conf <= CONF_RANGE[1]:
            raise PresetError(f"preset {name!r}: conf must be between {CONF_RANGE[0]} and {CONF_RANGE[1]}")
        tracking = TrackingSettings(device, float(conf))
    if view is None and tracking is None:
        raise PresetError(f"preset {name!r} has neither view nor tracking settings")
    return Preset(name, view, tracking)

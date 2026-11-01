"""Presets keep view and tracking settings apart (plan P1 step 5, D7)."""

import json

import pytest

from motionman import presets, store
from motionman.presets import Preset, PresetError, TrackingSettings
from motionman.render import ViewOptions


@pytest.fixture
def root(tmp_path):
    return store.data_root(tmp_path / "data")


def test_round_trip_keeps_view_and_tracking_separate(root):
    p = Preset("busy-street", ViewOptions(labels=False, trail_length=12, trail_opacity=0.5, classes=("person", "car")),
               TrackingSettings(device="auto", conf=0.4))  # fmt: skip
    presets.save(root, p)
    loaded = presets.load(root, "busy-street")
    assert loaded == p
    assert presets.list_names(root) == ["busy-street"]


def test_view_only_and_tracking_only_presets(root):
    presets.save(root, Preset("v", view=ViewOptions(trail_length=0)))
    presets.save(root, Preset("t", tracking=TrackingSettings(conf=0.6)))
    assert presets.load(root, "v").tracking is None and presets.load(root, "t").view is None
    with pytest.raises(PresetError, match="needs view settings, tracking settings, or both"):
        presets.save(root, Preset("empty"))


def test_a_spotlight_is_never_saved(root):
    presets.save(root, Preset("s", view=ViewOptions(spotlight=17)))
    body = json.loads((presets.folder(root) / "s.json").read_text())
    assert "spotlight" not in body["view"] and presets.load(root, "s").view.spotlight is None


@pytest.mark.parametrize("name", ["", "a" * 41, "../escape", "with space", "dot.json", "slash/name"])
def test_names_are_restricted(root, name):
    with pytest.raises(PresetError, match="1–40 letters"):
        presets.save(root, Preset(name, view=ViewOptions()))
    with pytest.raises(PresetError, match="1–40 letters"):
        presets.load(root, name)


def test_existing_names_need_explicit_overwrite(root):
    presets.save(root, Preset("x", view=ViewOptions()))
    with pytest.raises(PresetError, match="already exists"):
        presets.save(root, Preset("x", view=ViewOptions(labels=False)))
    presets.save(root, Preset("x", view=ViewOptions(labels=False)), overwrite=True)
    assert presets.load(root, "x").view.labels is False


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"view": {"trail_length": 61}}, "trail_length must be a whole number from 0 to 60"),
        ({"view": {"trail_opacity": 1.5}}, "trail_opacity must be between 0 and 1"),
        ({"view": {"labels": "yes"}}, "labels must be true or false"),
        ({"view": {"spotlight": 3}}, "a spotlight is never saved"),
        ({"view": {"classes": "person"}}, "classes must be a list"),
        ({"tracking": {"device": "tpu"}}, "device must be one of cpu, auto, cuda"),
        ({"tracking": {"conf": 0.99}}, "conf must be between 0.05 and 0.95"),
        ({"tracking": {"conf": 0.3, "model": "evil.pt"}}, "tracking has unknown fields"),
        ({"extra": 1}, "unknown fields: extra"),
        ({"schema_version": "9"}, "unsupported format"),
    ],
)
def test_tampered_files_are_refused_field_by_field(root, change, message):
    presets.save(root, Preset("p", view=ViewOptions(), tracking=TrackingSettings()))
    path = presets.folder(root) / "p.json"
    body = json.loads(path.read_text())
    for key, value in change.items():
        body[key] = {**body[key], **value} if isinstance(value, dict) and isinstance(body.get(key), dict) else value
    path.write_text(json.dumps(body))
    with pytest.raises(PresetError, match=message):
        presets.load(root, "p")


def test_missing_and_unreadable_presets(root):
    with pytest.raises(PresetError, match="no preset named 'nope'"):
        presets.load(root, "nope")
    presets.folder(root).mkdir(parents=True)
    (presets.folder(root) / "bad.json").write_text("{not json")
    with pytest.raises(PresetError, match="not readable JSON"):
        presets.load(root, "bad")
    assert presets.list_names(root) == ["bad"]  # listed, but it refuses to load rather than half-applying

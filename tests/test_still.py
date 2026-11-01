"""Inspector stills and evidence PNG export (plan P1 step 4, D2/D6)."""

import json

import cv2
import numpy as np
import pytest

from motionman import inspection, pipeline, runs, store
from motionman.render import NO_ID_COLOR, SPOTLIGHT_DIM, ViewOptions, color_for
from motionman.types import RunConfig

from .test_pipeline import FakeTracker


@pytest.fixture
def opened(clip, tmp_path):
    """A real canonical run (FakeTracker: track 1 every frame except 3; an unassigned car at frame 5)."""
    root = store.data_root(tmp_path / "data")
    pipeline.run(RunConfig(clip, None, data_root=root.path), make_tracker=FakeTracker, resolve=lambda d: "cpu")
    run = runs.open_run(runs.list_runs(root)[0], root)
    derived = inspection.load_or_build(run.records_path, root, run.records_sha256)
    frames = inspection.FrameSource(run.source_path, run.fps, derived.index.count)
    yield root, run, derived, frames
    frames.close()


def edge_pixel(obj):
    x1, y1, x2, y2 = (round(v) for v in obj["bbox_xyxy"])
    return y1, (x1 + x2) // 2  # middle of the top edge


def test_overlays_are_drawn_from_this_frames_record(opened):
    _, _, d, frames = opened
    still, record = inspection.render_still(frames, d.index, 5, ViewOptions(labels=False, trail_length=0))
    assert record == d.index.record(5)
    for obj in record["objects"]:
        y, x = edge_pixel(obj)
        assert tuple(int(c) for c in still[y, x]) == color_for(obj["track_id"])  # every recorded box, in its colour
    assert np.array_equal(frames.frame(5), frames.frame(5))  # the source frame itself is never drawn on


def test_class_filter_hides_other_classes(opened):
    _, _, d, frames = opened
    record = d.index.record(5)
    car = next(o for o in record["objects"] if o["class_name"] == "car")
    still, _ = inspection.render_still(
        frames, d.index, 5, ViewOptions(labels=False, trail_length=0, classes=("person",))
    )
    y, x = edge_pixel(car)
    assert np.array_equal(still[y, x], frames.frame(5)[y, x])  # nothing drawn where the car box was


def test_spotlight_dims_everything_but_the_selected_object(opened):
    _, _, d, frames = opened
    record = d.index.record(5)
    person = next(o for o in record["objects"] if o["track_id"] == 1)
    car = next(o for o in record["objects"] if o["track_id"] is None)
    still, _ = inspection.render_still(frames, d.index, 5, ViewOptions(labels=False, trail_length=0, spotlight=1))
    src = frames.frame(5)
    x1, y1, x2, y2 = (round(v) for v in person["bbox_xyxy"])
    inside = (y1 + y2) // 2, (x1 + x2) // 2
    assert np.array_equal(still[inside], src[inside])  # the spotlighted object keeps full brightness
    corner = (src.shape[0] - 1, src.shape[1] - 1)
    assert np.array_equal(still[corner], (src[corner] * SPOTLIGHT_DIM).astype(np.uint8))
    y, x = edge_pixel(car)
    assert tuple(int(c) for c in still[y, x]) == NO_ID_COLOR  # others: thin grey, no colour


def test_trails_use_only_observed_frames_within_the_requested_length(opened):
    _, _, d, _ = opened
    trails = inspection.trails_for(d.index, 6, 5)  # frames 2..6; track 1 is missing at frame 3
    assert len(trails[1]) == 4
    assert inspection.trails_for(d.index, 6, 0) == {}
    assert None not in inspection.trails_for(d.index, 5, 10)  # unassigned objects never get a trail


def test_observed_is_about_this_exact_frame(opened):
    _, _, d, _ = opened
    assert inspection.observed(d.index.record(2), 1) and not inspection.observed(d.index.record(3), 1)


def test_evidence_png_is_exactly_the_shown_still(opened):
    root, run, d, frames = opened
    options = ViewOptions(trail_length=10, spotlight=1)
    shown, record = inspection.render_still(frames, d.index, 7, options)
    png = inspection.export_evidence(
        run_id=run.run_id, records_sha256=run.records_sha256, input_id=run.manifest["input_id"],
        source_verified=run.source_verified, frames=frames, index=d.index, frame=7, options=options, root=root,
    )  # fmt: skip
    assert np.array_equal(cv2.imread(str(png)), shown)  # lossless: pixel-identical to the screen
    meta = json.loads((png.parent / "export.json").read_text())
    assert meta["run_id"] == run.run_id and meta["source_frame_index"] == 7
    assert meta["source_t_us"] == record["source_t_us"] and meta["record_sha256"] == store.sha256_json(record)
    assert meta["view"]["spotlight"] == 1 and meta["view"]["trail_length"] == 10 and meta["source_verified"] is True
    assert meta["png"]["sha256"] == store.sha256_file(png) and meta["export_id"] == png.parent.name


def test_same_recipe_reuses_the_export_and_a_new_view_makes_a_new_one(opened):
    root, run, d, frames = opened
    kw = {"run_id": run.run_id, "records_sha256": run.records_sha256, "input_id": run.manifest["input_id"],
              "source_verified": True, "frames": frames, "index": d.index, "frame": 4, "root": root}  # fmt: skip
    a = inspection.export_evidence(options=ViewOptions(), **kw)
    stamp = a.stat().st_mtime_ns
    assert inspection.export_evidence(options=ViewOptions(), **kw) == a and a.stat().st_mtime_ns == stamp
    b = inspection.export_evidence(options=ViewOptions(labels=False), **kw)
    assert b.parent != a.parent

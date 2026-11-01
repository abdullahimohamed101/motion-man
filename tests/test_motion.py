"""Motion arrows and the heatmap overlay (plan P3 step 7, D10, D11): velocities from source time, never across a
break; overlays that change pixels only where there is something to show."""

import numpy as np
import pytest

from motionman import inspection, pipeline, render, store
from motionman.types import RunConfig

from .test_pipeline import FakeTracker


@pytest.fixture
def index(clip, tmp_path):
    """The fake track moves right 1 px per frame at 30 fps (30 px/s), is missing on frame 3, and an unassigned car
    appears on frame 5."""
    root = store.data_root(tmp_path / "data")
    report = pipeline.run(RunConfig(clip, None, data_root=root.path), make_tracker=FakeTracker, resolve=lambda d: "cpu")
    records = root.runs / report["run_id"] / pipeline.TRACKS
    return inspection.load_or_build(records, root).index


def test_velocity_comes_from_source_time_and_known_motion(index):
    m = inspection.motion_at(index, 12)[1]
    assert m.span == 5 and m.vx == pytest.approx(30, rel=1e-3) and m.vy == 0 and m.speed == pytest.approx(30, rel=1e-3)
    assert (m.x, m.y) == (17.0, 12.5)


def test_motion_never_spans_a_break_in_observation(index):
    assert inspection.motion_at(index, 0) == {}  # nothing before the first frame
    assert inspection.motion_at(index, 4) == {}  # first sighting after the frame-3 gap: no motion yet
    assert inspection.motion_at(index, 5)[1].span == 1  # only frame 4 → 5 is continuous
    assert inspection.motion_at(index, 2)[1].span == 2
    assert set(inspection.motion_at(index, 5)) == {1}  # the unassigned car has no ID, so no arrow


def test_arrows_are_drawn_only_for_moving_tracks():
    still = np.zeros((48, 64, 3), np.uint8)
    moving = {1: inspection.Motion(20, 20, 30, 0, 5)}
    assert render.draw_motion(still, moving).any()
    assert not render.draw_motion(still, {1: inspection.Motion(20, 20, 0, 0, 5)}).any()
    assert not render.draw_motion(still, {1: inspection.Motion(20, 20, 6, 6, 5)}).any()  # 8.5 px/s: jitter, not drawn


def test_the_heatmap_overlay_changes_pixels_only_where_there_is_observed_time():
    frame = np.full((144, 256, 3), 100, np.uint8)
    empty = {"grid": [256, 144], "cells": []}
    assert np.array_equal(render.heatmap_overlay(frame, empty, 0.6), frame)
    heat = {"grid": [256, 144], "cells": [[10, 20, 5_000], [11, 20, 1_000]]}
    assert np.array_equal(render.heatmap_overlay(frame, heat, 0.0), frame)
    out = render.heatmap_overlay(frame, heat, 0.6)
    changed = np.argwhere((out != frame).any(axis=2))
    assert len(changed) and all(abs(r - 20) <= 12 and abs(c - 10.5) <= 12 for r, c in changed)  # only near the cells
    assert np.array_equal(out[100:, 100:], frame[100:, 100:])  # far from any observed time: untouched
    diff = np.abs(out.astype(int) - frame.astype(int)).sum(axis=2)
    assert diff[20, 10] > diff[20, 13] > 0  # strongest at the busiest cell, fading with distance
    assert np.array_equal(frame, np.full((144, 256, 3), 100, np.uint8))  # the input frame is never modified

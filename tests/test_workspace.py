"""Per-run context and display helpers for the result workspace (plan P1 step 6, D9)."""

from motionman.inspection import TrackSummary
from motionman.render import ViewOptions
from motionman.ui import workspace
from motionman.ui.workspace import RunContext


def summary(tid=1, intervals=((0, 2), (5, 9)), observations=None):
    obs = observations if observations is not None else sum(b - a + 1 for a, b in intervals)
    return TrackSummary(tid, "person", {"person": obs}, intervals[0][0], intervals[-1][1], 0, 0, obs,
                        [list(i) for i in intervals], [], 0.5, 0.6, 0.7)  # fmt: skip


def test_restore_drops_what_this_run_does_not_have():
    saved = RunContext(frame=500, track=42, view={"labels": False, "trail_length": 999}, spotlight=False)
    ctx = workspace.restore(saved, frame_count=100, tracks={1: summary()})
    assert (ctx.frame, ctx.track, ctx.spotlight) == (0, None, False)
    assert ctx.view["labels"] is False and ctx.view["trail_length"] == 60  # clamped to MAX_TRAIL_LENGTH


def test_restore_keeps_a_valid_context_and_defaults_a_new_run():
    ctx = workspace.restore(RunContext(frame=7, track=1), 20, {1: summary()})
    assert (ctx.frame, ctx.track) == (7, 1)
    assert workspace.restore(None, 20, {}) == RunContext()


def test_view_options_clamp_and_take_the_spotlight_separately():
    v = workspace.view_options({"trail_length": -3, "trail_opacity": 4, "classes": ["car"]}, spotlight=9)
    assert v == ViewOptions(labels=True, trail_length=0, trail_opacity=1.0, spotlight=9, classes=("car",))
    assert workspace.view_options({"classes": []}).classes is None


def test_tracks_are_ordered_longest_observed_first():
    tracks = {3: summary(3, observations=5), 1: summary(1, observations=5), 2: summary(2, observations=40)}
    assert workspace.track_order(tracks) == [2, 1, 3]


def test_intervals_text_lists_source_times_and_counts_the_rest():
    s = summary(intervals=((0, 3), (30, 33), (60, 63), (90, 93), (120, 123), (150, 153)))
    assert workspace.intervals_text(s, 30.0) == "0.0–0.1 s, 1.0–1.1 s, 2.0–2.1 s, 3.0–3.1 s and 2 more"


def test_contact_frames_never_fall_in_a_gap():
    s = summary(intervals=((0, 2), (50, 59)))
    frames = workspace.contact_frames(s, n=6)
    assert len(frames) == 6 and frames[0] == 0 and frames[-1] == 59
    assert all(f <= 2 or 50 <= f <= 59 for f in frames)
    assert workspace.contact_frames(summary(intervals=((4, 5),)), n=6) == [4, 5]


def test_crop_box_pads_and_stays_in_bounds():
    assert workspace.crop_box([10, 10, 30, 50], 64, 48) == (5, 0, 36, 48)
    assert workspace.crop_box([0, 0, 64, 48], 64, 48, pad=0) == (0, 0, 64, 48)


def test_captions():
    assert workspace.frame_caption(1064, 35_466_667, 1356) == "Source frame 1,064 of 1,355 · 35.47 s"
    assert workspace.track_label(summary(7, observations=1200)) == "#7 person · 1,200 frames"

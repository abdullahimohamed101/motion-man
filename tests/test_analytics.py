"""Temporal rules on synthetic tracks: MD§18 fixtures 1–4 and the edge cases around them (plan P3 step 2, D5–D8).

Frame 1000×1000 px at 25 fps (40 ms per frame). The line runs left to right at y = 500 from x = 100 to x = 900,
so "positive" (left of A→B on screen) is above it and moving down is POSITIVE_TO_NEGATIVE.
"""

import pytest

from motionman import analytics, regions
from motionman.analytics import Obs

W = H = 1000
DT = 40_000
LINE = regions.Region("door", "Door", "line", (0.1, 0.5, 0.9, 0.5))
ZONE = regions.Region("kerb", "Kerb", "rect", (0.4, 0.4, 0.6, 0.6))  # 400..600 px both ways


def run(region_list, *paths, params=None, finish=True):
    """Each path is one track (IDs 1, 2, ...): per frame an (x, y) centre, or None when not observed."""
    engine = analytics.Engine(region_list, W, H, params)
    events = []
    for f in range(max(len(p) for p in paths)):
        objs = [Obs(i + 1, "person", *p[f]) for i, p in enumerate(paths) if f < len(p) and p[f] is not None]
        events += engine.feed(f, f * DT, objs)
    if finish:
        engine.finish(max(len(p) for p in paths) * DT)
    return events, engine


def down(y0, y1, step=20, x=500):
    return [(x, y) for y in range(y0, y1 + (step if y1 >= y0 else -step), step if y1 >= y0 else -step)]


def hold(point, n):
    return [point] * n


def types(events):
    return [(e.type, e.track_id, e.direction) for e in events]


def test_one_crossing_gives_one_directional_event():
    path = down(400, 600) + hold((500, 600), 5)
    events, engine = run([LINE], path)
    assert types(events) == [("LINE_CROSSING", 1, "POSITIVE_TO_NEGATIVE")]
    e = events[0]
    assert e.evidence_us[0] == 4 * DT  # the last point above the line (y=480) starts the evidence
    assert e.evidence_us[1] == e.confirming_t_us and e.confirming_t_us - 6 * DT >= 100_000  # stable from y=520
    assert engine.stats["door"].crossings == {"NEGATIVE_TO_POSITIVE": 0, "POSITIVE_TO_NEGATIVE": 1}


def test_the_opposite_direction_is_named_the_other_way():
    events, _ = run([LINE], down(600, 400) + hold((500, 400), 5))
    assert types(events) == [("LINE_CROSSING", 1, "NEGATIVE_TO_POSITIVE")]


def test_a_centre_exactly_on_the_line_still_crosses():
    path = [(500, 460), (500, 480), (500, 500), (500, 520), (500, 540)] + hold((500, 540), 5)
    assert types(run([LINE], path)[0]) == [("LINE_CROSSING", 1, "POSITIVE_TO_NEGATIVE")]


def test_jitter_inside_the_deadband_gives_no_event():
    path = hold((500, 400), 3) + [(500, 499), (500, 501)] * 10 + hold((500, 400), 3)
    assert run([LINE], path)[0] == []


def test_touching_the_line_and_going_back_gives_no_event():
    path = down(400, 500) + down(500, 400) + hold((500, 400), 3)
    assert run([LINE], path)[0] == []


def test_a_short_excursion_that_is_not_confirmed_gives_no_event():
    path = down(400, 480) + [(500, 520), (500, 520)] + down(480, 400) + hold((500, 400), 3)  # 2 frames < 100 ms
    assert run([LINE], path)[0] == []


def test_passing_beyond_an_endpoint_is_not_a_crossing_and_does_not_poison_the_next():
    around = [(950, y) for y in range(400, 620, 20)] + hold((950, 600), 3)  # x = 950 is past B (x = 900)
    back = [(x, 600) for x in range(950, 490, -20)] + down(600, 400) + hold((500, 400), 5)
    events, _ = run([LINE], around + back)
    assert types(events) == [("LINE_CROSSING", 1, "NEGATIVE_TO_POSITIVE")]  # only the genuine return


def test_a_genuine_return_after_rearming_counts_again():
    path = down(400, 600) + hold((500, 600), 5) + down(600, 400) + hold((500, 400), 5)
    events, engine = run([LINE], path)
    assert types(events) == [("LINE_CROSSING", 1, "POSITIVE_TO_NEGATIVE"), ("LINE_CROSSING", 1, "NEGATIVE_TO_POSITIVE")]
    assert sum(engine.stats["door"].crossings.values()) == 2


def test_a_gap_across_the_line_invents_no_crossing():
    path = hold((500, 400), 3) + [None] * 2 + hold((500, 600), 6)  # seen above, missing, seen below
    assert run([LINE], path)[0] == []


def test_a_break_while_pending_forgets_the_crossing():
    path = down(400, 520) + [None] + hold((500, 600), 6)
    assert run([LINE], path)[0] == []


def test_a_long_time_gap_breaks_continuity_even_without_missing_frames():
    engine = analytics.Engine([LINE], W, H)
    events = engine.feed(0, 0, [Obs(1, "person", 500, 400)])
    events += engine.feed(1, 600_000, [Obs(1, "person", 500, 600)])  # one frame, but 600 ms later
    for f in range(2, 8):
        events += engine.feed(f, 600_000 + (f - 1) * DT, [Obs(1, "person", 500, 600)])
    assert events == []


def test_unassigned_detections_never_enter_rules():
    engine = analytics.Engine([LINE, ZONE], W, H)
    events = []
    for f, (x, y) in enumerate(down(300, 700)):
        events += engine.feed(f, f * DT, [Obs(None, "person", x, y)])
    assert events == [] and engine.stats["kerb"].peak_occupancy == 0 and engine.stats["kerb"].peak_unassigned == 1


def test_the_class_filter_applies_before_everything():
    cars = regions.Region("door", "Door", "line", LINE.coords, classes=("car",))
    assert run([cars], down(400, 600) + hold((500, 600), 5))[0] == []


def test_frames_must_be_fed_in_order():
    engine = analytics.Engine([LINE], W, H)
    engine.feed(5, 5 * DT, [])
    with pytest.raises(ValueError, match="in order"):
        engine.feed(5, 5 * DT, [])


# --- zones ---


def walk_in_and_out():
    return [(x, 500) for x in range(300, 520, 20)] + hold((500, 500), 10) + [(x, 500) for x in range(500, 720, 20)]


def test_walking_in_and_out_gives_one_entry_one_exit_and_one_visit():
    events, engine = run([ZONE], walk_in_and_out() + hold((700, 500), 5))
    assert [e.type for e in events] == ["ZONE_ENTRY", "ZONE_EXIT"]
    s = engine.stats["kerb"]
    assert (s.entries, s.exits, s.peak_occupancy) == (1, 1, 1)
    # Inside, edge-inclusive: x = 400..500 (6) + the 10-frame hold + x = 500..600 (6) = 22 centres, 21 intervals
    assert len(s.visits_us) == 1 and s.visits_us[0] == 21 * DT


def test_starting_inside_is_not_an_entry_and_disappearing_is_not_an_exit():
    events, engine = run([ZONE], hold((500, 500), 10) + [None] * 3)
    assert events == [] and engine.stats["kerb"].visits_us == [9 * DT]


def test_reappearing_inside_after_a_gap_is_not_an_entry():
    path = [(x, 500) for x in range(300, 520, 20)] + hold((500, 500), 5) + [None] * 2 + hold((500, 500), 5)
    events, engine = run([ZONE], path)
    assert [e.type for e in events] == ["ZONE_ENTRY"]  # once, before the gap
    assert len(engine.stats["kerb"].visits_us) == 2  # the gap ends the visit; dwell never bridges it


def test_edge_jitter_gives_no_extra_entries_or_exits():
    path = hold((300, 500), 3) + [(399, 500), (401, 500)] * 10 + hold((300, 500), 3)
    events, engine = run([ZONE], path)
    assert events == [] and engine.stats["kerb"].peak_occupancy == 1  # the edge counts as inside for occupancy


def test_occupancy_counts_distinct_tracks_and_is_time_weighted():
    a = hold((500, 500), 10)
    b = [None] * 5 + hold((450, 450), 5)
    _, engine = run([ZONE], a, b)
    s = engine.stats["kerb"]
    assert (s.peak_occupancy, s.peak_frame) == (2, 5)
    assert s.occupancy_us == 5 * DT * 1 + 5 * DT * 2 and s.covered_us == 10 * DT


def test_dwell_threshold_fires_once_per_visit():
    zone = regions.Region("kerb", "Kerb", "rect", ZONE.coords, dwell_threshold_s=0.2)
    path = hold((500, 500), 20) + hold((800, 500), 6) + hold((500, 500), 20)  # out and back in: two visits
    events, _ = run([zone], path)
    dwell = [e for e in events if e.type == "DWELL_THRESHOLD"]
    assert len(dwell) == 2 and all(e.value >= 200_000 for e in dwell)
    assert dwell[0].confirming_frame == 5 and dwell[0].evidence_us == (0, 5 * DT)


def test_occupancy_threshold_fires_on_upward_transitions_only():
    zone = regions.Region("kerb", "Kerb", "rect", ZONE.coords, occupancy_threshold=2)
    a = hold((500, 500), 30)
    b = [None] * 5 + hold((450, 450), 10) + [None] * 5 + hold((450, 450), 10)
    events, _ = run([zone], a, b)
    occ = [e for e in events if e.type == "OCCUPANCY_THRESHOLD"]
    assert [(e.confirming_frame, e.value, e.track_id) for e in occ] == [(5, 2, None), (20, 2, None)]


def test_events_in_one_frame_are_ordered_by_rule_then_track():
    zone_a = regions.Region("a-zone", "A", "rect", ZONE.coords)
    zone_b = regions.Region("b-zone", "B", "rect", ZONE.coords)
    path = [(x, 500) for x in range(300, 520, 20)] + hold((500, 500), 5)
    events, _ = run([zone_b, zone_a], path, path)
    assert [(e.rule_id, e.track_id) for e in events] == [("a-zone", 1), ("a-zone", 2), ("b-zone", 1), ("b-zone", 2)]


def test_same_input_gives_identical_events():
    path = down(400, 600) + hold((500, 600), 5) + down(600, 400) + hold((500, 400), 5)
    assert run([LINE, ZONE], path)[0] == run([LINE, ZONE], path)[0]

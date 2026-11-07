"""Tracker replay fixtures (plan native-g2 step 2, N2-3, spec C§8). Checks the committed fixtures without loading
Ultralytics (INV-8); `python tools/tracker_replay/reference.py --check` reruns the reference itself."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "tracker_replay"
FIXTURES = ROOT / "tests" / "fixtures" / "tracker"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"tracker_replay_{name}", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


fixtures = _load("fixtures")
scenarios = _load("scenarios")
NAMES = [s.name for s in scenarios.ALL]


def doc(name):
    return fixtures.load(FIXTURES / f"{name}.json")


def ids(name):
    return [sorted({int(r[4]) for r in f["tracks"]}) for f in doc(name)["frames"]]


def test_every_scenario_has_a_fixture_and_nothing_else():
    assert sorted(p.stem for p in FIXTURES.glob("*.json")) == sorted(NAMES)


@pytest.mark.parametrize("scenario", scenarios.ALL, ids=NAMES)
def test_fixture_inputs_match_their_scenario_and_pinned_reference(scenario):
    d = doc(scenario.name)
    assert [f["dets"] for f in d["frames"]] == scenario.frames()  # the generator was not changed silently
    assert d["reference"]["ultralytics"] == "8.4.196" and d["reference"]["lap"] == "0.5.13"
    assert d["class_policy"] == "agnostic" and d["instances"] == scenario.instances
    assert d["config"]["track_buffer"] == 30 and d["config"]["fuse_score"] is True


def test_scenario_values_are_exact_in_float32():
    np = pytest.importorskip("numpy")
    for s in scenarios.ALL:
        for frame in s.frames():
            for row in frame:
                assert list(np.asarray(row, dtype=np.float32).astype(float)) == row, s.name


def test_fixtures_show_the_behaviour_their_names_claim():
    assert ids("entry_exit")[3] == [] and ids("entry_exit")[4] == [1]  # a later start is confirmed next frame
    assert set(map(tuple, ids("reappear_within_buffer"))) == {(), (1,)}  # same ID after a 10-frame gap
    assert ids("reappear_after_buffer")[9] == [1] and ids("reappear_after_buffer")[-1] == [2]
    assert all(i == [1] for i in ids("low_conf_recovery"))  # low-band frames kept by the second stage
    floor = ids("low_conf_below_floor")
    assert floor[10:16] == [[]] * 6 and floor[16] == [1]  # lost below track_low_thresh, then recovered
    conflict = doc("class_conflict")["frames"]
    assert conflict[5]["tracks"][0][6] == 2.0 and conflict[15]["tracks"][0][6] == 0.0  # car track becomes person
    assert ids("class_conflict")[15] == [1]  # the agnostic reference: why native defaults to `aware` (ADR-006)
    assert all(i == [] for i in ids("new_track_threshold")[:7]) and ids("new_track_threshold")[7] == [5]
    assert all(len(f["tracks"]) == 1 for f in doc("degenerate_boxes")["frames"])  # zero-width box dropped
    assert len({i for f in ids("crowd") for i in f}) == 8
    assert ids("low_conf_jump")[10] == [1] and doc("low_conf_jump")["frames"][10]["tracks"][0][5] < 0.25


def test_canonical_ids_follow_first_appearance():
    frames = [
        {"tracks": [[0, 0, 1, 1, 7, 0.9, 0, 0]]},
        {"tracks": [[0, 0, 1, 1, 3, 0.9, 0, 0], [0, 0, 1, 1, 7, 0.9, 0, 1]]},
    ]
    assert fixtures.canonical_ids(frames) == [[1], [2, 1]]


def test_compare_accepts_renamed_ids_and_small_box_error_only():
    frames = doc("crossing")["frames"]
    renamed = [{**f, "tracks": [[*r[:4], {1: 11, 2: 12}[int(r[4])], *r[5:]] for r in f["tracks"]]} for f in frames]
    assert fixtures.compare(frames, renamed) == []
    assert fixtures.compare(frames, renamed, rename=False)  # raw IDs differ
    nudged = [{**f, "tracks": [[r[0] + 5e-4, *r[1:]] for r in f["tracks"]]} for f in frames]
    assert fixtures.compare(frames, nudged) == []
    moved = [{**f, "tracks": [[r[0] + 0.01, *r[1:]] for r in f["tracks"]]} for f in frames]
    assert "box off by" in fixtures.compare(frames, moved)[0]


def test_compare_catches_a_swap_a_missing_track_and_a_flag():
    frames = doc("crossing")["frames"]
    swapped = [dict(f) for f in frames]
    a, b = frames[20]["tracks"]
    swapped[20] = {**frames[20], "tracks": [[*a[:4], b[4], *a[5:]], [*b[:4], a[4], *b[5:]]]}
    assert any("id" in p for p in fixtures.compare(frames, swapped))
    dropped = [dict(f) for f in frames]
    dropped[5] = {**frames[5], "tracks": frames[5]["tracks"][:1]}
    assert "output detections" in fixtures.compare(frames, dropped)[0]
    flagged = [dict(f) for f in frames]
    flagged[3] = {**frames[3], "unconfirmed": not frames[3]["unconfirmed"]}
    assert "unconfirmed" in fixtures.compare(frames, flagged)[0]
    assert fixtures.compare(frames, frames[:-1]) == [f"{len(frames) - 1} frames, expected {len(frames)}"]

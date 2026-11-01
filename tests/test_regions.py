"""Region sets: validation with named refusals, content-addressed storage per video (plan P3 step 1, D2)."""

import json

import pytest

from motionman import regions, store

INPUT = "sha256:" + "ab" * 32
OTHER = "sha256:" + "cd" * 32


def doc(*items, name="Crossing", input_id=INPUT):
    return {"schema_version": "1.0", "input_id": input_id, "name": name, "regions": list(items)}


LINE = {"id": "entrance", "name": "Entrance", "kind": "line", "points": [[0.1, 0.5], [0.9, 0.5]], "classes": None}
RECT = {"id": "kerb", "name": "Kerb", "kind": "rect", "box": [0.0, 0.6, 0.3, 1.0], "classes": ["person"],
        "dwell_threshold_s": 2.5, "occupancy_threshold": 3}  # fmt: skip


@pytest.fixture
def root(tmp_path):
    return store.data_root(tmp_path / "data")


def test_a_valid_set_parses_and_round_trips(root):
    rs = regions.parse(doc(LINE, RECT))
    line, rect = rs.regions
    assert line.coords == (0.1, 0.5, 0.9, 0.5) and line.classes is None
    assert rect.classes == ("person",) and rect.dwell_threshold_s == 2.5 and rect.occupancy_threshold == 3
    assert regions.parse(rs.to_dict()) == rs
    assert line.to_px(1920, 1080) == pytest.approx((192, 540, 1728, 540))


def test_identical_content_gets_the_same_id_and_one_file(root):
    a, b = regions.parse(doc(LINE, RECT)), regions.parse(json.loads(json.dumps(doc(LINE, RECT))))
    assert a.region_set_id == b.region_set_id and a.region_set_id.startswith("rs_")
    assert regions.save(root, a) == regions.save(root, b)
    assert len(list(regions.folder(root, INPUT).iterdir())) == 1
    moved = regions.parse(doc({**LINE, "points": [[0.1, 0.5], [0.9, 0.55]]}, RECT))
    assert moved.region_set_id != a.region_set_id


def test_rounding_keeps_ids_stable_for_float_noise():
    noisy = regions.parse(doc({**LINE, "points": [[0.1 + 1e-12, 0.5], [0.9, 0.5]]}))
    assert noisy.region_set_id == regions.parse(doc(LINE)).region_set_id


def test_saved_sets_load_by_id_and_list_per_video(root):
    rs = regions.parse(doc(LINE, name="b set"))
    other = regions.parse(doc(RECT, name="A set"))
    elsewhere = regions.parse(doc(LINE, input_id=OTHER))
    for s in (rs, other, elsewhere):
        regions.save(root, s)
    assert regions.load(root, INPUT, rs.region_set_id) == rs
    assert [s.name for s in regions.list_sets(root, INPUT)] == ["A set", "b set"]
    assert regions.list_sets(root, OTHER) == [elsewhere]
    with pytest.raises(regions.RegionError, match="no region set"):
        regions.load(root, OTHER, rs.region_set_id)


def test_hand_edited_or_unreadable_files_are_refused_or_skipped(root):
    rs = regions.parse(doc(LINE))
    path = regions.save(root, rs)
    body = json.loads(path.read_text())
    body["name"] = "edited"
    path.write_text(json.dumps(body))
    with pytest.raises(regions.RegionError, match="does not match its content"):
        regions.load(root, INPUT, rs.region_set_id)
    assert regions.list_sets(root, INPUT) == []  # the edited file no longer matches its name
    (path.parent / "rs_0000000000000000.json").write_text("{not json")
    assert regions.list_sets(root, INPUT) == []
    with pytest.raises(regions.RegionError, match="not a region set id"):
        regions.load(root, INPUT, "../escape")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("nope", "JSON object"),
        ({**doc(LINE), "extra": 1}, "unknown field"),
        ({**doc(LINE), "schema_version": "2.0"}, "unsupported schema_version"),
        (doc(LINE, input_id="clip.mp4"), "input_id"),
        (doc(LINE, name=" "), "name must be"),
        (doc(), "1 to 32 regions"),
        (doc(*[{**LINE, "id": f"l{i}"} for i in range(33)]), "1 to 32 regions"),
        (doc(LINE, LINE), "duplicate region id"),
        (doc({**LINE, "id": "Entrance!"}), "id must be"),
        (doc({**LINE, "kind": "polygon"}), "kind must be"),
        (doc({**LINE, "box": [0, 0, 1, 1]}), "unknown field"),
        (doc({**LINE, "points": [[0.5, 0.5], [0.5, 0.5]]}), "zero length"),
        (doc({**LINE, "points": [[0.1, 0.5]]}), "points must be"),
        (doc({**LINE, "points": [[-0.1, 0.5], [0.9, 0.5]]}), "from 0 to 1"),
        (doc({**LINE, "points": [[True, 0.5], [0.9, 0.5]]}), "from 0 to 1"),
        (doc({**LINE, "points": [[float("nan"), 0.5], [0.9, 0.5]]}), "from 0 to 1"),
        (doc({**RECT, "box": [0.3, 0.6, 0.3, 1.0]}), "positive area"),
        (doc({**RECT, "box": [0.3, 0.6, 0.1, 1.0]}), "positive area"),
        (doc({**RECT, "box": [0.1, 0.6, 0.3]}), "box must be"),
        (doc({**RECT, "dwell_threshold_s": 0}), "dwell_threshold_s"),
        (doc({**RECT, "occupancy_threshold": 1.5}), "occupancy_threshold"),
        (doc({**RECT, "occupancy_threshold": True}), "occupancy_threshold"),
        (doc({**RECT, "classes": []}), "classes must be"),
        (doc({**RECT, "classes": ["person", "person"]}), "listed twice"),
        (doc({**LINE, "dwell_threshold_s": 3}), "unknown field"),
    ],
)
def test_invalid_sets_are_refused_with_a_reason(body, message):
    with pytest.raises(regions.RegionError, match=message):
        regions.parse(body)


def test_load_file_reports_unreadable_files(tmp_path):
    bad = tmp_path / "regions.json"
    bad.write_text("{")
    with pytest.raises(regions.RegionError, match="can't read region set regions.json"):
        regions.load_file(bad)

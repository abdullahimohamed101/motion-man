import json

import pytest

from motionman.records import RecordError, frame_record, read_records, source_t_us, validate
from motionman.types import Track

FPS = 24000 / 1001  # the demo clip's rate


def track(tid=1, box=(10.0, 20.0, 50.0, 80.0), conf=0.912345):
    return Track(tid, 0, "person", conf, box)


def test_record_has_v1_fields_and_observation_ids():
    rec = frame_record("r_x", 3, FPS, 1920, 1080, [track(1), track(None, (100, 100, 200, 300))])
    assert rec["schema_version"] == "1.0" and rec["run_id"] == "r_x"
    assert (rec["source_frame_index"], rec["sample_index"], rec["frame_status"]) == (3, 3, "INFERRED")
    assert (rec["source_width"], rec["source_height"]) == (1920, 1080)
    assert [o["observation_id"] for o in rec["objects"]] == ["3:0", "3:1"]
    assert rec["objects"][1]["track_id"] is None  # never invented (INV-2)
    assert rec["objects"][0]["confidence"] == 0.9123 and rec["objects"][0]["mask_ref"] is None
    validate(rec)
    json.dumps(rec)


def test_source_time_is_integer_microseconds_from_index_and_fps():
    assert source_t_us(0, FPS) == 0
    assert source_t_us(1, FPS) == 41708  # 1001/24000 s
    assert source_t_us(1355, FPS) == 56514792
    assert isinstance(source_t_us(7, 30.0), int)


def test_empty_frame_is_a_record_with_no_objects():
    rec = frame_record("r_x", 0, FPS, 64, 48, [])
    assert rec["objects"] == []
    validate(rec)


def test_small_overrun_is_clipped_to_the_image():
    rec = frame_record("r_x", 0, 30.0, 64, 48, [track(box=(-0.3, 0.0, 64.4, 48.2))])
    assert rec["objects"][0]["bbox_xyxy"] == [0.0, 0.0, 64.0, 48.0]


@pytest.mark.parametrize(
    "box",
    [(-5.0, 0, 10, 10), (0, 0, 70, 10), (10, 10, 10, 20), (10, 20, 30, 20), (float("nan"), 0, 5, 5), (0, 0, 5)],
)
def test_malformed_boxes_are_rejected(box):
    with pytest.raises(RecordError):
        frame_record("r_x", 0, 30.0, 64, 48, [track(box=box)])


def good_record():
    return frame_record("r_x", 0, 30.0, 64, 48, [track(box=(1, 1, 10, 10))])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": "2.0"}, "unsupported schema_version"),
        ({"frame_status": "GUESSED"}, "unknown frame_status"),
        ({"source_t_us": -1}, "source_t_us"),
        ({"source_frame_index": True}, "source_frame_index"),
        ({"objects": "none"}, "objects must be a list"),
    ],
)
def test_validate_rejects_bad_records(change, message):
    with pytest.raises(RecordError, match=message):
        validate({**good_record(), **change})


def test_validate_rejects_duplicate_observation_ids_and_bool_track_ids():
    rec = good_record()
    rec["objects"].append(dict(rec["objects"][0]))
    with pytest.raises(RecordError, match="unique"):
        validate(rec)
    rec = good_record()
    rec["objects"][0]["track_id"] = True
    with pytest.raises(RecordError, match="track_id"):
        validate(rec)


def write_lines(path, lines):
    path.write_text("".join((line if isinstance(line, str) else json.dumps(line)) + "\n" for line in lines))
    return path


# Shape copied from a real MVP output (runs/demo-recording/tracks.jsonl, first line).
LEGACY_LINE = {
    "frame_index": 0,
    "timestamp_s": 0.0,
    "objects": [
        {
            "track_id": 1,
            "class_id": 2,
            "class_name": "car",
            "confidence": 0.8962,
            "bbox_xyxy": [1.21, 588.73, 519.43, 850.47],
        },
        {"track_id": None, "class_id": 0, "class_name": "person", "confidence": 0.31, "bbox_xyxy": [5, 5, 9, 9]},
    ],
}


def test_reads_v1_records_back_unchanged(tmp_path):
    recs = [frame_record("r_x", i, FPS, 1920, 1080, [track()]) for i in range(3)]
    assert list(read_records(write_lines(tmp_path / "t.jsonl", recs))) == recs


def test_legacy_lines_are_adapted_without_inventing_data(tmp_path):
    second = {**LEGACY_LINE, "frame_index": 1, "timestamp_s": 0.041708}
    out = list(read_records(write_lines(tmp_path / "t.jsonl", [LEGACY_LINE, second])))
    assert out[0]["legacy"] is True and out[0]["schema_version"] == "1.0"
    assert out[0]["run_id"] is None and out[0]["source_width"] is None  # not recorded by the MVP, so unknown
    assert [r["source_t_us"] for r in out] == [0, 41708]
    assert [o["observation_id"] for o in out[0]["objects"]] == ["0:0", "0:1"]
    assert out[0]["objects"][1]["track_id"] is None


def test_truncated_tail_fails_clearly(tmp_path):
    path = write_lines(tmp_path / "t.jsonl", [good_record(), '{"schema_version": "1.0", "run_'])
    with pytest.raises(RecordError, match=r"t.jsonl:2: .*truncated"):
        list(read_records(path))


def test_unknown_version_and_format_fail_clearly(tmp_path):
    with pytest.raises(RecordError, match=r":1: unsupported schema_version"):
        list(read_records(write_lines(tmp_path / "a.jsonl", [{**good_record(), "schema_version": "9"}])))
    with pytest.raises(RecordError, match="unrecognised record format"):
        list(read_records(write_lines(tmp_path / "b.jsonl", [{"hello": 1}])))

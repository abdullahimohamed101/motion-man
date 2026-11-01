"""Records index and track summaries (plan P1 step 2, D3)."""

import json

import pytest

from motionman import inspection, records, store
from motionman.types import Track

W, H = 64, 48


def track(tid, cls="person", conf=0.9, x=1.0):
    return Track(tid, 0, cls, conf, (x, 1.0, x + 5, 6.0))


def write_records(path, frames):
    """frames: list of track lists, one per frame index."""
    path.write_text(
        "".join(json.dumps(records.frame_record("r_t", i, 30.0, W, H, ts)) + "\n" for i, ts in enumerate(frames))
    )
    return path


@pytest.fixture
def root(tmp_path):
    return store.data_root(tmp_path / "data")


def scripted():
    # Track 1 frames 0-4 and 8-9 (gap 5-7, class flips once); track 2 only at frame 3; unassigned at 1 and 2.
    frames = [[] for _ in range(10)]
    for f in [0, 1, 2, 3, 4, 8, 9]:
        frames[f].append(track(1, "car" if f == 2 else "person", conf=0.5 + f * 0.05))
    frames[3].append(track(2, conf=0.4))
    frames[1].append(track(None))
    frames[2].append(track(None))
    return frames


def test_track_summaries_report_intervals_gaps_classes_and_confidence(tmp_path, root):
    d = inspection.load_or_build(write_records(tmp_path / "t.jsonl", scripted()), root)
    t1 = d.tracks[1]
    assert (t1.first_frame, t1.last_frame, t1.observations) == (0, 9, 7)
    assert t1.intervals == [(0, 4), (8, 9)] and t1.gaps == [(5, 7)]  # span is not continuous presence
    assert t1.class_name == "person" and t1.class_counts == {"person": 6, "car": 1}
    assert t1.first_t_us == 0 and t1.last_t_us == records.source_t_us(9, 30.0)
    assert t1.confidence_p10 < t1.confidence_p50 < t1.confidence_p90
    assert d.tracks[2].intervals == [(3, 3)] and d.tracks[2].gaps == []
    assert d.unassigned_observations == 2 and None not in d.tracks  # null IDs never form a track


def test_random_access_matches_sequential_reading_across_index_blocks(tmp_path, root):
    frames = [[track(i % 7)] if i % 5 else [] for i in range(300)]  # 300 records: three index blocks
    path = write_records(tmp_path / "t.jsonl", frames)
    d = inspection.load_or_build(path, root)
    assert len(d.index.offsets) == 3 and d.index.count == 300
    sequential = list(records.read_records(path))
    for f in [0, 1, 127, 128, 129, 255, 256, 299]:
        assert d.index.record(f) == sequential[f]
    assert list(d.index.records(120, 140)) == sequential[120:140]


def test_out_of_range_frames_are_refused(tmp_path, root):
    d = inspection.load_or_build(write_records(tmp_path / "t.jsonl", [[]] * 5), root)
    with pytest.raises(inspection.InspectError, match="outside this run"):
        d.index.record(5)


def test_cache_is_reused_then_rebuilt_when_the_records_change(tmp_path, root):
    path = write_records(tmp_path / "t.jsonl", scripted())
    first = inspection.load_or_build(path, root)
    stamp = (first.folder / "index.json").stat().st_mtime_ns
    again = inspection.load_or_build(path, root)
    assert again.folder == first.folder and (again.folder / "index.json").stat().st_mtime_ns == stamp  # reused
    frames = scripted()
    frames[0].append(track(3))
    write_records(path, frames)
    rebuilt = inspection.load_or_build(path, root)
    assert rebuilt.folder != first.folder and 3 in rebuilt.tracks  # keyed by the new records hash


def test_a_corrupt_or_foreign_cache_is_rebuilt_not_trusted(tmp_path, root):
    path = write_records(tmp_path / "t.jsonl", scripted())
    d = inspection.load_or_build(path, root)
    (d.folder / "index.json").write_text("{not json")
    assert inspection.load_or_build(path, root).index.count == 10
    meta = json.loads((d.folder / "tracks.json").read_text())
    (d.folder / "tracks.json").write_text(json.dumps({**meta, "records_sha256": "sha256:other"}))
    assert inspection.load_or_build(path, root).tracks[1].intervals == [(0, 4), (8, 9)]


def test_records_that_differ_from_the_runs_recorded_hash_are_refused(tmp_path, root):
    path = write_records(tmp_path / "t.jsonl", scripted())
    with pytest.raises(inspection.InspectError, match="does not match the hash the run recorded"):
        inspection.load_or_build(path, root, records_sha256="sha256:" + "0" * 64)


def test_out_of_order_records_are_refused(tmp_path, root):
    path = tmp_path / "t.jsonl"
    recs = [records.frame_record("r_t", i, 30.0, W, H, []) for i in (0, 2, 1)]
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    with pytest.raises(inspection.InspectError, match="line 2 is frame 2"):
        inspection.load_or_build(path, root)


def test_legacy_records_are_indexed_through_the_adapter(tmp_path, root):
    path = tmp_path / "legacy.jsonl"
    path.write_text(
        "".join(
            json.dumps({"frame_index": i, "timestamp_s": round(i / 30, 6),
                        "objects": [{"track_id": 4, "class_id": 0, "class_name": "person", "confidence": 0.8,
                                     "bbox_xyxy": [1, 1, 5, 5]}] if i != 2 else []}) + "\n"
            for i in range(5)
        )
    )  # fmt: skip
    d = inspection.load_or_build(path, root)
    assert d.tracks[4].intervals == [(0, 1), (3, 4)] and d.index.record(3)["legacy"] is True

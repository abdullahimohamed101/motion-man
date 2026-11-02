"""Run comparison by position only (plan P4 step 3, D9; MD fixture 7): swapped track IDs still match, class
disagreements and unmatched boxes are reported, only common frames are compared, results publish and reuse."""

import pytest

from motionman import analysis, datasets, disagree, pipeline, store
from motionman.types import RunConfig, Track

PERSON, CAR, OTHER = (5.0, 5.0, 20.0, 30.0), (30.0, 10.0, 55.0, 30.0), (2.0, 35.0, 12.0, 45.0)


def tracker(*tracks):
    class T:
        def __init__(self, config):
            pass

        def update(self, image):
            return list(tracks)

    return T


A = tracker(Track(1, 0, "person", 0.9, PERSON), Track(2, 2, "car", 0.8, CAR))
SWAPPED = tracker(Track(2, 0, "person", 0.7, PERSON), Track(1, 2, "car", 0.6, CAR))  # same boxes, IDs swapped
FLIPPED = tracker(Track(1, 0, "person", 0.9, PERSON), Track(2, 7, "truck", 0.8, CAR))
MISSING = tracker(Track(1, 0, "person", 0.9, PERSON), Track(5, 0, "person", 0.5, OTHER))


@pytest.fixture
def root(tmp_path):
    return store.data_root(tmp_path / "data")


def publish(clip, root, make, max_frames=None):
    report = pipeline.run(RunConfig(clip, None, data_root=root.path, max_frames=max_frames), make_tracker=make,
                          resolve=lambda d: "cpu")  # fmt: skip
    return analysis.find_run(root, report["run_id"])


def test_swapped_track_ids_with_the_same_boxes_match_spatially(clip, root):
    res = disagree.compare(publish(clip, root, A), publish(clip, root, SWAPPED), root)
    c = res.doc["counts"]
    assert c == {"matched_same_class": 40, "class_disagreement": 0, "unmatched_a": 0, "unmatched_b": 0}
    assert res.doc["frames_compared"] == 20 and res.doc["frames_with_differences"] == 0
    assert list(disagree.read_items(res)) == [] and res.doc["matched_iou"]["p50"] == 1.0
    assert "never compared" in res.doc["note"]


def test_class_disagreements_and_unmatched_boxes_are_reported_both_ways(clip, root):
    a = publish(clip, root, A)
    flip = disagree.compare(a, publish(clip, root, FLIPPED), root)
    assert flip.doc["counts"]["class_disagreement"] == 20 and flip.doc["counts"]["matched_same_class"] == 20
    item = next(disagree.read_items(flip))
    assert (item["kind"], item["a"]["class_name"], item["b"]["class_name"], item["iou"]) == (
        "class_disagreement", "car", "truck", 1.0)  # fmt: skip
    miss = disagree.compare(a, publish(clip, root, MISSING), root)
    c = miss.doc["counts"]
    assert (c["matched_same_class"], c["unmatched_a"], c["unmatched_b"]) == (20, 20, 20)
    assert miss.doc["by_class"]["car"]["unmatched_a"] == 20 and miss.doc["by_class"]["person"]["unmatched_b"] == 20
    assert miss.doc["top_frames"][0] == {"frame": 0, "differences": 2}  # where to look first


def test_one_box_matches_at_most_one_box_best_iou_first():
    a = [{"observation_id": "0:0", "class_name": "person", "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]}]
    b = [{"observation_id": "0:0", "class_name": "person", "confidence": 0.9, "bbox_xyxy": [1, 0, 11, 10]},
         {"observation_id": "0:1", "class_name": "person", "confidence": 0.9, "bbox_xyxy": [0, 0, 10, 10]}]  # fmt: skip
    kinds = [(k, ob["observation_id"] if ob else None) for k, _, ob, _ in disagree.match_frame(a, b, 0.5)]
    assert kinds == [("matched_same_class", "0:1"), ("unmatched_b", "0:0")]  # the exact box wins; the other is extra


def test_only_frames_both_runs_processed_are_compared(clip, root):
    res = disagree.compare(publish(clip, root, A), publish(clip, root, A, max_frames=8), root)
    assert res.doc["frames_compared"] == 8 and res.doc["counts"]["matched_same_class"] == 16


def test_identical_comparisons_reuse_the_result_and_runs_must_share_a_video(clip, root, tmp_path):
    a, b = publish(clip, root, A), publish(clip, root, SWAPPED)
    first = disagree.compare(a, b, root)
    again = disagree.compare(a, b, root)
    assert again.reused and again.doc == first.doc and disagree.load(root, first.disagreement_id).doc == first.doc
    assert disagree.compare(a, b, root, iou_threshold=0.9).disagreement_id != first.disagreement_id
    import dataclasses

    other = dataclasses.replace(b, manifest={**b.manifest, "input_id": "sha256:" + "0" * 64})
    with pytest.raises(disagree.DisagreeError, match="different videos"):
        disagree.compare(a, other, root)
    with pytest.raises(disagree.DisagreeError, match="legacy run"):
        disagree.compare(a, dataclasses.replace(b, manifest=None), root)
    with b.records_path.open("ab") as f:
        f.write(b"\n")
    with pytest.raises(disagree.DisagreeError, match="changed after the run was verified"):
        disagree.compare(a, b, root, iou_threshold=0.7)


def test_sending_differences_to_review_only_selects_frames(clip, root):
    a, b = publish(clip, root, A), publish(clip, root, MISSING)
    res = disagree.compare(a, b, root)
    d = datasets.new_draft(root, "crosswalk-eval", a, [0])
    datasets.select_frames(d, [t["frame"] for t in res.doc["top_frames"][:3]])
    assert d["selected"] == [0, 1, 2] and d["frames"] == [] and d["annotations"] == []

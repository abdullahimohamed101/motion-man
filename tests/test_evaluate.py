"""Detection evaluation (plan P4 step 2, D6–D8): exact matching rules, scope, ignored areas, null rates with
reasons, tie-breaks, injected errors that move the counts exactly, refusals, and a published, reusable result."""

import pytest

from motionman import analysis, datasets, evaluate, pipeline, store
from motionman.types import RunConfig

from .test_pipeline import FakeTracker

BOX = [10.0, 10.0, 30.0, 40.0]


def obj(n, cls, box, conf=0.9, tid=1):
    return {"observation_id": f"0:{n}", "track_id": tid, "class_name": cls, "confidence": conf, "bbox_xyxy": box}


def rec(*objs, frame=0):
    return {"source_frame_index": frame, "objects": list(objs)}


def ann(n, cls, box, frame=0):
    return {"annotation_id": f"a{n}", "frame": frame, "class_name": cls, "bbox_xyxy": box}


def revision(frames, anns):
    return {"split": "eval", "frames": frames, "annotations": anns}


def reviewed(frame=0, classes=("person",), ignored=()):
    return {"frame": frame, "status": "reviewed", "classes_complete": list(classes), "ignored": list(ignored)}


def run_score(records_, frames, anns, conf=0.25, iou=0.5):
    return evaluate.score({r["source_frame_index"]: r for r in records_}, revision(frames, anns), conf, iou)


def counts(result, cls="person"):
    c = result["classes"][cls]
    return c["tp"], c["fp"], c["fn"]


def test_an_exact_match_is_one_true_positive():
    r = run_score([rec(obj(0, "person", BOX))], [reviewed()], [ann(1, "person", BOX)])
    assert counts(r) == (1, 0, 0) and r["classes"]["person"]["precision"] == 1.0
    assert r["classes"]["person"]["recall"] == 1.0 and r["matched_iou"]["p50"] == 1.0


def test_a_duplicate_prediction_is_a_false_positive():
    r = run_score([rec(obj(0, "person", BOX, 0.9), obj(1, "person", BOX, 0.8))], [reviewed()], [ann(1, "person", BOX)])
    assert counts(r) == (1, 1, 0) and r["examples"]["false_positives"][0]["observation_id"] == "0:1"


def test_undefined_rates_from_empty_positives_or_negatives_are_null_with_reasons_not_zeros():
    fp_only = run_score([rec(obj(0, "person", BOX))], [reviewed()], [])
    assert counts(fp_only) == (0, 1, 0) and fp_only["classes"]["person"]["precision"] == 0.0
    assert fp_only["classes"]["person"]["recall"] is None
    assert fp_only["classes"]["person"]["recall_reason"] == "no reviewed boxes of this class"
    nothing = run_score([rec()], [reviewed()], [])
    p = nothing["classes"]["person"]
    assert (p["precision"], p["precision_reason"]) == (None, "no predictions to score")
    assert p["recall"] is None and nothing["matched_iou"]["p50"] is None


def test_scope_only_complete_classes_on_reviewed_frames_are_scored():
    records_ = [rec(obj(0, "person", BOX), obj(1, "car", [40, 10, 60, 30])), rec(obj(0, "person", BOX), frame=1)]
    frames = [reviewed(0, classes=("person",)), {"frame": 1, "status": "excluded", "classes_complete": [],
                                                  "ignored": []}]  # fmt: skip
    anns = [ann(1, "person", BOX), ann(2, "car", [0, 0, 5, 5])]  # the car box isn't in frame 0's complete scope
    r = run_score(records_, frames, anns)
    assert list(r["classes"]) == ["person"] and counts(r) == (1, 0, 0)  # the car neither helps nor hurts
    assert r["frames"]["unscored_predictions"] == 1 and r["frames"]["unscored_annotations"] == 1
    assert (r["frames"]["reviewed"], r["frames"]["excluded"], r["frames"]["scored"]) == (1, 1, 1)


def test_predictions_centred_in_an_ignored_area_are_neither_true_nor_false():
    r = run_score([rec(obj(0, "person", [50, 0, 60, 10]))], [reviewed(ignored=[[45, 0, 64, 20]])], [])
    assert counts(r) == (0, 0, 0) and r["frames"]["ignored_predictions"] == 1


def test_reviewed_boxes_centred_in_an_ignored_area_are_not_counted_as_missed():
    box = [50, 0, 60, 10]
    r = run_score([rec(obj(0, "person", box))], [reviewed(ignored=[[45, 0, 64, 20]])], [ann(1, "person", box)])
    assert counts(r) == (0, 0, 0) and r["frames"]["ignored_annotations"] == 1


def test_injected_false_positive_and_missed_object_move_the_counts_exactly():
    gts = [ann(1, "person", BOX), ann(2, "person", [40, 10, 60, 40])]
    preds = [obj(0, "person", BOX), obj(1, "person", [40, 10, 60, 40])]
    base = run_score([rec(*preds)], [reviewed()], gts)
    assert counts(base) == (2, 0, 0)
    extra_fp = run_score([rec(*preds, obj(2, "person", [0, 50, 10, 60], 0.5))], [reviewed()], gts)
    assert counts(extra_fp) == (2, 1, 0) and extra_fp["classes"]["person"]["precision"] == round(2 / 3, 4)
    missed = run_score([rec(preds[0])], [reviewed()], gts)
    assert counts(missed) == (1, 0, 1) and missed["classes"]["person"]["recall"] == 0.5
    dropped_gt = run_score([rec(*preds)], [reviewed()], gts[:1])  # a reviewed box removed: its match becomes an FP
    assert counts(dropped_gt) == (1, 1, 0)


def test_higher_confidence_matches_first_and_equal_iou_goes_to_the_lower_annotation_id():
    a2, a10 = ann(2, "person", [0, 0, 10, 10]), ann(10, "person", [10, 0, 20, 10])
    pred = obj(0, "person", [5, 0, 15, 10])  # IoU 1/3 with both
    r = run_score([rec(pred)], [reviewed()], [a10, a2], iou=0.3)
    assert counts(r) == (1, 0, 1) and r["examples"]["false_negatives"][0]["annotation_id"] == "a10"
    strong, weak = obj(1, "person", [0, 0, 10, 12], 0.95), obj(0, "person", [0, 0, 10, 10], 0.6)
    r = run_score([rec(weak, strong)], [reviewed()], [ann(1, "person", [0, 0, 10, 10])])
    assert counts(r) == (1, 1, 0) and r["examples"]["false_positives"][0]["observation_id"] == "0:0"  # weak lost


def test_thresholds_are_applied():
    low = obj(0, "person", BOX, conf=0.3)
    assert counts(run_score([rec(low)], [reviewed()], [ann(1, "person", BOX)], conf=0.5)) == (0, 0, 1)
    shifted = obj(0, "person", [20, 10, 40, 40])  # IoU 1/3
    assert counts(run_score([rec(shifted)], [reviewed()], [ann(1, "person", BOX)], iou=0.5)) == (0, 1, 1)


def test_frames_the_run_never_processed_are_not_evaluated_rather_than_missed():
    r = run_score([rec()], [reviewed(0), reviewed(99)], [ann(1, "person", BOX, frame=99)])
    assert r["frames"]["not_evaluated"] == [99] and counts(r) == (0, 0, 0)


# --- with a real published run (fake tracker: person at (i, 5, i+10, 20) on frame i, none on frame 3) ---


@pytest.fixture
def root(tmp_path):
    return store.data_root(tmp_path / "data")


@pytest.fixture
def run(clip, root):
    report = pipeline.run(RunConfig(clip, None, data_root=root.path), make_tracker=FakeTracker, resolve=lambda d: "cpu")
    return analysis.find_run(root, report["run_id"])


def commit(run, root, split="eval", dataset_id="crosswalk-eval"):
    d = datasets.new_draft(root, dataset_id, run, [0, 10])
    datasets.set_frame(d, 0, "reviewed", ["person"])
    datasets.add_annotation(d, 0, "person", [0, 5, 10, 20], "accepted")
    datasets.set_frame(d, 10, "reviewed", ["person"])
    datasets.add_annotation(d, 10, "person", [10, 5, 20, 20], "accepted")
    datasets.add_annotation(d, 10, "person", [40, 25, 50, 40], "added", note="missed by the model")
    datasets.save_draft(root, d)
    return f"{dataset_id}@r{datasets.commit(root, dataset_id, split)[0]}"


def test_an_evaluation_names_its_revision_thresholds_and_evaluator(run, root):
    ref = commit(run, root)
    res = evaluate.evaluate(run, ref, root)
    d = res.doc
    assert (
        d["dataset"]["ref"] == ref and d["dataset"]["sha256"].startswith("sha256:") and d["dataset"]["split"] == "eval"
    )
    assert d["thresholds"] == {"confidence": 0.25, "iou": 0.5} and d["evaluator_version"] == "detection-v2"
    assert d["metric"] == "fixed-threshold detection precision/recall" and "Not mAP" in d["scope"]
    assert counts(d) == (2, 0, 1) and d["total"]["recall"] == round(2 / 3, 4) and d["total"]["precision"] == 1.0
    assert d["examples"]["false_negatives"][0]["frame"] == 10  # traceable to the frame


def test_identical_evaluations_reuse_the_published_result_and_settings_change_the_id(run, root):
    ref = commit(run, root)
    first = evaluate.evaluate(run, ref, root)
    again = evaluate.evaluate(run, ref, root)
    assert again.reused and again.evaluation_id == first.evaluation_id and again.doc == first.doc
    stricter = evaluate.evaluate(run, ref, root, iou_threshold=0.9)
    assert stricter.evaluation_id != first.evaluation_id
    assert evaluate.load(root, run.run_id, first.evaluation_id).doc == first.doc


def test_tuning_other_video_bad_thresholds_and_changed_records_are_refused(run, root):
    tuning = commit(run, root, split="tuning", dataset_id="tune")
    with pytest.raises(evaluate.EvaluationError, match="tuning revision"):
        evaluate.evaluate(run, tuning, root)
    ref = commit(run, root)
    with pytest.raises(evaluate.EvaluationError, match="run's own threshold"):
        evaluate.evaluate(run, ref, root, conf=0.1)
    with pytest.raises(evaluate.EvaluationError, match="IoU threshold"):
        evaluate.evaluate(run, ref, root, iou_threshold=0)
    import dataclasses

    other = dataclasses.replace(run, manifest={**run.manifest, "input_id": "sha256:" + "0" * 64})
    with pytest.raises(evaluate.EvaluationError, match="different video"):
        evaluate.evaluate(other, ref, root)
    with pytest.raises(evaluate.EvaluationError, match="px but this run"):
        evaluate.evaluate(dataclasses.replace(run, width=128), ref, root)
    with run.records_path.open("ab") as f:
        f.write(b"\n")
    with pytest.raises(evaluate.EvaluationError, match="changed after the run was verified"):
        evaluate.evaluate(run, ref, root)


def test_a_damaged_published_evaluation_is_rebuilt(run, root):
    ref = commit(run, root)
    res = evaluate.evaluate(run, ref, root)
    (res.folder / "evaluation.json").write_text("{}")
    with pytest.raises(evaluate.EvaluationError, match="missing or damaged"):
        evaluate.load(root, run.run_id, res.evaluation_id)
    rebuilt = evaluate.evaluate(run, ref, root)
    assert not rebuilt.reused and rebuilt.doc == res.doc

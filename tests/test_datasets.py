"""Reviewed datasets: drafts, immutable revisions, refusals, raw predictions untouched (plan P4 step 1, D1–D6).

The fake run is 20 frames of a 64×48 clip: one person (track 1) at bbox (i, 5, i+10, 20) on frame i, none on
frame 3, and an unassigned car on frame 5.
"""

import json
from datetime import UTC, datetime

import pytest

from motionman import analysis, datasets, inspection, pipeline, store
from motionman.types import RunConfig

from .test_pipeline import FakeTracker


@pytest.fixture
def root(tmp_path):
    return store.data_root(tmp_path / "data")


@pytest.fixture
def run(clip, root):
    report = pipeline.run(RunConfig(clip, None, data_root=root.path), make_tracker=FakeTracker, resolve=lambda d: "cpu")
    return analysis.find_run(root, report["run_id"])


def record(run, root, frame):
    return inspection.load_or_build(run.records_path, root, run.records_sha256).index.record(frame)


def reviewed_draft(run, root, dataset_id="crosswalk-eval"):
    """Frames 0 and 10 reviewed for person (one accepted, one corrected box), frame 5 excluded."""
    d = datasets.new_draft(root, dataset_id, run, [0, 5, 10])
    datasets.set_frame(d, 0, "reviewed", ["person"])
    datasets.accept_prediction(d, 0, record(run, root, 0)["objects"][0])
    datasets.set_frame(d, 5, "excluded", [])
    datasets.set_frame(d, 10, "reviewed", ["person", "car"], ignored=[[50, 0, 64, 10]])
    datasets.add_annotation(d, 10, "person", [9.5, 5, 21, 20], "corrected", note="box was 1 px off")
    datasets.save_draft(root, d)
    return d


def test_even_frames_selects_every_nth():
    assert datasets.even_frames(1356)[:3] == [0, 30, 60] and len(datasets.even_frames(1356)) == 46
    assert datasets.even_frames(20, 10) == [0, 10]


def test_a_draft_starts_unlabelled_and_names_its_video_and_run(run, root):
    d = datasets.new_draft(root, "crosswalk-eval", run, [0, 10, 10, 5])
    assert d["selected"] == [0, 5, 10] and d["frames"] == [] and d["annotations"] == []
    assert d["input_id"] == run.manifest["input_id"] and d["frame_size"] == [64, 48]
    assert datasets.load_draft(root, "crosswalk-eval") == d
    with pytest.raises(datasets.DatasetError, match="already exists"):
        datasets.new_draft(root, "crosswalk-eval", run, [0])
    with pytest.raises(datasets.DatasetError, match="within this run"):
        datasets.new_draft(root, "other", run, [0, 20])


def test_accept_copies_the_prediction_and_notes_where_it_came_from(run, root):
    d = reviewed_draft(run, root)
    a = d["annotations"][0]
    assert (a["annotation_id"], a["origin"], a["class_name"], a["bbox_xyxy"]) == ("a1", "accepted", "person",
                                                                                  [0.0, 5.0, 10.0, 20.0])  # fmt: skip
    assert run.run_id in a["note"] and "0:0" in a["note"]


def test_commit_writes_an_immutable_revision_and_the_raw_records_stay_unchanged(run, root):
    before = {p.name: store.sha256_file(p) for p in run.summary.path.iterdir()}
    reviewed_draft(run, root)
    number, path = datasets.commit(root, "crosswalk-eval", "eval", now=datetime(2026, 11, 1, tzinfo=UTC))
    rev = datasets.load_revision(root, "crosswalk-eval@r1")
    assert number == 1 and rev["split"] == "eval" and rev["parent_revision"] is None
    assert [f["frame"] for f in rev["frames"]] == [0, 5, 10] and len(rev["annotations"]) == 2
    assert {p.name: store.sha256_file(p) for p in run.summary.path.iterdir()} == before  # fixture 10
    first = path.read_bytes()
    d = datasets.load_draft(root, "crosswalk-eval")
    datasets.remove_annotation(d, "a2")
    datasets.save_draft(root, d)
    assert datasets.commit(root, "crosswalk-eval", "tuning")[0] == 2
    assert path.read_bytes() == first and len(datasets.load_revision(root, "crosswalk-eval@r1")["annotations"]) == 2
    rev2 = datasets.load_revision(root, "crosswalk-eval@r2")
    assert rev2["parent_revision"] == 1 and rev2["split"] == "tuning" and len(rev2["annotations"]) == 1
    assert [r["revision"] for r in datasets.list_revisions(root, "crosswalk-eval")] == [1, 2]


def test_a_changed_or_orphaned_revision_is_refused(run, root):
    reviewed_draft(run, root)
    _, path = datasets.commit(root, "crosswalk-eval", "eval")
    path.chmod(0o644)
    body = json.loads(path.read_text())
    body["annotations"] = []
    path.write_text(json.dumps(body))
    with pytest.raises(datasets.DatasetError, match="was changed"):
        datasets.load_revision(root, "crosswalk-eval@r1")
    (path.parent / "r2.json").write_text("{}")  # left by a commit interrupted before the index update
    assert datasets.commit(root, "crosswalk-eval", "eval")[0] == 3  # numbering skips it ...
    with pytest.raises(datasets.DatasetError, match="no revision"):
        datasets.load_revision(root, "crosswalk-eval@r2")  # ... and it is never loadable


def test_excluding_a_frame_drops_its_annotations(run, root):
    d = reviewed_draft(run, root)
    datasets.set_frame(d, 10, "excluded", [])
    assert [a["frame"] for a in d["annotations"]] == [0]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d["frames"].append({"frame": 7, "status": "reviewed", "classes_complete": [], "ignored": []}),
         "at least one class"),
        (lambda d: d["annotations"].append({**d["annotations"][0], "annotation_id": "a9", "frame": 5}),
         "only on reviewed frames"),
        (lambda d: d["annotations"].append({**d["annotations"][0], "annotation_id": "a9", "bbox_xyxy": [0, 0, 70, 9]}),
         "within the 64×48 frame"),
        (lambda d: d["annotations"].append({**d["annotations"][0], "annotation_id": "a9", "bbox_xyxy": [5, 5, 5, 9]}),
         "positive size"),
        (lambda d: d["annotations"].append({**d["annotations"][0], "annotation_id": "a9", "origin": "model"}),
         "origin must be"),
        (lambda d: d["annotations"].append(dict(d["annotations"][0])), "used twice"),
        (lambda d: d["frames"].append(dict(d["frames"][0])), "listed twice"),
        (lambda d: d.update(input_id="clip.mp4"), "input_id"),
        (lambda d: d.update(schema_version="2.0"), "schema_version"),
    ],
)  # fmt: skip
def test_invalid_drafts_are_refused_with_a_reason(run, root, change, message):
    d = reviewed_draft(run, root)
    change(d)
    with pytest.raises(datasets.DatasetError, match=message):
        datasets.save_draft(root, d)


def test_annotations_need_a_reviewed_frame_and_commit_needs_one(run, root):
    d = datasets.new_draft(root, "empty", run, [0])
    with pytest.raises(datasets.DatasetError, match="marked reviewed"):
        datasets.add_annotation(d, 0, "person", [0, 0, 5, 5], "added")
    with pytest.raises(datasets.DatasetError, match="nothing to commit"):
        datasets.commit(root, "empty", "eval")
    with pytest.raises(datasets.DatasetError, match="split must be"):
        datasets.commit(root, "empty", "final")


def test_refs_and_ids_are_checked(root):
    assert datasets.parse_ref("crosswalk-eval@r12") == ("crosswalk-eval", 12)
    for bad in ("crosswalk-eval", "crosswalk-eval@r0", "Crosswalk@r1", "../x@r1"):
        with pytest.raises(datasets.DatasetError):
            datasets.parse_ref(bad)
    with pytest.raises(datasets.DatasetError, match="dataset id"):
        datasets.folder(root, "../escape")


def test_import_makes_a_new_draft_whose_labels_say_imported(run, root, tmp_path):
    reviewed_draft(run, root)
    _, path = datasets.commit(root, "crosswalk-eval", "eval")
    copy = tmp_path / "shared.json"
    copy.write_text(path.read_text())
    d = datasets.import_file(root, copy, "from-colleague")
    assert d["kind"] == "draft" and {a["origin"] for a in d["annotations"]} == {"imported"}
    assert d["next_annotation"] == 3 and datasets.list_datasets(root) == ["crosswalk-eval", "from-colleague"]
    assert datasets.list_revisions(root, "from-colleague") == []  # ground truth only after a person commits
    with pytest.raises(datasets.DatasetError, match="already exists"):
        datasets.import_file(root, copy, "from-colleague")
    copy.write_text('{"kind": "something"}')
    with pytest.raises(datasets.DatasetError, match="not a MotionMan dataset"):
        datasets.import_file(root, copy, "x")

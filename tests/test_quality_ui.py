"""Quality lab, Review (plan P4 step 6): create a dataset, accept / add / exclude, commit a frozen revision.
Same world as test_app: real runs on disk, a fake runner, and model entry points that raise. Run A's fake track is a
person at (i, 5, i+10, 20) on frame i, missing on frame 3."""

from motionman import analysis, datasets, store

from . import test_app
from .test_app import open_run, render

world = test_app.world  # the same pytest fixture

DS = "crosswalk-eval"


def lab(world, step=10):
    """Open run A and create a dataset reviewing every ``step``-th frame (0 and 10 by default)."""
    at = open_run(render(), world["a"])
    at.text_input(key="qnew-id").set_value(DS)
    at.number_input(key="qnew-step").set_value(step)
    at = render(at)
    at.button(key="qcreate").click()
    return render(at)


def draft(world):
    return datasets.load_draft(world["root"], DS)


def test_a_new_dataset_starts_unlabelled_with_evenly_spaced_frames(world):
    at = lab(world)
    d = draft(world)
    assert d["selected"] == [0, 10] and d["frames"] == [] and d["annotations"] == []
    assert at.selectbox(key=f"qframe-{DS}").value == 0
    assert any(c.value.startswith("0 reviewed · 0 excluded · 2 to review of 2") for c in at.caption)


def test_review_accept_add_exclude_then_commit_and_the_run_is_unchanged(world):
    run = analysis.find_run(world["root"], world["a"])
    before = {p.name: store.sha256_file(p) for p in run.summary.path.iterdir()}
    at = lab(world)
    at.button(key="qacc-0-0:0").click()  # accept the person prediction on frame 0: the frame becomes reviewed
    at = render(at)
    for i, v in enumerate((40, 25, 50, 40)):
        at.number_input(key=f"qadd-{i}").set_value(v)
    at = render(at)
    at.button(key="qadd-save-0").click()  # a missed object
    at = render(at)
    at.selectbox(key=f"qframe-{DS}").set_value(10)
    at = render(at)
    at.button(key="qexcl-10").click()
    at = render(at)
    d = draft(world)
    assert [(a["annotation_id"], a["origin"]) for a in d["annotations"]] == [("a1", "accepted"), ("a2", "added")]
    assert [(f["frame"], f["status"], f["classes_complete"]) for f in d["frames"]] == [
        (0, "reviewed", ["person"]), (10, "excluded", [])]  # fmt: skip
    at.button(key=f"qcommit-{DS}").click()
    at = render(at)
    rev = datasets.load_revision(world["root"], f"{DS}@r1")
    assert rev["split"] == "eval" and len(rev["annotations"]) == 2
    assert any(c.value.startswith(f"Committed {DS}@r1 (eval)") for c in at.caption)
    commit = next(b for b in at.button if b.key == f"qcommit-{DS}")
    assert commit.label == "Committed" and commit.disabled  # nothing new to freeze
    assert {p.name: store.sha256_file(p) for p in run.summary.path.iterdir()} == before  # fixture 10: unchanged
    assert world["fake"].starts == []


def test_a_frame_cannot_be_reviewed_without_naming_its_complete_classes(world):
    at = lab(world)
    at.multiselect(key=f"qclasses-{DS}-0").set_value([])
    at = render(at)
    at.button(key="qrev-0").click()
    at = render(at)
    assert draft(world)["frames"] == [] and any("choose the classes" in e.value for e in at.error)
    assert next(b for b in at.button if b.key == f"qcommit-{DS}").disabled  # nothing reviewed: nothing to commit


def test_fixing_a_box_replaces_it_as_corrected(world):
    at = lab(world)
    at.button(key="qacc-0-0:0").click()
    at = render(at)
    for i, v in enumerate((1, 5, 11, 20)):
        at.number_input(key=f"qadd-{i}").set_value(v)
    at.selectbox(key="qadd-fix").set_value("a1")
    at = render(at)
    at.button(key="qadd-save-0").click()
    at = render(at)
    (a,) = draft(world)["annotations"]
    assert (a["origin"], a["bbox_xyxy"], a["note"]) == ("corrected", [1.0, 5.0, 11.0, 20.0], "replaces a1")


def test_the_review_queue_adds_its_frames_and_labels_nothing(world):
    at = lab(world)
    assert any(c.value.startswith("Candidates, not errors") for c in at.caption)
    at.button(key="qqueue-add").click()
    at = render(at)
    d = draft(world)
    assert 4 in d["selected"] and d["frames"] == [] and d["annotations"] == []  # frame 4: the track's return


def test_reviewing_never_infers(world):
    at = lab(world)
    at.selectbox(key=f"qframe-{DS}").set_value(10)
    at = render(at)
    at.button(key="qnext").click()
    at = render(at)
    assert world["fake"].starts == []


# --- step 7: Evaluate and Compare runs ---


def committed(world, split="eval", dataset_id=DS):
    """Frames 0 and 10 reviewed for person: both predictions accepted, plus one object the model missed."""
    run = analysis.find_run(world["root"], world["a"])
    d = datasets.new_draft(world["root"], dataset_id, run, [0, 10])
    datasets.set_frame(d, 0, "reviewed", ["person"])
    datasets.add_annotation(d, 0, "person", [0, 5, 10, 20], "accepted")
    datasets.set_frame(d, 10, "reviewed", ["person"])
    datasets.add_annotation(d, 10, "person", [10, 5, 20, 20], "accepted")
    datasets.add_annotation(d, 10, "person", [40, 25, 50, 40], "added")
    datasets.save_draft(world["root"], d)
    return f"{dataset_id}@r{datasets.commit(world['root'], dataset_id, split)[0]}"


def section(world, name):
    at = open_run(render(), world["a"])
    at.segmented_control(key="qsection").set_value(name)
    return render(at)


def results_folder(world, kind):
    base = world["root"].path / kind
    return (
        sorted(p.name for p in base.rglob("*") if p.is_dir() and p.name[:3] in ("ev_", "dg_")) if base.exists() else []
    )


def test_evaluate_scores_the_run_and_names_revision_thresholds_and_version_on_screen(world):
    ref = committed(world)
    committed(world, split="tuning", dataset_id="tune")
    at = section(world, "Evaluate")
    assert at.selectbox(key="qeval-ref").options == [ref]  # tuning revisions are never offered
    assert results_folder(world, "evaluations") == []  # nothing is written by opening the section
    at.button(key="qeval-go").click()
    at = render(at)
    assert len(results_folder(world, "evaluations")) == 1
    go = next(b for b in at.button if b.key == "qeval-go")
    assert go.label == "Up to date" and go.disabled
    assert any(c.value.startswith(f"fixed-threshold detection precision/recall · {ref} (sha ") and
               "confidence ≥ 0.25 · IoU ≥ 0.5 · detection-v1" in c.value for c in at.caption)  # fmt: skip
    m = {x.label: x.value for x in at.metric}
    assert m["Precision"] == "1.000" and m["Recall"] == "0.667" and m["False pos. / missed"] == "0 / 1"
    at.button(key="qex-false_negatives-0").click()  # the missed object, traced to its frame
    at = render(at)
    assert at.slider(key=f"frame-{world['a']}").value == 10 and world["fake"].starts == []


def test_evaluate_explains_when_there_is_no_eval_revision(world):
    committed(world, split="tuning", dataset_id="tune")
    at = section(world, "Evaluate")
    assert any(c.value.startswith("No frozen eval revision") for c in at.caption)


def test_compare_runs_counts_differences_and_sends_frames_to_review_without_labelling(world):
    committed(world)
    at = section(world, "Compare runs")
    (label,) = at.selectbox(key="qcmp-run").options  # shown by date, size, frames and the run id's tail
    assert label.endswith(world["b"][-6:]) and "imgsz 640" in label and results_folder(world, "disagreements") == []
    at.button(key="qcmp-go").click()
    at = render(at)
    m = {x.label: x.value for x in at.metric}
    assert m["Same box, same class"] == "0" and m["Only in this run"] == "20" and m["Only in the other"] == "20"
    assert any("track IDs are local to each run" in c.value for c in at.caption)
    before = datasets.load_draft(world["root"], DS)
    at.button(key="qcmp-send").click()
    at = render(at)
    after = datasets.load_draft(world["root"], DS)
    assert len(after["selected"]) > len(before["selected"]) and after["annotations"] == before["annotations"]
    assert any(c.value.startswith("Added 10 frames to crosswalk-eval") for c in at.caption)
    assert world["fake"].starts == []

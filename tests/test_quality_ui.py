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

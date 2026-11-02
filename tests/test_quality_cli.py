"""The P4 commands (plan P4 step 5, D11): dataset import/show, evaluate, disagree; happy paths and exit-2 refusals."""

import json

import pytest

from motionman import analysis, cli, datasets, pipeline, store
from motionman.types import RunConfig

from .test_disagree import MISSING
from .test_pipeline import FakeTracker


@pytest.fixture
def root(tmp_path):
    return store.data_root(tmp_path / "data")


def publish(clip, root, make=FakeTracker):
    report = pipeline.run(RunConfig(clip, None, data_root=root.path), make_tracker=make, resolve=lambda d: "cpu")
    return analysis.find_run(root, report["run_id"])


def commit(run, root, split="eval", dataset_id="crosswalk-eval"):
    d = datasets.new_draft(root, dataset_id, run, [0, 10])
    datasets.set_frame(d, 0, "reviewed", ["person"])
    datasets.add_annotation(d, 0, "person", [0, 5, 10, 20], "accepted")
    datasets.set_frame(d, 10, "excluded", [])
    datasets.save_draft(root, d)
    return f"{dataset_id}@r{datasets.commit(root, dataset_id, split)[0]}"


def args(root, *rest):
    return [*rest, "--data-root", str(root.path)]


def test_evaluate_prints_the_metric_its_revision_thresholds_and_counts(clip, root, capsys):
    run = publish(clip, root)
    ref = commit(run, root)
    assert cli.main(args(root, "evaluate", "--run", run.run_id, "--dataset", ref)) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert out.startswith(f"fixed-threshold detection precision/recall · {ref} · confidence ≥ 0.25 · IoU ≥ 0.5")
    assert "person         TP    1  FP    0  FN    0  precision 1.000  recall 1.000" in out
    assert "1 scored, 1 excluded, 0 not in this run" in out
    assert cli.main(args(root, "evaluate", "--run", run.run_id, "--dataset", ref, "--json")) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["dataset"]["ref"] == ref


def test_evaluate_refuses_bad_input_with_exit_2(clip, root, capsys):
    run = publish(clip, root)
    tuning = commit(run, root, split="tuning", dataset_id="tune")
    cases = [["--run", run.run_id, "--dataset", tuning], ["--run", run.run_id, "--dataset", "nope@r1"],
             ["--run", run.run_id, "--dataset", "crosswalk"], ["--run", "r_nope", "--dataset", tuning],
             ["--run", run.run_id, "--dataset", commit(run, root), "--conf", "0.05"]]  # fmt: skip
    for case in cases:
        assert cli.main(args(root, "evaluate", *case)) == 2, case
    err = capsys.readouterr().err
    assert "tuning revision" in err and "no revision nope@r1" in err and "run's own threshold" in err


def test_disagree_prints_counts_and_where_to_look(clip, root, capsys):
    a, b = publish(clip, root), publish(clip, root, MISSING)
    assert cli.main(args(root, "disagree", a.run_id, b.run_id)) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "20 frames compared at IoU ≥ 0.5" in out and "track IDs are never compared" in out
    assert "most differences: frame " in out and "only in A 20 · only in B 40" in out
    assert cli.main(args(root, "disagree", a.run_id, "r_nope")) == 2


def test_dataset_show_and_import(clip, root, tmp_path, capsys):
    run = publish(clip, root)
    ref = commit(run, root)
    assert cli.main(args(root, "dataset", "show", "crosswalk-eval")) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "revisions: r1 eval" in out and "1 reviewed, 1 excluded" in out and "person ×1" in out
    assert cli.main(args(root, "dataset", "show", ref)) == cli.EXIT_OK
    assert capsys.readouterr().out.startswith(f"{ref} (eval, committed ")
    copy = tmp_path / "shared.json"
    copy.write_text((root.path / "datasets" / "crosswalk-eval" / "revisions" / "r1.json").read_text())
    assert cli.main(args(root, "dataset", "import", str(copy), "--as", "colleague")) == cli.EXIT_OK
    assert "origin imported" in capsys.readouterr().out
    assert cli.main(args(root, "dataset", "import", str(copy), "--as", "colleague")) == 2  # already exists
    assert cli.main(args(root, "dataset", "show", "missing-set")) == 2

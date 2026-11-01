"""Failure handling on the canonical path (MD§25 P0.4, MD§18 fixture 9). Legacy-mode equivalents live in
test_pipeline.py; this file covers what the P0 step 6 audit found untested for canonical runs."""

import json

import pytest

from motionman import pipeline, records, store, video_io
from motionman.types import Limits, RunConfig, Track

from .conftest import write_clip
from .test_pipeline import FakeTracker


def canonical(clip, root, **kw):
    return RunConfig(clip, None, data_root=root, **kw)


def run(config, tracker=FakeTracker):
    return pipeline.run(config, make_tracker=tracker, resolve=lambda d: "cpu")


def entries(folder):
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def only_staged_manifest(root):
    (staged,) = list((root / "staging").iterdir())
    return json.loads((staged / store.MANIFEST).read_text())


@pytest.mark.parametrize(
    ("make_source", "limits", "message"),
    [
        (lambda d: d / "missing.mp4", Limits(), "source not found"),
        (lambda d: (d / "notes.mp4").write_text("text") and d / "notes.mp4", Limits(), "cannot open video"),
        (lambda d: write_clip(d / "c.mp4", 20), Limits(max_frames=5), "frame limit"),
    ],
)
def test_invalid_input_creates_nothing_in_the_data_root(tmp_path, make_source, limits, message):
    root = tmp_path / "data"
    with pytest.raises(video_io.VideoError, match=message):
        run(canonical(make_source(tmp_path), root, limits=limits))
    assert entries(root / "inputs") == [] and entries(root / "staging") == [] and entries(root / "runs") == []


def test_a_run_of_only_empty_frames_succeeds_with_empty_records(clip, tmp_path):
    class Nothing(FakeTracker):
        def update(self, image):
            super().update(image)
            return []

    root = tmp_path / "data"
    report = run(canonical(clip, root), tracker=Nothing)
    recs = list(records.read_records(root / "runs" / report["run_id"] / "tracks.jsonl"))
    assert len(recs) == 20 and all(r["objects"] == [] for r in recs)  # empty is a result, not a gap


def test_writer_failure_is_not_published(clip, tmp_path, monkeypatch):
    def broken_writer(path, info):
        raise video_io.VideoError(f"cannot open video writer: {path}")

    monkeypatch.setattr(video_io, "open_writer", broken_writer)
    root = tmp_path / "data"
    with pytest.raises(video_io.VideoError, match="cannot open video writer"):
        run(canonical(clip, root))
    assert entries(root / "runs") == []
    m = only_staged_manifest(root)
    assert m["status"] == "FAILED" and m["records_sha256"] is None


def test_output_that_cannot_finalize_never_gets_a_succeeded_manifest(clip, tmp_path, monkeypatch):
    # MD§18 fixture 9: the written video decodes to fewer frames than were processed.
    monkeypatch.setattr(video_io, "count_frames", lambda path: 17)
    root = tmp_path / "data"
    with pytest.raises(pipeline.OutputCheckError, match="decodes to 17"):
        run(canonical(clip, root))
    assert entries(root / "runs") == []
    m = only_staged_manifest(root)
    assert m["status"] == "FAILED" and m["validation"]["video_decode"] == "FAIL"
    assert m["counts"]["written_frames"] == 17


def test_an_invalid_box_from_the_tracker_fails_the_run(clip, tmp_path):
    class OffImage(FakeTracker):
        def update(self, image):
            tracks = super().update(image)
            return [Track(9, 0, "person", 0.9, (10.0, 10.0, 500.0, 20.0))] if self.calls == 4 else tracks

    root = tmp_path / "data"
    with pytest.raises(records.RecordError, match="outside the 64x48 source image"):
        run(canonical(clip, root), tracker=OffImage)
    assert entries(root / "runs") == []
    assert only_staged_manifest(root)["status"] == "FAILED"


@pytest.mark.parametrize(
    "fail", [RuntimeError("boom"), KeyboardInterrupt()], ids=["error", "interrupt"]
)  # fmt: skip
def test_the_lease_is_released_after_a_failed_run(clip, tmp_path, fail):
    root = tmp_path / "data"
    with pytest.raises(type(fail)):
        run(canonical(clip, root), tracker=lambda c: FakeTracker(c, fail_at=3, exc=fail))
    assert run(canonical(clip, root))["status"] == "success"  # the next run can take the lease

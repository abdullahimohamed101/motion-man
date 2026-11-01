"""Inspection over saved evidence (spec A§8, plan P1 D3): a sparse records index for random access by frame
and per-track summaries, built in one streaming pass and cached as disposable derived data.

Nothing here runs a model. The cache lives in ``<data root>/derived/<key>/`` where the key hashes the records
file's hash and this implementation's version, so a changed records file can never be served stale data.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from motionman import records, store

INDEX_STRIDE = 128  # one byte offset every 128 records (MD§4)
IMPL_VERSION = 1  # bump when the derived format or its rules change


class InspectError(records.RecordError):
    """Saved evidence can't be read for inspection; the message says why."""


@dataclass(frozen=True)
class TrackSummary:
    """What a run observed for one run-local track ID. First-to-last span is not continuous presence (MD§6)."""

    track_id: int
    class_name: str  # the most frequent class for this ID
    class_counts: dict[str, int]
    first_frame: int
    last_frame: int
    first_t_us: int
    last_t_us: int
    observations: int
    intervals: list[tuple[int, int]]  # inclusive runs of consecutive observed frames
    gaps: list[tuple[int, int]]  # inclusive runs of frames between intervals where the ID was not observed
    confidence_p10: float
    confidence_p50: float
    confidence_p90: float


class RecordIndex:
    """Random access to a run's frame records without reading the whole file."""

    def __init__(self, records_path: Path, offsets: list[int], count: int) -> None:
        self.path = records_path
        self.offsets = offsets
        self.count = count

    def record(self, frame: int) -> dict:
        return next(self.records(frame, frame + 1))

    def records(self, start: int, stop: int) -> Iterator[dict]:
        """Records for frames [start, stop), in order, read from the nearest preceding offset."""
        if not 0 <= start < stop <= self.count:
            raise InspectError(f"frames [{start}, {stop}) are outside this run's 0..{self.count - 1}")
        return self._iter(start, stop)

    def _iter(self, start: int, stop: int) -> Iterator[dict]:
        block = start // INDEX_STRIDE
        frame = block * INDEX_STRIDE
        with self.path.open("rb") as f:
            f.seek(self.offsets[block])
            for line in f:
                if frame >= stop:
                    return
                if frame >= start:
                    rec = records.parse_line(line, f"{self.path.name} frame {frame}")
                    if rec["source_frame_index"] != frame:
                        raise InspectError(f"{self.path.name}: index points at the wrong line for frame {frame}")
                    yield rec
                frame += 1
        if frame < stop:
            raise InspectError(f"{self.path.name}: ended at frame {frame}, expected {stop}")


@dataclass(frozen=True)
class Derived:
    index: RecordIndex
    tracks: dict[int, TrackSummary]
    unassigned_observations: int  # detections without a track ID; never folded into a track
    folder: Path


def load_or_build(records_path: Path, root: store.DataRoot, records_sha256: str | None = None) -> Derived:
    """Return the index and track summaries for a records file, rebuilding the cache if missing or stale."""
    actual = store.sha256_file(records_path)
    if records_sha256 is not None and actual != records_sha256:
        raise InspectError(f"{records_path.name} does not match the hash the run recorded")
    key = store.sha256_json({"kind": "inspect", "records_sha256": actual, "impl_version": IMPL_VERSION})
    folder = root.path / "derived" / key.removeprefix("sha256:")
    cached = _load(folder, actual, records_path)
    return cached if cached is not None else _build(records_path, actual, folder)


def _build(records_path: Path, records_sha: str, folder: Path) -> Derived:
    offsets: list[int] = []
    obs: dict[int, list[tuple[int, int, str, float]]] = {}  # id -> (frame, t_us, class, confidence)
    unassigned = 0
    count = 0
    position = 0
    with records_path.open("rb") as f:
        for line in f:
            if count % INDEX_STRIDE == 0:
                offsets.append(position)
            position += len(line)
            rec = records.parse_line(line, f"{records_path.name}:{count + 1}")
            if rec["source_frame_index"] != count:
                raise InspectError(f"{records_path.name}: line {count + 1} is frame {rec['source_frame_index']}")
            for o in rec["objects"]:
                if o["track_id"] is None:
                    unassigned += 1
                else:
                    obs.setdefault(o["track_id"], []).append(
                        (count, rec["source_t_us"], o["class_name"], o["confidence"])
                    )
            count += 1
    tracks = {tid: _summarize(tid, rows) for tid, rows in obs.items()}
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"records_sha256": records_sha, "impl_version": IMPL_VERSION, "count": count,
            "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    store.write_json_atomic(folder / "index.json", {**meta, "stride": INDEX_STRIDE, "offsets": offsets})
    store.write_json_atomic(
        folder / "tracks.json",
        {**meta, "unassigned_observations": unassigned, "tracks": [asdict(t) for t in tracks.values()]},
    )
    return Derived(RecordIndex(records_path, offsets, count), tracks, unassigned, folder)


def _load(folder: Path, records_sha: str, records_path: Path) -> Derived | None:
    try:
        index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
        tracks = json.loads((folder / "tracks.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for meta in (index, tracks):
        if meta.get("records_sha256") != records_sha or meta.get("impl_version") != IMPL_VERSION:
            return None
    if index.get("stride") != INDEX_STRIDE:
        return None
    summaries = {}
    for t in tracks.get("tracks", []):
        t["intervals"] = [tuple(x) for x in t["intervals"]]
        t["gaps"] = [tuple(x) for x in t["gaps"]]
        summaries[t["track_id"]] = TrackSummary(**t)
    return Derived(
        RecordIndex(records_path, index["offsets"], index["count"]),
        summaries,
        tracks.get("unassigned_observations", 0),
        folder,
    )


def _summarize(track_id: int, rows: list[tuple[int, int, str, float]]) -> TrackSummary:
    frames = [r[0] for r in rows]
    intervals: list[tuple[int, int]] = []
    start = prev = frames[0]
    for f in frames[1:]:
        if f != prev + 1:
            intervals.append((start, prev))
            start = f
        prev = f
    intervals.append((start, prev))
    gaps = [(a[1] + 1, b[0] - 1) for a, b in zip(intervals, intervals[1:], strict=False)]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r[2]] = counts.get(r[2], 0) + 1
    conf = np.array([r[3] for r in rows])
    return TrackSummary(
        track_id=track_id,
        class_name=max(counts, key=lambda c: (counts[c], c)),
        class_counts=counts,
        first_frame=frames[0],
        last_frame=frames[-1],
        first_t_us=rows[0][1],
        last_t_us=rows[-1][1],
        observations=len(rows),
        intervals=intervals,
        gaps=gaps,
        confidence_p10=round(float(np.percentile(conf, 10)), 4),
        confidence_p50=round(float(np.percentile(conf, 50)), 4),
        confidence_p90=round(float(np.percentile(conf, 90)), 4),
    )

"""Spatial and temporal rules over saved observations (spec B§4, MD§10, plan P3 D4–D8).

Pure and incremental: feed one frame at a time in order and get the events it confirms. Nothing here reads files,
renders or runs a model. Every rule uses the bounding-box centre in source pixels and source time in µs.

Release A infers every frame, so a track missing from a frame has broken continuity: its state is dropped, any
pending transition is forgotten, and its next observation re-establishes state without an event. Disappearing is
never an exit, and two sightings on either side of a gap are never a crossing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from motionman import regions

LINE_CROSSING, ZONE_ENTRY, ZONE_EXIT = "LINE_CROSSING", "ZONE_ENTRY", "ZONE_EXIT"
DWELL_THRESHOLD, OCCUPANCY_THRESHOLD = "DWELL_THRESHOLD", "OCCUPANCY_THRESHOLD"
POSITIVE, NEGATIVE = 1, -1
INSIDE, OUTSIDE, UNKNOWN = "INSIDE", "OUTSIDE", "UNKNOWN"
DIRECTION = {(NEGATIVE, POSITIVE): "NEGATIVE_TO_POSITIVE", (POSITIVE, NEGATIVE): "POSITIVE_TO_NEGATIVE"}


@dataclass(frozen=True)
class Params:
    """Boundary and continuity policy (MD§10 starting values). Persisted in every AnalysisSpec."""

    deadband_px: float = 3.0  # a centre must be this far from a boundary to count as stably on one side
    confirm_us: int = 100_000  # and stay there this long ...
    confirm_obs: int = 2  # ... over at least this many consecutive observations
    max_gap_us: int = 500_000  # a longer source-time gap between observations always breaks continuity

    def to_dict(self) -> dict:
        return {"deadband_px": self.deadband_px, "confirm_us": self.confirm_us, "confirm_obs": self.confirm_obs,
                "max_gap_us": self.max_gap_us}  # fmt: skip


DEFAULT_PARAMS = Params()


@dataclass(frozen=True)
class Obs:
    track_id: int | None  # None = unassigned detection: counted, never part of a rule (INV-2)
    class_name: str
    x: float  # bounding-box centre, source pixels
    y: float


@dataclass(frozen=True)
class Event:
    type: str
    rule_id: str
    track_id: int | None  # None only for OCCUPANCY_THRESHOLD, which is about a count, not a track
    class_name: str | None
    direction: str | None  # LINE_CROSSING only
    confirming_frame: int
    confirming_t_us: int
    evidence_us: tuple[int, int]  # [start, confirming time]: the observations that justify the event
    value: int | None = None  # DWELL_THRESHOLD: visit dwell µs; OCCUPANCY_THRESHOLD: the count

    def sort_key(self) -> tuple:
        return (self.confirming_frame, self.rule_id, -1 if self.track_id is None else self.track_id, self.type)

    def to_dict(self) -> dict:
        return {"type": self.type, "rule_id": self.rule_id, "track_id": self.track_id,
                "class_name": self.class_name, "direction": self.direction,
                "confirming_frame": self.confirming_frame, "confirming_t_us": self.confirming_t_us,
                "evidence_interval_us": list(self.evidence_us), "value": self.value}  # fmt: skip


@dataclass
class RegionStats:
    """Totals one region accumulates; integers only (µs and counts), so they are exact and repeatable."""

    kind: str
    crossings: dict[str, int] = field(default_factory=lambda: dict.fromkeys(DIRECTION.values(), 0))
    entries: int = 0
    exits: int = 0
    peak_occupancy: int = 0
    peak_frame: int | None = None
    occupancy_us: int = 0  # ∑ occupancy × frame duration: time-weighted mean = occupancy_us / covered_us
    covered_us: int = 0
    peak_unassigned: int = 0  # most null-ID detections inside at once (never in occupancy)
    visits_us: list[int] = field(default_factory=list)  # observed dwell of each closed visit


class Geometry:
    """One region in source pixels, with its class filter and thresholds."""

    def __init__(self, region: regions.Region, width: int, height: int) -> None:
        self.id, self.kind = region.id, region.kind
        self.coords = region.to_px(width, height)
        self.classes = frozenset(region.classes) if region.classes is not None else None
        self.dwell_us = round(region.dwell_threshold_s * 1e6) if region.dwell_threshold_s is not None else None
        self.occupancy_threshold = region.occupancy_threshold
        if self.kind == "line":
            ax, ay, bx, by = self.coords
            self._length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5

    def accepts(self, class_name: str) -> bool:
        return self.classes is None or class_name in self.classes

    def signed_distance(self, x: float, y: float) -> float:
        """Line: px from the infinite line; positive = left when facing A→B on screen (image y points down)."""
        ax, ay, bx, by = self.coords
        return -((bx - ax) * (y - ay) - (by - ay) * (x - ax)) / self._length

    def crosses(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Line: does the motion segment properly intersect the finite line? Touching or collinear is not."""
        ax, ay, bx, by = self.coords
        d1, d2 = _orient(ax, ay, bx, by, x0, y0), _orient(ax, ay, bx, by, x1, y1)
        d3, d4 = _orient(x0, y0, x1, y1, ax, ay), _orient(x0, y0, x1, y1, bx, by)
        return d1 * d2 < 0 and d3 * d4 < 0

    def inside(self, x: float, y: float) -> bool:
        """Rect: raw membership for occupancy and dwell; a centre on the edge counts as inside (MD§10)."""
        x1, y1, x2, y2 = self.coords
        return x1 <= x <= x2 and y1 <= y <= y2

    def zone_side(self, x: float, y: float, deadband: float) -> str | None:
        """Rect: INSIDE or OUTSIDE when at least ``deadband`` px from the edge, else None (in the band)."""
        x1, y1, x2, y2 = self.coords
        if min(x - x1, x2 - x, y - y1, y2 - y) >= deadband:
            return INSIDE
        if max(x1 - x, x - x2, y1 - y, y - y2) >= deadband:
            return OUTSIDE
        return None


@dataclass
class _Pending:
    to: int | str  # the side or zone state being confirmed
    start_us: int  # evidence start
    first_t: int | None = None  # first consecutive stable observation on the new side
    n: int = 0


@dataclass
class _Track:
    frame: int
    t: int
    x: float
    y: float
    stable: int | str | None = None  # line: POSITIVE/NEGATIVE; rect: INSIDE/OUTSIDE/UNKNOWN
    pending: _Pending | None = None
    ref: tuple[float, float, int] | None = None  # line: last observation strictly on the stable side (x, y, t)
    inside: bool = False  # rect: raw membership at the last observation
    last_old_t: int = 0  # rect: last time stably in the current state (an entry/exit's evidence start)
    visit_us: int = 0  # rect: observed dwell of the current visit
    visit_start: int | None = None
    dwell_fired: bool = False


class Engine:
    """Runs every region's rules over frames fed in order. ``events`` come back per frame, sorted (plan D8)."""

    def __init__(self, region_list: Sequence[regions.Region], width: int, height: int, params: Params | None = None):
        self.params = params or DEFAULT_PARAMS
        self.geometry = [Geometry(r, width, height) for r in region_list]
        self.stats = {g.id: RegionStats(g.kind) for g in self.geometry}
        self._tracks: dict[str, dict[int, _Track]] = {g.id: {} for g in self.geometry}
        self._occupancy = {g.id: 0 for g in self.geometry}
        self._above = {g.id: False for g in self.geometry}
        self._prev_t: int | None = None
        self._prev_frame: int | None = None

    def feed(self, frame: int, t_us: int, objects: Iterable[Obs]) -> list[Event]:
        if self._prev_frame is not None and (frame <= self._prev_frame or t_us <= self._prev_t):
            raise ValueError(f"frames must be fed in order: {frame} at {t_us} µs after {self._prev_frame}")
        self._close_interval(t_us)
        objects = list(objects)
        events: list[Event] = []
        for g in self.geometry:
            eligible: dict[int, Obs] = {}
            for o in objects:
                if o.track_id is not None and g.accepts(o.class_name):
                    eligible.setdefault(o.track_id, o)  # a duplicated ID in one frame: the first wins, consistently
            if g.kind == "line":
                events += self._line(g, frame, t_us, eligible)
            else:
                unassigned = [o for o in objects if o.track_id is None and g.accepts(o.class_name)]
                events += self._zone(g, frame, t_us, eligible, unassigned)
        self._prev_t, self._prev_frame = t_us, frame
        return sorted(events, key=Event.sort_key)

    def finish(self, end_t_us: int) -> None:
        """Close the last frame's occupancy interval at ``end_t_us`` and every open visit. Emits nothing: the end
        of the clip is not an exit."""
        if self._prev_t is not None and end_t_us > self._prev_t:
            self._close_interval(end_t_us)
        for g in self.geometry:
            for st in self._tracks[g.id].values():
                self._close_visit(g, st)
            self._tracks[g.id].clear()

    def _close_interval(self, t_us: int) -> None:
        if self._prev_t is None:
            return
        dt = t_us - self._prev_t
        for g in self.geometry:
            if g.kind == "rect":
                s = self.stats[g.id]
                s.occupancy_us += self._occupancy[g.id] * dt
                s.covered_us += dt

    def _continues(self, st: _Track | None, frame: int, t_us: int) -> bool:
        return st is not None and frame == st.frame + 1 and t_us - st.t <= self.params.max_gap_us

    def _drop_absent(self, g: Geometry, present: dict[int, Obs]) -> None:
        tracks = self._tracks[g.id]
        for tid in [t for t in tracks if t not in present]:
            self._close_visit(g, tracks.pop(tid))

    def _confirm(self, st: _Track, t_us: int) -> bool:
        p, prm = st.pending, self.params
        return p is not None and p.first_t is not None and p.n >= prm.confirm_obs and t_us - p.first_t >= prm.confirm_us

    @staticmethod
    def _step_pending(st: _Track, side: int | str | None, t_us: int) -> None:
        p = st.pending
        if side == p.to:
            if p.first_t is None:
                p.first_t = t_us
            p.n += 1
        elif side == st.stable:
            st.pending = None  # back on the original side: no event
        else:
            p.first_t, p.n = None, 0  # in the band: the pending transition stays, its confirmation restarts

    def _line(self, g: Geometry, frame: int, t_us: int, present: dict[int, Obs]) -> list[Event]:
        self._drop_absent(g, present)
        tracks, db, out = self._tracks[g.id], self.params.deadband_px, []
        for tid in sorted(present):
            o = present[tid]
            d = g.signed_distance(o.x, o.y)
            raw = POSITIVE if d > 0 else NEGATIVE if d < 0 else 0
            side = raw if abs(d) >= db else None
            st = tracks.get(tid)
            if not self._continues(st, frame, t_us):
                tracks[tid] = st = _Track(frame, t_us, o.x, o.y, stable=side)  # first sighting: state, no event
                st.ref = (o.x, o.y, t_us) if side is not None else None
                continue
            if st.stable is None:
                st.stable = side
            else:
                if st.pending is None:
                    # From the last point strictly on the stable side, so a centre landing exactly on the line
                    # (where neither neighbouring pair "properly" crosses) still counts. Normally ref = previous.
                    if raw == -st.stable and st.ref is not None and g.crosses(st.ref[0], st.ref[1], o.x, o.y):
                        st.pending = _Pending(-st.stable, st.ref[2])
                    elif side == -st.stable:
                        st.stable = side  # went round an endpoint: new side, no crossing
                if st.pending is not None:
                    self._step_pending(st, side, t_us)
                    if self._confirm(st, t_us):
                        p, direction = st.pending, DIRECTION[(st.stable, st.pending.to)]
                        out.append(Event(LINE_CROSSING, g.id, tid, o.class_name, direction, frame, t_us,
                                         (p.start_us, t_us)))  # fmt: skip
                        self.stats[g.id].crossings[direction] += 1
                        st.stable, st.pending = p.to, None  # rearmed: the next crossing starts from the new side
            if st.stable is not None and raw == st.stable:
                st.ref = (o.x, o.y, t_us)
            st.frame, st.t, st.x, st.y = frame, t_us, o.x, o.y
        return out

    def _zone(self, g: Geometry, frame: int, t_us: int, present: dict[int, Obs], unassigned: list[Obs]) -> list[Event]:
        self._drop_absent(g, present)
        tracks, db, s, out = self._tracks[g.id], self.params.deadband_px, self.stats[g.id], []
        for tid in sorted(present):
            o = present[tid]
            raw, side = g.inside(o.x, o.y), g.zone_side(o.x, o.y, db)
            st = tracks.get(tid)
            if not self._continues(st, frame, t_us):
                if st is not None:
                    self._close_visit(g, st)
                tracks[tid] = _Track(frame, t_us, o.x, o.y, stable=side or UNKNOWN, inside=raw, last_old_t=t_us)
                continue
            if raw and st.inside:  # an observed interval with both ends inside counts towards dwell
                st.visit_start = st.t if st.visit_start is None else st.visit_start
                st.visit_us += t_us - st.t
                if g.dwell_us is not None and not st.dwell_fired and st.visit_us >= g.dwell_us:
                    st.dwell_fired = True
                    out.append(Event(DWELL_THRESHOLD, g.id, tid, o.class_name, None, frame, t_us,
                                     (st.visit_start, t_us), st.visit_us))  # fmt: skip
            if st.stable == UNKNOWN:
                if side is not None:
                    st.stable, st.last_old_t = side, t_us  # established, not entered: no event
            else:
                if st.pending is None:
                    if side == st.stable:
                        st.last_old_t = t_us
                    elif side is not None:
                        st.pending = _Pending(side, st.last_old_t)
                if st.pending is not None:
                    self._step_pending(st, side, t_us)
                    if st.pending is None:
                        st.last_old_t = t_us
                    elif self._confirm(st, t_us):
                        p = st.pending
                        entry = p.to == INSIDE
                        out.append(Event(ZONE_ENTRY if entry else ZONE_EXIT, g.id, tid, o.class_name, None, frame,
                                         t_us, (p.start_us, t_us)))  # fmt: skip
                        if entry:
                            s.entries += 1
                        else:
                            s.exits += 1
                            self._close_visit(g, st)
                        st.stable, st.pending, st.last_old_t = p.to, None, t_us
            st.frame, st.t, st.x, st.y, st.inside = frame, t_us, o.x, o.y, raw
        occupancy = sum(g.inside(o.x, o.y) for o in present.values())
        self._occupancy[g.id] = occupancy
        s.peak_unassigned = max(s.peak_unassigned, sum(g.inside(o.x, o.y) for o in unassigned))
        if occupancy > s.peak_occupancy:
            s.peak_occupancy, s.peak_frame = occupancy, frame
        if g.occupancy_threshold is not None:
            above = occupancy >= g.occupancy_threshold
            if above and not self._above[g.id]:  # upward transitions only, never every frame above the limit
                out.append(Event(OCCUPANCY_THRESHOLD, g.id, None, None, None, frame, t_us, (t_us, t_us), occupancy))
            self._above[g.id] = above
        return out

    def _close_visit(self, g: Geometry, st: _Track) -> None:
        if g.kind == "rect" and st.visit_us > 0:
            self.stats[g.id].visits_us.append(st.visit_us)
        st.visit_us, st.visit_start, st.dwell_fired = 0, None, False


def _orient(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

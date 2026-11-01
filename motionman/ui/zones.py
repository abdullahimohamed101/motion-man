"""Events & zones tab, part 1: draw regions, save them, analyze (spec B§6, plan P3 D12).

Layout and state only. Regions are validated by ``regions``, analysis runs in ``analysis`` over saved records, and
nothing here can run the model. Editing or saving regions never analyzes; Analyze is the only action that writes
results, and an identical analysis is reused instead of rebuilt.
"""

from __future__ import annotations

import functools
import re

import streamlit as st

from motionman import analysis, inspection, regions, render, runs, store
from motionman.ui import view

KINDS = {"Line": "line", "Rectangle": "rect"}
DEFAULTS = {"line": (10.0, 50.0, 90.0, 50.0), "rect": (25.0, 25.0, 75.0, 75.0)}  # percent of the frame
COORD_LABELS = {"line": ("A · x %", "A · y %", "B · x %", "B · y %"),
                "rect": ("Left %", "Top %", "Right %", "Bottom %")}  # fmt: skip
NEW_SET = "new"


def draft_key(input_id: str) -> str:
    return "zdraft-" + input_id.removeprefix("sha256:")[:16]  # regions belong to the video, not to one run


def slug(name: str, taken: set[str]) -> str:
    """A region id from its display name: lowercase, a–z 0–9 and '-', unique within the set."""
    base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:28] or "region"
    rid, n = base, 2
    while rid in taken:
        rid, n = f"{base}-{n}", n + 1
    return rid


def candidate(regions_so_far: list[dict], *, preview: bool = False) -> dict | None:
    """The region described by the editor's widgets right now. Adding needs a name (None until then); the preview
    draws the shape straight away under a placeholder name, so it can be placed before it is named."""
    ss = st.session_state
    name = (ss.get("zname") or "").strip() or ("New region" if preview else "")
    if not name:
        return None
    kind = KINDS[ss.get("zkind") or "Line"]
    coords = [round(float(ss.get(f"zc{i}", DEFAULTS[kind][i])) / 100, 6) for i in range(4)]
    body = {"id": slug(name, {r["id"] for r in regions_so_far}), "name": name, "kind": kind,
            "classes": list(ss.get("zclasses") or []) or None}  # fmt: skip
    if kind == "line":
        body["points"] = [coords[:2], coords[2:]]
    else:
        dwell, occ = ss.get("zdwell") or 0.0, ss.get("zocc") or 0
        body.update(box=coords, dwell_threshold_s=float(dwell) or None, occupancy_threshold=int(occ) or None)
    return body


def as_set(input_id: str, name: str, items: list[dict]) -> regions.RegionSet:
    return regions.parse({"schema_version": regions.SCHEMA_VERSION, "input_id": input_id,
                          "name": name.strip() or "Regions", "regions": items})  # fmt: skip


def zones_tab(run: runs.OpenRun, derived, frames, root: store.DataRoot) -> None:
    if run.manifest is None or frames is None:
        st.info("Events & zones needs a run with verified source time, so legacy runs can be inspected but not "
                "analyzed. Next step: run the video again to analyze it.", icon=":material/info:")  # fmt: skip
        return
    input_id, rid = run.manifest["input_id"], run.run_id
    draft = st.session_state.setdefault(draft_key(input_id), {"regions": [], "name": "Regions"})
    left, right = st.columns([3, 2], gap="medium")
    with right.container(border=True):  # filled first: the preview on the left shows what the editor holds
        editor(run, derived, root, input_id, draft)
    # Draw the shape being edited while placing the first region or once it has a name; otherwise a just-added
    # region would be drawn twice (the editor still holds its coordinates).
    pending = candidate(draft["regions"], preview=not draft["regions"])
    with left.container(border=True):
        st.markdown("#### Regions")
        frame = st.slider("Preview frame", 0, derived.index.count - 1, key=f"zframe-{rid}")
        shown, pend = [], None
        try:
            shown = list(as_set(input_id, draft["name"], draft["regions"]).regions) if draft["regions"] else []
            pend = as_set(input_id, "preview", [*draft["regions"], pending]).regions[-1] if pending else None
        except regions.RegionError:
            pass  # the editor shows the reason; the preview draws what is valid
        try:
            image = render.draw_regions(frames.frame(frame), shown, pend)
            st.image(image, channels="BGR", width="stretch")
        except inspection.InspectError as e:  # a frame that can't be read must not break the editor
            st.warning(f"This frame can't be shown: {e}. Try another frame.", icon=":material/warning:")
        st.caption("Lines point from A to B. Crossing from the right side to the left side (towards “+”) counts "
                   "as negative → positive. Rules use each box's centre.")  # fmt: skip
    analyze_bar(run, derived, frames, root, input_id, draft)


def editor(run, derived, root, input_id: str, draft: dict) -> None:
    st.markdown("#### Edit regions")
    saved = {s.region_set_id: s for s in regions.list_sets(root, input_id)}
    choice = st.selectbox("Region set", [NEW_SET, *saved], key=f"zset-{input_id[7:23]}",
                          format_func=lambda k: "New set" if k == NEW_SET else saved[k].name)  # fmt: skip
    name_key = f"zsetname-{input_id[7:23]}"
    if choice != NEW_SET and draft.get("loaded") != choice:  # load a saved set into the draft once
        draft.update(regions=[r.to_dict() for r in saved[choice].regions], name=saved[choice].name, loaded=choice)
        st.session_state[name_key] = saved[choice].name  # the field renders below, so it can still be set
    if choice == NEW_SET and draft.get("loaded"):
        draft.update(regions=[], name="Regions", loaded=None)
        st.session_state[name_key] = "Regions"
    for i, r in enumerate(draft["regions"]):
        text, remove = st.columns([5, 1], vertical_alignment="center")
        classes = ", ".join(r["classes"]) if r["classes"] else "all classes"
        text.markdown(f"**{r['name']}** · {'line' if r['kind'] == 'line' else 'rectangle'} · {classes}")
        remove.button("", icon=":material/close:", key=f"zrm-{i}", help=f"Remove {r['name']}",
                      on_click=_remove, args=(draft, i))  # fmt: skip
    if not draft["regions"]:
        st.caption("No regions yet. Add a line to count crossings or a rectangle to count entries, exits and time "
                   "spent inside.")  # fmt: skip
    with st.expander("Add a region", expanded=not draft["regions"]):
        st.text_input("Name", key="zname", placeholder="e.g. Crosswalk")
        kind = KINDS[st.segmented_control("Shape", list(KINDS), default="Line", required=True, key="zkind")]
        labels = COORD_LABELS[kind]
        cols = st.columns(2)
        for i in range(4):
            cols[i % 2].number_input(labels[i], 0.0, 100.0, DEFAULTS[kind][i], step=0.5, format="%.1f",
                                     key=f"zc{i}")  # fmt: skip
        classes = sorted({s.class_name for s in derived.tracks.values()})
        st.multiselect("Classes", classes, key="zclasses", placeholder="All classes")
        if kind == "rect":
            a, b = st.columns(2)
            a.number_input("Dwell alert (s, 0 = off)", 0.0, float(regions.MAX_DWELL_S), 0.0, step=0.5, key="zdwell")
            b.number_input("Occupancy alert (0 = off)", 0, regions.MAX_OCCUPANCY, 0, key="zocc")
        st.button("Add region", icon=":material/add:", key="zadd", on_click=_add, args=(input_id, draft))
        if st.session_state.get("zerror"):
            st.error(st.session_state["zerror"], icon=":material/error:")
    st.session_state.setdefault(name_key, draft["name"])
    draft["name"] = st.text_input("Set name", key=name_key)
    if st.button("Save regions", icon=":material/save:", key="zsave", disabled=not draft["regions"]):
        try:
            rs = as_set(input_id, draft["name"], draft["regions"])
        except regions.RegionError as e:
            st.error(str(e), icon=":material/error:")
        else:
            regions.save(root, rs)
            st.caption(f"Saved “{rs.name}”. Saving never analyzes; press Analyze when ready.")


def _add(input_id: str, draft: dict) -> None:
    """Callback (runs before the rerun, so it may clear the Name field): validate, then append."""
    body = candidate(draft["regions"])
    if body is None:
        st.session_state["zerror"] = "Give the region a name first."
        return
    try:
        as_set(input_id, draft["name"], [*draft["regions"], body])
    except regions.RegionError as e:
        st.session_state["zerror"] = str(e)
        return
    draft["regions"].append(body)
    draft["loaded"] = None
    st.session_state.pop("zerror", None)
    st.session_state["zname"] = ""


def _remove(draft: dict, i: int) -> None:
    draft["regions"].pop(i)
    draft["loaded"] = None


def analyze_bar(run, derived, frames, root, input_id: str, draft: dict) -> None:
    """Analyze is the tab's primary action only while the drawn regions have no up-to-date analysis."""
    if not draft["regions"]:
        return
    try:
        rs = as_set(input_id, draft["name"], draft["regions"])
        aid = analysis.analysis_id(analysis.build_spec(run, rs))
    except (regions.RegionError, analysis.AnalysisError) as e:
        st.error(str(e), icon=":material/error:")
        return
    done = aid in analysis.list_analyses(root, run.run_id)
    with st.container(horizontal=True):
        clicked = st.button("Up to date" if done else "Analyze", type="secondary" if done else "primary",
                            icon=":material/check:" if done else ":material/insights:", key="zanalyze",
                            disabled=done)  # fmt: skip
        st.caption("Results below match these regions." if done else "Reads the saved tracking data; the model "
                   "does not run again.")  # fmt: skip
    if clicked:
        with st.spinner("Analyzing saved tracking data…"):
            regions.save(root, rs)
            result = analysis.analyze(run, rs, root)
        st.session_state[f"zresult-{run.run_id}"] = result.analysis_id
        st.rerun()
    if done:
        st.session_state[f"zresult-{run.run_id}"] = aid
        results(run, derived, frames, root, aid)


def results(run, derived, frames, root, aid: str) -> None:
    """Totals, bookmarks, the event table and the selected event's frame. Filters are display-only (fixture 6)."""
    result = analysis.load(root, run.run_id, aid)
    s, rid = result.summary, run.run_id
    names = {r["id"]: r["name"] for r in result.spec["region_set"]["regions"]}
    regions_by_id = {r.id: r for r in regions.parse(result.spec["region_set"]).regions}
    st.markdown("#### Results")
    when = view.when_label(run.summary.started_at) or "this run"
    st.caption(f"For this run only ({when}); other runs of this video need their own analysis. "
               f"{s['events']['total']} events over {s['coverage']['frames']:,} frames · "
               f"{s['unassigned_detections']:,} detections without a track ID are not counted · "
               f"analysis `{aid}`")  # fmt: skip
    per_row = 4
    for start in range(0, len(s["regions"]), per_row):
        for col, r in zip(st.columns(per_row), s["regions"][start : start + per_row], strict=False):
            label, value, caption = view.region_card(r)
            with col:
                st.metric(label, value, border=True)
                st.caption(caption)
    st.caption(view.OCCUPANCY_NOTE)
    if s["bookmarks"]:
        with st.container(horizontal=True):
            for i, b in enumerate(s["bookmarks"]):
                st.button(f"{b['criterion'].capitalize()} · {names.get(b['rule_id'], b['rule_id'])} · "
                          f"{b['t_us'] / 1e6:.1f} s", key=f"zbm-{aid}-{i}", icon=":material/bookmark:",
                          on_click=_select, args=(rid, b["frame"], b["track_id"], b["rule_id"],
                                                  f"Bookmark: {b['criterion']}"))  # fmt: skip
    events = _events(result)
    table, detail = st.columns([3, 2], gap="medium")
    with table.container(border=True):
        st.markdown("#### Events")
        f = event_filter(events, names, run, aid)
        shown = [e for e in events if f.keeps(e)]
        st.session_state[f"zrows-{aid}"] = shown
        if shown:
            st.dataframe([view.event_row(e, names) for e in shown], hide_index=True, width="stretch",
                         selection_mode="single-row", on_select=functools.partial(_select_row, rid, aid),
                         key=f"ztable-{aid}")  # fmt: skip
        st.caption(f"Showing {len(shown):,} of {len(events):,} events. Filters change this view only; the saved "
                   "totals above stay the same.")  # fmt: skip
    with detail.container(border=True):
        selected_event(run, derived, frames, regions_by_id)


def event_filter(events: list[dict], names: dict[str, str], run, aid: str) -> view.EventFilter:
    a, b = st.columns(2)
    rules = a.multiselect("Region", list(names), format_func=names.get, key=f"zf-rule-{aid}", placeholder="All")
    types = b.multiselect("Event", list(view.EVENT_LABELS), format_func=view.EVENT_LABELS.get,
                          key=f"zf-type-{aid}", placeholder="All")  # fmt: skip
    c, d = st.columns(2)
    classes = c.multiselect("Class", sorted({e["class_name"] for e in events if e["class_name"]}),
                            key=f"zf-class-{aid}", placeholder="All")  # fmt: skip
    track = d.number_input("Track #", min_value=0, value=None, step=1, key=f"zf-track-{aid}",
                           placeholder="Any")  # fmt: skip
    end = max(1.0, round(run.summary.frames / run.fps, 1)) if run.summary.frames else 1.0
    window = st.slider("Time (s)", 0.0, end, (0.0, end), step=0.1, key=f"zf-time-{aid}")
    return view.EventFilter(tuple(rules), tuple(types), tuple(classes), int(track) if track is not None else None,
                            None if window == (0.0, end) else window)  # fmt: skip


def selected_event(run, derived, frames, regions_by_id: dict) -> None:
    rid = run.run_id
    sel = st.session_state.get(f"zsel-{rid}")
    st.markdown("#### Selected")
    if not sel or sel["frame"] >= derived.index.count:
        st.caption("Choose an event in the table or a bookmark to see its source frame.")
        return
    frame, track = sel["frame"], sel["track"]
    try:
        still, record = inspection.render_still(frames, derived.index, frame, render.ViewOptions(spotlight=track))
    except inspection.InspectError as e:
        st.warning(f"This frame can't be shown: {e}.", icon=":material/warning:")
        return
    rule = regions_by_id.get(sel["rule"])
    others = [r for r in regions_by_id.values() if r is not rule]
    st.image(render.draw_regions(still, others, rule), channels="BGR", width="stretch")
    who = f"track #{track}" if track is not None else "region count"
    st.caption(f"{sel['label']} · frame {frame:,} at {record['source_t_us'] / 1e6:.2f} s · {who}")
    st.button("Show in Inspect", icon=":material/open_in_new:", key=f"zgo-{rid}", on_click=_show_in_inspect,
              args=(rid, frame, track, track is not None and track in derived.tracks))  # fmt: skip
    if st.session_state.get(f"zgone-{rid}") == frame:
        st.caption(f"Inspect is now on frame {frame:,}. Open the Inspect tab to look around it.")


def _events(result) -> list[dict]:
    cache = st.session_state.setdefault("zevents", {})
    if result.analysis_id not in cache:
        cache.clear()  # one analysis's events at a time: bounded memory
        cache[result.analysis_id] = list(analysis.read_events(result))
    return cache[result.analysis_id]


def _select(rid: str, frame: int, track: int | None, rule: str, label: str) -> None:
    st.session_state[f"zsel-{rid}"] = {"frame": frame, "track": track, "rule": rule, "label": label}


def _select_row(rid: str, aid: str) -> None:
    rows = st.session_state[f"ztable-{aid}"]["selection"]["rows"]
    shown = st.session_state.get(f"zrows-{aid}", [])
    if rows and rows[0] < len(shown):
        e = shown[rows[0]]
        _select(rid, e["confirming_frame"], e["track_id"], e["rule_id"], view.EVENT_LABELS.get(e["type"], e["type"]))


def _show_in_inspect(rid: str, frame: int, track: int | None, known_track: bool) -> None:
    """Callback, so it may set the Inspect tab's widgets, which rendered earlier in this run."""
    st.session_state[f"frame-{rid}"] = frame
    if known_track:
        st.session_state[f"track-{rid}"] = track
    st.session_state[f"zgone-{rid}"] = frame


def export_items(run, root) -> None:
    """Events CSV and the full analysis JSON in the header's Export menu, once this run has a current analysis."""
    aid = st.session_state.get(f"zresult-{run.run_id}")
    if not aid:
        return
    try:
        result = analysis.load(root, run.run_id, aid)
    except analysis.AnalysisError:
        return
    st.download_button("Events · one row per event (CSV)", data=lambda: analysis.export_csv(result).encode(),
                       file_name=f"{run.run_id}-{aid}-events.csv", mime="text/csv", on_click="ignore",
                       icon=":material/download:", width="stretch", key=f"dl-{run.run_id}-events-csv")  # fmt: skip
    st.download_button("Events & zones analysis (JSON)", data=lambda: analysis.export_json(result).encode(),
                       file_name=f"{run.run_id}-{aid}-analysis.json", mime="application/json", on_click="ignore",
                       icon=":material/download:", width="stretch", key=f"dl-{run.run_id}-events-json")  # fmt: skip

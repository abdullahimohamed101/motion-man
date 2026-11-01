"""Events & zones tab, part 1: draw regions, save them, analyze (spec B§6, plan P3 D12).

Layout and state only. Regions are validated by ``regions``, analysis runs in ``analysis`` over saved records, and
nothing here can run the model. Editing or saving regions never analyzes; Analyze is the only action that writes
results, and an identical analysis is reused instead of rebuilt.
"""

from __future__ import annotations

import re

import streamlit as st

from motionman import analysis, inspection, regions, render, runs, store

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
    analyze_bar(run, root, input_id, draft)


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


def analyze_bar(run, root, input_id: str, draft: dict) -> None:
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
        result = analysis.load(root, run.run_id, aid)
        s = result.summary
        st.caption(f"{s['events']['total']} events over {s['coverage']['frames']:,} frames · analysis "
                   f"`{aid}`")  # fmt: skip

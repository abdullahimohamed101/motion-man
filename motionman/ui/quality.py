"""Quality lab: review frames into frozen revisions, evaluate runs, compare runs (spec B§7–B§9; plan P4 D10).

Layout and state only. Datasets are validated and stored by ``datasets``; the review queue comes from ``review``;
the run's records are only read and the model never runs. Every edit reloads the draft from disk, changes it,
validates and saves it, so the page can't drift from the file. Ground truth exists only where the reviewer acted.
"""

from __future__ import annotations

import json
import re

import streamlit as st

from motionman import analysis, datasets, disagree, evaluate, inspection, render, review, runs, store

NEW_STEP = datasets.DEFAULT_STEP


def quality_tab(run: runs.OpenRun, derived, frames, root: store.DataRoot) -> None:
    if run.manifest is None or frames is None:
        st.info("The Quality lab needs a verified run with its source video, so legacy runs can be inspected but not "
                "reviewed. Next step: run the video again.", icon=":material/info:")  # fmt: skip
        return
    section = st.segmented_control("Section", ["Review", "Evaluate", "Compare runs"], default="Review", required=True,
                                   key="qsection", label_visibility="collapsed")  # fmt: skip
    if section == "Evaluate":
        evaluate_section(run, root)
    elif section == "Compare runs":
        compare_section(run, root)
    else:
        review_section(run, derived, frames, root)


def _datasets_for(root: store.DataRoot, input_id: str) -> list[str]:
    out = []
    for did in datasets.list_datasets(root):
        try:
            if datasets.load_draft(root, did)["input_id"] == input_id:
                out.append(did)
        except datasets.DatasetError:
            continue
    return out


def review_section(run, derived, frames, root) -> None:
    input_id = run.manifest["input_id"]
    ids = _datasets_for(root, input_id)
    st.markdown("#### Review")
    st.caption("Mark what is really in each frame. Only reviewed frames, and on each only the classes you mark "
               "complete, are ever scored.")  # fmt: skip
    pick, new = st.columns([3, 2], gap="medium")
    with new.expander("New dataset", expanded=not ids):
        st.text_input("Dataset name", key="qnew-id", placeholder="crosswalk-eval",
                      help="a–z, 0–9 and '-'; it names the dataset in every result")  # fmt: skip
        st.number_input("Review every Nth frame", 1, 1000, NEW_STEP, key="qnew-step")
        st.button("Create dataset", icon=":material/add:", key="qcreate", on_click=_create,
                  args=(root, run, derived.index.count))  # fmt: skip
    if st.session_state.get("qerror"):
        st.error(st.session_state["qerror"], icon=":material/error:")
    if not ids:
        pick.caption("No dataset for this video yet. Create one: it starts with evenly spaced frames and nothing "
                     "labelled.")  # fmt: skip
        return
    key = "qds-" + input_id[7:23]
    if st.session_state.get(key) not in ids:
        st.session_state[key] = st.session_state.get("qcreated") if st.session_state.get("qcreated") in ids else ids[0]
    did = pick.selectbox("Dataset", ids, key=key)
    draft = datasets.load_draft(root, did)
    done = {f["frame"]: f["status"] for f in draft["frames"]}
    reviewed = sum(s == "reviewed" for s in done.values())
    excluded, total = sum(s == "excluded" for s in done.values()), len(draft["selected"])
    pick.caption(
        f"{reviewed} reviewed · {excluded} excluded · {total - len(done)} to review of {total} selected frames"
    )
    frame_key = f"qframe-{did}"
    if st.session_state.get(frame_key) not in draft["selected"]:
        st.session_state[frame_key] = next((f for f in draft["selected"] if f not in done), draft["selected"][0])
    left, right = st.columns([3, 2], gap="medium")
    with left.container(border=True):
        nav, nxt = st.columns([3, 1], vertical_alignment="bottom")
        # Plain labels: Streamlit doesn't refresh the selected option's label after a status change, so a status
        # in the label would go stale; the counts above and the form's status line are always current.
        frame = nav.selectbox("Frame", draft["selected"], key=frame_key, format_func=lambda f: f"frame {f:,}")
        nxt.button("Next to review", key="qnext", on_click=_next, args=(frame_key, draft["selected"], done),
                   width="stretch")  # fmt: skip
        record = derived.index.record(frame)
        entry = datasets.frame_entry(draft, frame)
        anns = [a for a in draft["annotations"] if a["frame"] == frame]
        accepted = frozenset(_accepted(anns))
        pending = _pending_box(run)  # nothing until a box with positive size is typed
        try:
            image = render.draw_review(frames.frame(frame), record["objects"], anns,
                                       entry["ignored"] if entry else [], pending, accepted)  # fmt: skip
            st.image(image, channels="BGR", width="stretch")
        except inspection.InspectError as e:
            st.warning(f"This frame can't be shown: {e}.", icon=":material/warning:")
        st.caption("Grey: the run's predictions. Light boxes: what you marked as really there. Shaded: ignored areas.")
    with right.container(border=True):
        frame_form(run, derived, root, did, draft, frame, record, entry, anns, accepted)
    with st.expander("Review queue: frames worth a look (candidates, not errors)"):
        queue(run, derived, root, did)
    commit_bar(root, did, draft)


def frame_form(run, derived, root, did, draft, frame, record, entry, anns, accepted) -> None:
    status = entry["status"] if entry else None
    st.markdown(f"#### Frame {frame:,}")
    st.caption({"reviewed": ":material/check_circle: Reviewed", "excluded": ":material/block: Excluded",
                None: "Not reviewed yet"}[status])  # fmt: skip
    classes = sorted({s.class_name for s in derived.tracks.values()} | {a["class_name"] for a in draft["annotations"]})
    ckey = f"qclasses-{did}-{frame}"
    if ckey not in st.session_state:
        fallback = st.session_state.get(f"qclasses-last-{did}", [c for c in ("person",) if c in classes])
        st.session_state[ckey] = entry["classes_complete"] if entry and entry["classes_complete"] else fallback
    st.multiselect("Classes fully reviewed on this frame", classes, key=ckey, on_change=_classes_changed,
                   args=(root, did, frame, ckey),
                   help="Every object of these classes is boxed below; other classes aren't scored.")  # fmt: skip
    preds = record["objects"]
    st.markdown(f"**Predictions** ({len(preds)})")
    if preds:
        st.caption("Accept the ones that are right. Once the frame is reviewed, any you leave out count as false "
                   "positives for the classes marked complete.")  # fmt: skip
        st.button("Accept all", key=f"qaccall-{frame}", on_click=_accept, args=(root, did, frame, ckey, preds, run))
    box = st.container(height=min(300, 44 * len(preds) + 8), border=False) if preds else st.container()
    for o in preds:
        row, act = box.columns([4, 1], vertical_alignment="center")
        tid = "—" if o["track_id"] is None else f"#{o['track_id']}"
        row.caption(f"{o['class_name']} {o['confidence']:.2f} · track {tid} · {_box_text(o['bbox_xyxy'])}")
        if o["observation_id"] in accepted:
            act.caption(":material/check: kept")
        else:
            act.button("Accept", key=f"qacc-{frame}-{o['observation_id']}", on_click=_accept,
                       args=(root, did, frame, ckey, [o], run))  # fmt: skip
    st.markdown(f"**Marked as really there** ({len(anns)})")
    for a in anns:
        row, act = st.columns([4, 1], vertical_alignment="center")
        row.caption(f"{a['annotation_id']} · {a['class_name']} · {a['origin']} · {_box_text(a['bbox_xyxy'])}")
        act.button("", icon=":material/close:", key=f"qrm-{a['annotation_id']}", help=f"Remove {a['annotation_id']}",
                   on_click=_remove, args=(root, did, a["annotation_id"]))  # fmt: skip
    with st.expander("Add a missed object or fix a box", expanded=False):
        st.selectbox("Class", classes or ["person"], key="qadd-class")
        a, b = st.columns(2)
        w, h = run.width, run.height
        for i, (label, top) in enumerate((("x1 px", w), ("y1 px", h), ("x2 px", w), ("y2 px", h))):
            (a if i % 2 == 0 else b).number_input(label, 0.0, float(top), 0.0, step=1.0,
                                                  key=f"qadd-{i}")  # fmt: skip
        replace = st.selectbox("Replaces (to fix a box)", [None, *[x["annotation_id"] for x in anns]], key="qadd-fix",
                               format_func=lambda v: "nothing: a missed object" if v is None else v)  # fmt: skip
        st.button("Save box", icon=":material/add_box:", key=f"qadd-save-{frame}", on_click=_add,
                  args=(root, did, frame, ckey, replace))  # fmt: skip
    with st.expander("Ignored areas", expanded=False):
        st.caption("Predictions centred in an ignored area count neither for nor against (crowds too small to box).")
        for i, box in enumerate(entry["ignored"] if entry else []):
            row, act = st.columns([4, 1], vertical_alignment="center")
            row.caption(_box_text(box))
            act.button("", icon=":material/close:", key=f"qign-rm-{frame}-{i}", on_click=_ignore_remove,
                       args=(root, did, frame, ckey, i))  # fmt: skip
        st.button("Ignore the box typed above", key=f"qign-{frame}", on_click=_ignore_add,
                  args=(root, did, frame, ckey))  # fmt: skip
    a, b = st.columns(2)
    a.button("Mark reviewed", icon=":material/check:", key=f"qrev-{frame}", width="stretch", on_click=_mark,
             args=(root, did, frame, ckey, "reviewed"), help="Also for a frame with nothing to box")  # fmt: skip
    b.button("Exclude frame", icon=":material/block:", key=f"qexcl-{frame}", width="stretch", on_click=_mark,
             args=(root, did, frame, ckey, "excluded"), help="Too unclear to judge; never scored")  # fmt: skip


def queue(run, derived, root, did) -> None:
    cache = st.session_state.setdefault("qcands", {})
    if run.run_id not in cache:
        cache.clear()
        cache[run.run_id] = review.candidates(derived.index.records(0, derived.index.count))
    shown, totals = cache[run.run_id]
    st.caption(review.NOTE + " " + " · ".join(f"{review.KIND_LABELS[k]} {n:,}" for k, n in totals.items() if n))
    if not shown:
        st.caption("Nothing stands out in this run.")
        return
    st.dataframe([{"Frame": c.frame, "Reason": review.KIND_LABELS[c.kind], "Detail": c.detail} for c in shown],
                 hide_index=True, width="stretch", height=240, key=f"qqueue-{run.run_id}")  # fmt: skip
    picks = review.frames_of(shown)
    st.button(f"Add these {len(picks)} frames to review", key="qqueue-add", on_click=_select, args=(root, did, picks))


def commit_bar(root, did, draft) -> None:
    revs = datasets.list_revisions(root, did)
    last = datasets.load_revision(root, f"{did}@r{revs[-1]['revision']}") if revs else None
    changed = last is None or (last["frames"], last["annotations"]) != (draft["frames"], draft["annotations"])
    ready = any(f["status"] == "reviewed" for f in draft["frames"])
    with st.container(horizontal=True):
        split = st.segmented_control("Split", ["eval", "tuning"], default="eval", required=True, key=f"qsplit-{did}",
                                     help="eval: frozen for scoring. tuning: for choosing settings.")  # fmt: skip
        st.button("Commit revision" if changed else "Committed", type="primary" if changed and ready else "secondary",
                  icon=":material/lock:", key=f"qcommit-{did}", disabled=not (changed and ready), on_click=_commit,
                  args=(root, did, split))  # fmt: skip
    if revs:
        listed = [
            f"r{r['revision']} {r['split']}, {r['frames_reviewed']} frames, {r['annotations']} boxes" for r in revs
        ]
        st.caption("Revisions (read-only once committed): " + " · ".join(listed))
    if st.session_state.get("qcommitted"):
        st.caption(f"Committed {st.session_state['qcommitted']}. Evaluate runs against it in the Evaluate section.")


# --- callbacks: reload, change, validate, save -------------------------------------------------------------------


def _edit(root, did, change) -> None:
    try:
        draft = datasets.load_draft(root, did)
        change(draft)
        datasets.save_draft(root, draft)
        st.session_state.pop("qerror", None)
    except datasets.DatasetError as e:
        st.session_state["qerror"] = str(e)


def _ensure_reviewed(draft, frame, classes) -> None:
    entry = datasets.frame_entry(draft, frame)
    if not classes:
        raise datasets.DatasetError("choose the classes you fully reviewed on this frame first")
    if entry is None or entry["status"] != "reviewed" or entry["classes_complete"] != sorted(set(classes)):
        datasets.set_frame(draft, frame, "reviewed", classes, entry["ignored"] if entry else [])


def _classes(ckey) -> list[str]:
    return list(st.session_state.get(ckey) or [])


def _create(root, run, count) -> None:
    did = (st.session_state.get("qnew-id") or "").strip()
    try:
        datasets.new_draft(
            root, did, run, datasets.even_frames(count, int(st.session_state.get("qnew-step") or NEW_STEP))
        )
        st.session_state["qcreated"] = did
        st.session_state.pop("qerror", None)
    except datasets.DatasetError as e:
        st.session_state["qerror"] = str(e)


def _next(frame_key, selected, done) -> None:
    current = st.session_state.get(frame_key)
    after = [f for f in selected if f not in done and f > (current or -1)] or [f for f in selected if f not in done]
    if after:
        st.session_state[frame_key] = after[0]


def _classes_changed(root, did, frame, ckey) -> None:
    st.session_state[f"qclasses-last-{did}"] = _classes(ckey)
    entry_reviewed = lambda d: (e := datasets.frame_entry(d, frame)) is not None and e["status"] == "reviewed"  # noqa: E731
    _edit(root, did, lambda d: _ensure_reviewed(d, frame, _classes(ckey)) if entry_reviewed(d) else None)


def _accept(root, did, frame, ckey, objs, run) -> None:
    def change(d):
        _ensure_reviewed(d, frame, _classes(ckey))
        have = set(_accepted([a for a in d["annotations"] if a["frame"] == frame]))
        for o in objs:
            if o["observation_id"] not in have:
                datasets.accept_prediction(d, frame, o)

    _edit(root, did, change)


def _remove(root, did, aid) -> None:
    _edit(root, did, lambda d: datasets.remove_annotation(d, aid))


def _typed_box() -> list[float]:
    return [float(st.session_state.get(f"qadd-{i}", 0.0)) for i in range(4)]


def _add(root, did, frame, ckey, replace) -> None:
    def change(d):
        _ensure_reviewed(d, frame, _classes(ckey))
        if replace:
            datasets.remove_annotation(d, replace)
        origin, note = ("corrected", f"replaces {replace}") if replace else ("added", "")
        datasets.add_annotation(d, frame, st.session_state.get("qadd-class") or "person", _typed_box(), origin, note)

    _edit(root, did, change)


def _ignore_add(root, did, frame, ckey) -> None:
    def change(d):
        _ensure_reviewed(d, frame, _classes(ckey))
        e = datasets.frame_entry(d, frame)
        datasets.set_frame(d, frame, "reviewed", e["classes_complete"], [*e["ignored"], _typed_box()])

    _edit(root, did, change)


def _ignore_remove(root, did, frame, ckey, i) -> None:
    def change(d):
        e = datasets.frame_entry(d, frame)
        datasets.set_frame(
            d, frame, e["status"], e["classes_complete"], [b for j, b in enumerate(e["ignored"]) if j != i]
        )

    _edit(root, did, change)


def _mark(root, did, frame, ckey, status) -> None:
    if status == "reviewed":
        _edit(root, did, lambda d: _ensure_reviewed(d, frame, _classes(ckey)))
    else:
        _edit(root, did, lambda d: datasets.set_frame(d, frame, "excluded", []))


def _select(root, did, picks) -> None:
    _edit(root, did, lambda d: datasets.select_frames(d, picks))


def _commit(root, did, split) -> None:
    try:
        number, _ = datasets.commit(root, did, split or "eval")
        st.session_state["qcommitted"] = f"{did}@r{number} ({split or 'eval'})"
        st.session_state.pop("qerror", None)
    except datasets.DatasetError as e:
        st.session_state["qerror"] = str(e)


# --- helpers -----------------------------------------------------------------------------------------------------


def _accepted(anns) -> list[str]:
    """Observation ids already accepted on this frame (accept notes name the observation)."""
    out = []
    for a in anns:
        m = re.search(r"observation (\S+)$", a.get("note") or "")
        if a["origin"] == "accepted" and m:
            out.append(m.group(1))
    return out


def _pending_box(run) -> list[float] | None:
    x1, y1, x2, y2 = _typed_box()
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def _box_text(box) -> str:
    return "[" + ", ".join(f"{v:.0f}" for v in box) + "]"


# --- Evaluate ----------------------------------------------------------------------------------------------------


def _eval_refs(root, input_id: str) -> list[str]:
    """Frozen eval revisions of datasets for this video, newest first. Tuning revisions are never offered."""
    refs = []
    for did in _datasets_for(root, input_id):
        refs += [f"{did}@r{r['revision']}" for r in datasets.list_revisions(root, did) if r["split"] == "eval"]
    return sorted(refs, key=lambda r: (r.split("@")[0], -int(r.split("@r")[1])))


def evaluate_section(run, root) -> None:
    st.markdown("#### Evaluate")
    refs = _eval_refs(root, run.manifest["input_id"])
    if not refs:
        st.caption("No frozen eval revision for this video yet. Review frames and commit one as eval in Review.")
        return
    run_conf = float(run.runspec["settings"]["inference"]["conf"])
    a, b, c = st.columns([2, 1, 1])
    ref = a.selectbox("Reviewed revision", refs, key="qeval-ref")
    conf = b.number_input("Confidence ≥", run_conf, 1.0, run_conf, step=0.05, key="qeval-conf",
                          help="Can't go below the run's own threshold: it never kept those boxes.")  # fmt: skip
    iou_t = c.number_input("IoU ≥", 0.05, 1.0, evaluate.DEFAULT_IOU, step=0.05, key="qeval-iou")
    try:
        eid = evaluate.evaluation_id(run, ref, datasets.revision_sha256(root, ref), conf, iou_t)
    except datasets.DatasetError as e:
        st.error(str(e), icon=":material/error:")
        return
    done = evaluate.published(root, run.run_id, eid)
    with st.container(horizontal=True):
        clicked = st.button("Up to date" if done else "Evaluate", type="secondary" if done else "primary",
                            icon=":material/check:" if done else ":material/fact_check:", key="qeval-go",
                            disabled=done)  # fmt: skip
        st.caption("Scores this run's saved boxes against the reviewed frames; the model does not run.")
    if clicked:
        try:
            with st.spinner("Scoring against the reviewed frames…"):
                evaluate.evaluate(run, ref, root, conf=conf, iou_threshold=iou_t)
            st.rerun()
        except evaluate.EvaluationError as e:
            st.error(str(e), icon=":material/error:")
    if done:
        evaluation_results(run, evaluate.load(root, run.run_id, eid).doc)


def evaluation_results(run, d: dict) -> None:
    th, tot = d["thresholds"], d["total"]
    st.caption(f"{d['metric']} · {d['dataset']['ref']} (sha {d['dataset']['sha256'][7:19]}…) · confidence ≥ "
               f"{th['confidence']} · IoU ≥ {th['iou']} · {d['evaluator_version']}")  # fmt: skip
    cols = st.columns(4)
    for col, (label, key) in zip(cols[:2], (("Precision", "precision"), ("Recall", "recall")), strict=True):
        value = f"{tot[key]:.3f}" if tot[key] is not None else "—"
        col.metric(label, value, help=tot.get(f"{key}_reason"), border=True)
    cols[2].metric("Matched (TP)", f"{tot['tp']:,}", border=True)
    cols[3].metric("False pos. / missed", f"{tot['fp']:,} / {tot['fn']:,}", border=True)
    f = d["frames"]
    st.caption(f"{f['scored']} frames scored, {f['excluded']} excluded, {len(f['not_evaluated'])} not in this run, "
               f"{f['ignored_predictions']} predictions in ignored areas. {d['scope']}")  # fmt: skip
    rows = [{"Class": cls, "TP": c["tp"], "FP": c["fp"], "FN": c["fn"], "Frames": c["frames"],
             "Precision": c["precision"], "Recall": c["recall"]} for cls, c in d["classes"].items()]  # fmt: skip
    st.dataframe(rows, hide_index=True, width="stretch", key=f"qeval-classes-{d['evaluation_id']}")
    left, right = st.columns(2)
    for col, kind, title in ((left, "false_positives", "False positives"), (right, "false_negatives", "Missed")):
        with col.container(border=True):
            st.markdown(f"**{title}** (examples)")
            examples = d["examples"][kind]
            if not examples:
                st.caption("None.")
            for i, e in enumerate(examples[:6]):
                text = f"frame {e['frame']:,} · {e['class_name']}" + (
                    f" {e['confidence']:.2f}" if "confidence" in e else ""
                )
                st.button(text, key=f"qex-{kind}-{i}", icon=":material/open_in_new:", on_click=_inspect,
                          args=(run.run_id, e["frame"]))  # fmt: skip
    if st.session_state.get(f"qinspect-{run.run_id}") is not None:
        st.caption(f"Inspect is now on frame {st.session_state[f'qinspect-{run.run_id}']:,}. Open the Inspect tab.")


def _inspect(rid: str, frame: int) -> None:
    st.session_state[f"frame-{rid}"] = frame
    st.session_state[f"qinspect-{rid}"] = frame


# --- Compare runs ------------------------------------------------------------------------------------------------


def _same_video_runs(root, run) -> dict[str, tuple[str, str]]:
    """Other published runs of this video: run id → (label, records hash), read from manifests only (cheap)."""
    out = {}
    for s in runs.list_runs(root):
        if s.run_id == run.run_id or s.kind != "canonical":
            continue
        try:
            m = json.loads((s.path / store.MANIFEST).read_text(encoding="utf-8"))
            spec = json.loads((s.path / store.RUNSPEC).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # an unreadable run is simply not offered; opening it would refuse it anyway
        if m.get("input_id") == run.manifest["input_id"] and m.get("records_sha256"):
            imgsz = (spec.get("settings") or {}).get("inference", {}).get("imgsz")
            out[s.run_id] = (f"{(s.started_at or '')[:16].replace('T', ' ')} · imgsz {imgsz} · {s.frames} frames · "
                             f"{s.run_id[-6:]}", m["records_sha256"])  # fmt: skip
    return out


def compare_section(run, root) -> None:
    st.markdown("#### Compare runs")
    others = _same_video_runs(root, run)
    if not others:
        st.caption("No other run of this video to compare with. Run the video again with different settings.")
        return
    a, b = st.columns([3, 1])
    other = a.selectbox("Compare this run with", list(others), key="qcmp-run", format_func=lambda r: others[r][0])
    iou_t = b.number_input("Same box at IoU ≥", 0.05, 1.0, disagree.DEFAULT_IOU, step=0.05, key="qcmp-iou")
    did = disagree.disagreement_id(run.run_id, run.records_sha256, other, others[other][1], iou_t)
    done = disagree.published(root, did)
    with st.container(horizontal=True):
        clicked = st.button("Up to date" if done else "Compare", type="secondary" if done else "primary",
                            icon=":material/check:" if done else ":material/compare:", key="qcmp-go",
                            disabled=done)  # fmt: skip
        st.caption("Matches boxes by position on every frame both runs processed; track IDs are never compared.")
    if clicked:
        try:
            with st.spinner("Comparing the two runs frame by frame…"):
                disagree.compare(run, analysis.find_run(root, other), root, iou_t)
            st.rerun()
        except (disagree.DisagreeError, analysis.AnalysisError) as e:
            st.error(str(e), icon=":material/error:")
    if done:
        comparison_results(run, root, disagree.load(root, did).doc)


def comparison_results(run, root, d: dict) -> None:
    c = d["counts"]
    cols = st.columns(4)
    for col, (label, key) in zip(cols, (("Same box, same class", "matched_same_class"), ("Class differs",
                                 "class_disagreement"), ("Only in this run", "unmatched_a"),
                                 ("Only in the other", "unmatched_b")), strict=True):  # fmt: skip
        col.metric(label, f"{c[key]:,}", border=True)
    iou_p50 = d["matched_iou"]["p50"]
    st.caption(f"{d['frames_compared']:,} frames compared · differences on {d['frames_with_differences']:,} · "
               f"matched IoU p50 {iou_p50 if iou_p50 is not None else '—'}. {d['note']}")  # fmt: skip
    rows = [{"Class": cls, "Same": v["matched_same_class"], "Class differs": v["class_disagreement"],
             "Only this run": v["unmatched_a"], "Only other": v["unmatched_b"]}
            for cls, v in d["by_class"].items()]  # fmt: skip
    st.dataframe(rows, hide_index=True, width="stretch", key=f"qcmp-classes-{d['disagreement_id']}")
    top = d["top_frames"]
    if not top:
        return
    st.markdown("**Most differences**")
    st.caption(", ".join(f"frame {t['frame']:,} ({t['differences']})" for t in top[:10]))
    drafts = _datasets_for(root, run.manifest["input_id"])
    if drafts:
        a, b = st.columns([2, 1], vertical_alignment="bottom")
        target = a.selectbox("Send these frames to review in", drafts, key="qcmp-send-to")
        b.button(f"Send top {min(10, len(top))} frames", key="qcmp-send", width="stretch", on_click=_send,
                 args=(root, target, [t["frame"] for t in top[:10]]))  # fmt: skip
    else:
        st.caption("Create a dataset in Review to send these frames there.")
    if st.session_state.get("qsent"):
        st.caption(st.session_state["qsent"])


def _send(root, did, frames_) -> None:
    _edit(root, did, lambda d: datasets.select_frames(d, frames_))
    if not st.session_state.get("qerror"):
        st.session_state["qsent"] = f"Added {len(frames_)} frames to {did} (nothing labelled; review them in Review)."

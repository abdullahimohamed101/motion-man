"""MotionMan workbench (spec A§7/A§8, UI2): Choose video / Try demo → Run tracking → Inspect → Export.

Layout and state only. Tracking runs in the pipeline via Runner; inspection reads saved evidence through
``runs`` and ``inspection`` and never runs the model. Start with ``scripts/ui.sh``.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from motionman import inspection, presets, records, runs, store
from motionman.presets import Preset, PresetError
from motionman.ui import style, view, workspace
from motionman.ui.runner import Runner, RunSettings, SourceInfo, describe_error, probe, probe_upload
from motionman.video_io import VideoError

# Overridable so tests (and other machines) can point elsewhere.
DEMO_CLIP = Path(os.environ.get("MOTIONMAN_DEMO_CLIP", "clips/crosswalk.mp4"))
LEGACY_RUNS = Path(os.environ.get("MOTIONMAN_LEGACY_RUNS", "runs"))
DEVICES = {"CPU": "cpu", "Auto": "auto", "CUDA": "cuda"}
DEVICE_NAMES = {v: k for k, v in DEVICES.items()}
MODEL_LABEL = "YOLO11n · yolo11n.pt · imgsz 640 · FP32"
LIMITS = "MP4 with a constant frame rate · up to 512 MiB, 10 minutes and 1920×1080"
OPEN_RUNS_KEPT = 3  # opened runs (verified, indexed, with a frame reader) kept per session


@st.cache_resource
def get_runner() -> Runner:
    """One runner per process: a refresh or second tab sees the run in progress (one job at a time)."""
    return Runner()


def probe_cached(key: str, load) -> tuple[SourceInfo | None, str | None]:
    """Probe each source once per session (uploads can be large); returns (info, error message)."""
    cache = st.session_state.setdefault("probes", {})
    if key not in cache:
        try:
            cache[key] = (load(), None)
        except VideoError as e:
            cache[key] = (None, f"{e} {describe_error(e)[1]}")
    return cache[key]


def _signature(run) -> tuple:
    """Size and mtime of every file a run serves; if any changes after opening, the run is verified again."""
    paths = (run.records_path, run.video_path, run.records_path.parent / "report.json", run.source_path)
    return tuple((p.stat().st_size, p.stat().st_mtime_ns) if p and p.is_file() else None for p in paths)


def open_cached(summary: runs.RunSummary, root: store.DataRoot):
    """Open (verify), index and attach a frame reader to a run once per session; keep the last few."""
    cache: dict = st.session_state.setdefault("opened", {})
    if summary.run_id in cache:
        entry = cache.pop(summary.run_id)
        if _signature(entry[0]) == entry[3]:
            cache[summary.run_id] = entry  # most recent last
            return entry[:3]
        if entry[2] is not None:
            entry[2].close()  # a file changed since it was verified: open it from scratch
    run = runs.open_run(summary, root)
    derived = inspection.load_or_build(run.records_path, root, run.records_sha256)
    frames = inspection.FrameSource(run.source_path, run.fps, derived.index.count) if run.source_path else None
    cache[summary.run_id] = (run, derived, frames, _signature(run))
    while len(cache) > OPEN_RUNS_KEPT:
        old_frames = cache.pop(next(iter(cache)))[2]  # the least recently used run
        if old_frames is not None:
            old_frames.close()
    return cache[summary.run_id][:3]


def show_run(run_id: str, *, sync_picker: bool = True) -> None:
    st.session_state["run_id"], st.session_state["page"] = run_id, "result"
    if sync_picker:  # only before the picker renders; afterwards it already holds this value
        st.session_state["run_picker"] = run_id


def new_video() -> None:
    """Back to setup. Clears the picker too, or its old value would immediately reopen that run."""
    st.session_state["page"] = "setup"
    st.session_state["run_picker"] = None


st.set_page_config(page_title="MotionMan", page_icon="◰", layout="wide", initial_sidebar_state="collapsed")
st.html(style.CSS)
runner = get_runner()
state = runner.snapshot()
root = store.data_root()
st.session_state.setdefault("page", "setup")
st.session_state["seen_status"] = state.status
# A run finished: open its result once (also in a tab that didn't start it).
if state.status == "complete" and state.report and st.session_state.get("opened_job") != state.report["run_id"]:
    st.session_state["opened_job"] = state.report["run_id"]
    show_run(state.report["run_id"])

show_legacy = st.session_state.get("show_legacy", False)
shelf = runs.list_runs(root, LEGACY_RUNS if show_legacy else None)
by_id = {r.run_id: r for r in shelf}
page = st.session_state["page"]
current = by_id.get(st.session_state.get("run_id")) if page == "result" else None
if page == "result" and current is None:
    page = st.session_state["page"] = "setup"  # the run disappeared (or legacy runs were hidden)


# --- Header: wordmark, source, run selector, written status, New video ------------------------------------
@st.fragment(run_every=0.5 if state.busy else 2.0)
def status_badge() -> None:
    s = runner.snapshot()
    if s.status != st.session_state.get("seen_status"):
        st.rerun()  # a run started, finished or failed: redraw the whole page
    if s.busy:
        badge = view.BADGES["processing"]
    elif st.session_state.get("page") == "result":
        badge = view.BADGES["complete"]
    elif s.status == "error":
        badge = view.BADGES["error"]
    else:
        badge = view.Badge("Ready", "gray", ":material/radio_button_unchecked:")
    st.badge(badge.label, icon=badge.icon, color=badge.color)


head_left, head_runs, head_status, head_new = st.columns([3, 4, 1.2, 1.4], vertical_alignment="center")
with head_left:
    st.markdown("## MotionMan")
    st.caption(current.display_name if current else "Object Tracking Workbench")
with head_runs:
    ids = [r.run_id for r in shelf]
    if ids:
        if st.session_state.get("run_picker") not in ids:
            st.session_state["run_picker"] = current.run_id if current else None
        choice = st.selectbox(
            "Saved runs",
            ids,
            format_func=lambda rid: by_id[rid].label,
            placeholder="Open a saved run",
            key="run_picker",
        )
        if choice is not None and (current is None or choice != current.run_id):
            show_run(choice, sync_picker=False)
            st.rerun()
    else:
        st.caption("No saved runs yet.")
    st.toggle("Show legacy runs", key="show_legacy", help="Older outputs from before verified runs. Read-only.")
with head_status:
    status_badge()
with head_new:
    if page == "result":
        st.button("New video", icon=":material/add:", on_click=new_video, width="stretch")


# --- Setup: Choose video / Try demo → Run tracking ---------------------------------------------------------
@st.fragment(run_every=0.5 if state.busy else None)
def progress_panel() -> None:
    s = runner.snapshot()
    if s.status != st.session_state.get("seen_status"):
        st.rerun()
    if s.busy:
        fraction, line = view.progress(s)
        if fraction is None:
            st.caption(f":material/progress_activity: Processing · {line}")
        else:
            st.progress(fraction, text=f"Processing · {line}")


def setup_page() -> None:
    st.markdown("### Choose a video to track objects, or try the demo.")
    st.caption(LIMITS)
    demo_ok = DEMO_CLIP.is_file()
    options = (["Try demo"] if demo_ok else []) + ["Choose video"]
    mode = st.segmented_control("Source", options, default=options[0], required=True, key="mode")
    info, source_error, start_kwargs = None, None, None
    if mode == "Try demo":
        info, source_error = probe_cached(f"demo:{DEMO_CLIP.stat().st_mtime_ns}", lambda: probe(DEMO_CLIP))
        start_kwargs = {"source": DEMO_CLIP}
    else:
        if not demo_ok:
            st.caption(f"The demo clip isn't at `{DEMO_CLIP}` (see README → Sample clip).")
        upload = st.file_uploader("Video file", type=["mp4", "m4v", "mov"], key="upload")
        if upload is not None:
            data = upload.getvalue()
            info, source_error = probe_cached(f"upload:{upload.file_id}", lambda: probe_upload(upload.name, data))
            start_kwargs = {"upload": (upload.name, data)}
    if info is not None:
        st.caption("  \n".join(view.source_lines(info)))
    if source_error:
        st.error(source_error, icon=":material/error:")
        start_kwargs = None

    with st.expander("Advanced settings", expanded=False):
        names = preset_names("tracking")
        if names:
            chosen = st.selectbox("Apply a tracking preset", [None, *names], format_func=lambda n: n or "None",
                                  key="tracking_preset")  # fmt: skip
            if chosen and st.session_state.get("applied_preset") != chosen:
                try:
                    p = presets.load(root, chosen)
                    if p.tracking is not None:
                        st.session_state["device"] = DEVICE_NAMES[p.tracking.device]
                        st.session_state["conf"] = p.tracking.conf
                except PresetError as e:
                    st.error(str(e), icon=":material/error:")
                st.session_state["applied_preset"] = chosen
                st.rerun()
        st.session_state.setdefault("device", "CPU")
        st.session_state.setdefault("conf", 0.25)
        device = st.segmented_control("Device", list(DEVICES), required=True, key="device")
        conf = st.slider(
            "Detection threshold", 0.05, 0.95, step=0.05, key="conf",
            help="Minimum detection score. This filters model outputs; it is not an accuracy percentage.",
        )  # fmt: skip
        st.caption(f"Model: {MODEL_LABEL}. These settings apply to your next run.")
    if st.button("Run tracking", type="primary", disabled=state.busy or start_kwargs is None):
        if runner.start(RunSettings(device=DEVICES[device], conf=conf), **start_kwargs):
            st.session_state.pop("opened_job", None)
        else:
            st.session_state["refused"] = True
        st.rerun()
    if st.session_state.pop("refused", False):
        st.warning("A run is already in progress.")
    progress_panel()
    if state.status == "error":
        st.error(state.message or "The run failed.", icon=":material/error:")
        st.markdown(f"**Next step:** {state.next_step}")


# --- Result workspace: Inspect, Performance, Export ----------------------------------------------------------
def bind(key: str, value):
    """Restore a widget's value from the run's saved context the first time it renders in this run."""
    if key not in st.session_state:
        st.session_state[key] = value
    return key


def result_page(summary: runs.RunSummary) -> None:
    try:
        run, derived, frames = open_cached(summary, root)
    except (runs.RunError, records.RecordError, OSError) as e:  # RecordError covers InspectError
        st.error(f"This run can't be opened: {e}", icon=":material/error:")
        st.markdown("**Next step:** Choose another saved run, or start a new video.")
        return
    rid, count = run.run_id, derived.index.count
    if count == 0:
        st.warning("This run has no frame records, so there is nothing to inspect.", icon=":material/warning:")
        return
    contexts: dict = st.session_state.setdefault("contexts", {})
    ctx = workspace.restore(contexts.get(rid), count, derived.tracks)

    top_left, top_right = st.columns([5, 1], vertical_alignment="center")
    with top_left:
        note = " · source unverified (legacy run)" if not run.source_verified else ""
        st.caption(f"Run `{rid}` · {count:,} frames · {run.width}×{run.height} at {run.fps:.3f} fps playback{note}")
    inspect_tab, perf_tab = st.tabs(["Inspect", "Performance"])

    with inspect_tab:
        left, right = st.columns([3, 1], gap="large")
        with right:
            st.markdown("#### Object")
            order = workspace.track_order(derived.tracks)
            track = st.selectbox(
                "Selected object", [None, *order], key=bind(f"track-{rid}", ctx.track),
                format_func=lambda t: "None" if t is None else workspace.track_label(derived.tracks[t]),
            )  # fmt: skip
        with left:
            frame_key = bind(f"frame-{rid}", ctx.frame)
            st.markdown("#### Playback")
            choice = st.segmented_control("Video", ["Tracked", "Original"], default="Tracked", required=True,
                                          key=f"playback-{rid}")  # fmt: skip
            if choice == "Original" and run.source_path is None:
                st.caption("The original video for this legacy run isn't available.")
            else:
                st.video(str(run.video_path if choice == "Tracked" else run.source_path))
            st.caption("Native playback. It is not synchronized with the inspected frame below.")
            st.markdown("#### Inspected frame")
            frame = st.slider("Source frame", 0, count - 1, key=frame_key)
            with st.expander("View options", expanded=False):
                vd = ctx.view
                vd["labels"] = st.toggle("Labels", key=bind(f"labels-{rid}", vd["labels"]))
                vd["trail_length"] = st.slider(
                    "Trail length (frames)", 0, 60, key=bind(f"trail-{rid}", vd["trail_length"])
                )
                vd["trail_opacity"] = st.slider("Trail opacity", 0.1, 1.0, step=0.1,
                                                key=bind(f"opacity-{rid}", max(0.1, vd["trail_opacity"])))  # fmt: skip
                classes = sorted({s.class_name for s in derived.tracks.values()})
                vd["classes"] = st.multiselect("Classes", classes, key=bind(f"classes-{rid}", vd["classes"] or []),
                                               placeholder="All classes") or None  # fmt: skip
                spotlight = st.toggle("Spotlight the selected object", key=bind(f"spot-{rid}", ctx.spotlight))
                st.caption("View updated. Saved results are unchanged.")
                view_presets(vd, rid)
            options = workspace.view_options(vd, spotlight=track if spotlight else None)
            record = derived.index.record(frame)
            if frames is None:
                st.info("The source video for this legacy run isn't available, so exact frames can't be shown.")
            else:
                try:
                    still, _ = inspection.render_still(frames, derived.index, frame, options)
                    st.image(still, channels="BGR", width="stretch",
                             caption=workspace.frame_caption(frame, record["source_t_us"], count))  # fmt: skip
                except inspection.InspectError as e:
                    st.warning(f"{e}. Try another frame.", icon=":material/warning:")
        with right:
            object_panel(run, derived, frames, track, record, frame_key)
            unassigned = sum(1 for o in record["objects"] if o["track_id"] is None)
            st.caption(f"Unassigned detections in this frame: {unassigned}")
        contexts[rid] = workspace.RunContext(frame, track, dict(vd), spotlight)

    with perf_tab:
        perf_panel(run, summary)

    with top_right:
        export_menu(run, derived, frames, frame, options, record)


def object_panel(run, derived, frames, track, record, frame_key) -> None:
    if track is None:
        st.caption("Select an object to see when it was observed.")
        return
    s = derived.tracks[track]
    st.markdown(f"**#{s.track_id} {s.class_name}**")
    if inspection.observed(record, track):
        st.caption(":material/check_circle: Observed in this frame")
    else:
        st.caption(":material/visibility_off: Not observed in this frame")
    st.caption(f"Observed {s.observations:,} frames: {workspace.intervals_text(s, run.fps)}")
    st.caption("Track IDs are local to this run and can fragment; they are not counts of people.")
    go_first, go_last = st.columns(2)
    go_first.button("First seen", on_click=lambda: st.session_state.__setitem__(frame_key, s.first_frame),
                    key=f"first-{run.run_id}-{track}", width="stretch")  # fmt: skip
    go_last.button("Last seen", on_click=lambda: st.session_state.__setitem__(frame_key, s.last_frame),
                   key=f"last-{run.run_id}-{track}", width="stretch")  # fmt: skip
    with st.expander("Details", expanded=False):
        st.caption(
            f"Confidence p10 / p50 / p90: {s.confidence_p10:.2f} / {s.confidence_p50:.2f} / {s.confidence_p90:.2f}"
        )
        st.caption("Classes: " + ", ".join(f"{c} ×{n}" for c, n in sorted(s.class_counts.items())))
        gaps = ", ".join(f"{a / run.fps:.1f}–{b / run.fps:.1f} s" for a, b in s.gaps[:6]) or "none"
        st.caption(f"Gaps: {gaps}")
    if frames is not None:
        with st.expander("Contact sheet", expanded=False):
            if st.button("Show contact sheet", key=f"sheet-{run.run_id}-{track}"):
                crops, caps = [], []
                try:
                    for f in workspace.contact_frames(s):
                        obj = next(o for o in derived.index.record(f)["objects"] if o["track_id"] == track)
                        x1, y1, x2, y2 = workspace.crop_box(obj["bbox_xyxy"], run.width, run.height)
                        crops.append(frames.frame(f)[y1:y2, x1:x2])
                        caps.append(f"{f / run.fps:.1f} s")
                except inspection.InspectError as e:
                    st.warning(f"Contact sheet unavailable: {e}", icon=":material/warning:")
                else:
                    st.image(crops, channels="BGR", caption=caps, width=80)
                    st.caption("Samples come only from frames where the object was observed.")


def perf_panel(run, summary) -> None:
    report = run.report
    if not report.get("timing"):
        st.caption("This run has no timing report.")
        return
    for col, m in zip(st.columns(4), view.report_metrics(report), strict=True):
        col.metric(m.label, m.value, help=m.help, border=True)
    st.caption(f"Source playback: {run.fps:.3f} fps. Processing FPS above is work completed per wall second.")
    with st.expander("Run details", expanded=False):
        rows = view.report_details(report, summary.display_name, summary.path)
        rows.insert(0, ("Run ID", run.run_id))
        if run.manifest:
            rows += [("Manifest", f"{run.manifest['status']} · config {run.manifest['config_sha256'][:19]}…"),
                     ("Input", f"{run.manifest['input_id'][:19]}… (verified)")]  # fmt: skip
        for k, v in rows:
            st.markdown(f"**{k}** · `{v}`")


def export_menu(run, derived, frames, frame, options, record) -> None:
    with st.popover("Export", type="primary", icon=":material/ios_share:", width="stretch"):
        st.caption(f"From run `{run.run_id}`. Audio is not included in any export.")
        files = (
            (run.video_path, "Annotated video", "video/mp4", "annotated.mp4"),
            (
                run.records_path,
                "Tracking data · per-frame object records (JSONL)",
                "application/x-ndjson",
                "tracks.jsonl",
            ),
            (run.records_path.parent / "report.json", "Run report", "application/json", "report.json"),
        )
        for path, label, mime, name in files:
            if path.is_file():
                st.download_button(label, data=lambda p=path: p.read_bytes(), file_name=name, mime=mime,
                                   on_click="ignore", icon=":material/download:", width="stretch",
                                   key=f"dl-{run.run_id}-{name}")  # fmt: skip
            else:
                st.caption(f"{label}: not available for this run.")
        st.divider()
        st.caption(f"Evidence image: source frame {frame:,} ({workspace.format_t(record['source_t_us'])}) "
                   "with the current view options.")  # fmt: skip
        if frames is None:
            st.caption("Not available: the source video for this legacy run is missing.")
        elif st.button("Create evidence PNG", key=f"evidence-{run.run_id}", width="stretch"):
            try:
                png = inspection.export_evidence(
                    run_id=run.run_id, records_sha256=derived.records_sha256,
                    input_id=run.manifest["input_id"] if run.manifest else None, source_verified=run.source_verified,
                    frames=frames, index=derived.index, frame=frame, options=options, root=root,
                )  # fmt: skip
                st.session_state[f"png-{run.run_id}"] = str(png)
            except (inspection.InspectError, OSError) as e:
                st.warning(f"Couldn't create the evidence PNG: {e}", icon=":material/warning:")
        made = st.session_state.get(f"png-{run.run_id}")
        if made and Path(made).is_file():
            st.download_button("Download evidence PNG", data=lambda p=Path(made): p.read_bytes(),
                               file_name=f"{run.run_id}-evidence.png", mime="image/png", on_click="ignore",
                               icon=":material/download:", width="stretch", key=f"dl-png-{run.run_id}")  # fmt: skip


def preset_names(kind: str) -> list[str]:
    """Saved presets that carry ``kind`` ("view" or "tracking") settings; unreadable ones are left out."""
    names = []
    for name in presets.list_names(root):
        try:
            if getattr(presets.load(root, name), kind) is not None:
                names.append(name)
        except PresetError:
            pass
    return names


def apply_view_preset(rid: str) -> None:
    """Callback: write a view preset into this run's view widgets before they render. Only re-renders."""
    name = st.session_state.get(f"view_preset-{rid}")
    if not name:
        return
    try:
        v = presets.load(root, name).view
    except PresetError as e:
        st.session_state["preset_error"] = str(e)
        return
    if v is None:
        return
    classes = sorted({s.class_name for s in st.session_state["opened"][rid][1].tracks.values()})
    st.session_state[f"labels-{rid}"] = v.labels
    st.session_state[f"trail-{rid}"] = v.trail_length
    st.session_state[f"opacity-{rid}"] = max(0.1, v.trail_opacity)
    st.session_state[f"classes-{rid}"] = [c for c in (v.classes or []) if c in classes]


def view_presets(vd: dict, rid: str) -> None:
    names = preset_names("view")
    if names:
        st.selectbox("Apply a view preset", [None, *names], format_func=lambda n: n or "None",
                     key=f"view_preset-{rid}", on_change=apply_view_preset, args=(rid,))  # fmt: skip
    if err := st.session_state.pop("preset_error", None):
        st.error(err, icon=":material/error:")
    cols = st.columns([3, 2])
    name = cols[0].text_input("Save view as preset", placeholder="busy-street", key="preset_name")
    replace = st.checkbox("Replace a preset with the same name", key="preset_replace")
    if cols[1].button("Save view", key="save_view") and name:
        try:
            presets.save(root, Preset(name, view=workspace.view_options(vd)), overwrite=replace)
            st.toast(f"Saved view preset {name}")
        except PresetError as e:
            st.error(str(e), icon=":material/error:")


if page == "result" and current is not None:
    result_page(current)
else:
    setup_page()

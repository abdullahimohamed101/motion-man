"""MotionMan workbench page. Layout and state only; inference runs in the pipeline via Runner (spec §9).

Start with ``scripts/ui.sh``.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from motionman.ui import style, view
from motionman.ui.runner import Runner, RunSettings, SourceInfo, describe_error, probe, probe_upload
from motionman.video_io import VideoError

# Overridable so tests (and other machines) can point at a different clip.
DEMO_CLIP = Path(os.environ.get("MOTIONMAN_DEMO_CLIP", "clips/crosswalk.mp4"))
DEVICES = {"CPU": "cpu", "Auto": "auto", "CUDA": "cuda"}
MODEL_LABEL = "YOLO11n · yolo11n.pt · imgsz 640 · FP32"
DOWNLOADS = (
    ("annotated.mp4", "Annotated video", "video/mp4"),
    ("tracks.jsonl", "Tracking JSONL", "application/x-ndjson"),
    ("report.json", "Report JSON", "application/json"),
)


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


st.set_page_config(page_title="MotionMan", page_icon="◰", layout="wide", initial_sidebar_state="expanded")
st.html(style.CSS)
runner = get_runner()
state = runner.snapshot()
st.session_state["seen_status"] = state.status

# --- Controls sidebar -------------------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Source")
    mode = st.segmented_control("Input", ["Demo clip", "Upload MP4"], default="Demo clip", required=True, key="mode")
    info, source_error, start_kwargs, original = None, None, None, None
    if mode == "Demo clip":
        if DEMO_CLIP.is_file():
            info, source_error = probe_cached(f"demo:{DEMO_CLIP.stat().st_mtime_ns}", lambda: probe(DEMO_CLIP))
            start_kwargs, original = {"source": DEMO_CLIP}, str(DEMO_CLIP)
        else:
            st.caption(f"Demo clip not found at `{DEMO_CLIP}`. See README → Sample clip, or upload an MP4.")
    else:
        upload = st.file_uploader("MP4 file", type=["mp4", "m4v", "mov"], key="upload")
        if upload is not None:
            data = upload.getvalue()
            info, source_error = probe_cached(f"upload:{upload.file_id}", lambda: probe_upload(upload.name, data))
            start_kwargs, original = {"upload": (upload.name, data)}, data
    if info is not None:
        st.caption("  \n".join(view.source_lines(info)))
    if source_error:
        st.error(source_error, icon=":material/error:")
        start_kwargs = None

    st.subheader("Settings")
    device = st.segmented_control("Device", list(DEVICES), default="CPU", required=True, key="device")
    conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05, key="conf")
    st.caption(f"Model: {MODEL_LABEL}")
    if st.button("Run tracking", type="primary", disabled=state.busy or start_kwargs is None, width="stretch"):
        if runner.start(RunSettings(device=DEVICES[device], conf=conf), **start_kwargs):
            st.session_state["view"] = "Original"
            st.rerun()
        st.warning("A run is already in progress.")


# --- Header and live status (polls the runner while a run is in progress) ---------------------------------
@st.fragment(run_every=0.5 if state.busy else None)
def live_status() -> None:
    s = runner.snapshot()
    if s.status != st.session_state.get("seen_status"):
        st.rerun()  # a run finished or failed: redraw the whole page (results, Run button, downloads)
    left, right = st.columns([3, 2], vertical_alignment="center")
    with left:
        st.markdown("## MotionMan")
        st.caption("Object Tracking Workbench")
    with right:
        badge = view.BADGES[s.status]
        with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center"):
            st.caption(view.device_label(s, device))
            st.badge(badge.label, icon=badge.icon, color=badge.color)
    if s.busy:
        fraction, line = view.progress(s)
        if fraction is None:
            st.caption(f":material/progress_activity: Processing · {line}")
        else:
            st.progress(fraction, text=f"Processing · {line}")


live_status()

# --- Video workspace --------------------------------------------------------------------------------------
annotated = state.artifacts.get("annotated.mp4") if state.status == "complete" else None
st.session_state.setdefault("view", "Original")  # set here, not as a default: Run also resets it
picked = st.segmented_control("View", ["Original", "Tracked"], required=True, key="view")
with st.container(border=True, key="video-panel"):
    if picked == "Tracked" and annotated is not None:
        st.caption(f"Tracked · {state.source_name}")
        st.video(str(annotated))
    elif picked == "Tracked":
        st.caption("The tracked video appears here when a run completes.")
    elif original is not None:
        st.video(original)
    else:
        st.caption("Choose the demo clip or upload an MP4 to preview it here.")

# --- Results ----------------------------------------------------------------------------------------------
if state.status == "error":
    st.error(state.message or "The run failed.", icon=":material/error:")
    st.markdown(f"**Next step:** {state.next_step}")

results_for = f" · {state.source_name}" if state.status == "complete" else ""
st.markdown(f"#### Metrics{results_for}")
for col, metric in zip(st.columns(4), view.metrics(state), strict=True):
    col.metric(metric.label, metric.value, help=metric.help, border=True)

st.markdown("#### Exports")
for col, (name, label, mime) in zip(st.columns(3), DOWNLOADS, strict=True):
    path = state.artifacts.get(name)
    col.download_button(
        label,
        data=(lambda p=path: p.read_bytes()) if path else b"",
        file_name=name,
        mime=mime,
        on_click="ignore",  # downloading never reruns the page, let alone inference
        disabled=path is None,
        icon=":material/download:",
        width="stretch",
        key=f"dl-{name}",
    )

with st.expander("Run details", expanded=False):
    details = view.run_details(state)
    if not details:
        st.caption("Settings, versions and hardware for a run appear here once it has a report.")
    for key, value in details:
        st.markdown(f"**{key}** · `{value}`")

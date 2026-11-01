"""MotionMan workbench page. Layout and state only; inference lives in the pipeline (spec §9).

Start with ``scripts/ui.sh``.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="MotionMan", page_icon="◰", layout="wide", initial_sidebar_state="expanded")

st.title("MotionMan")
st.caption("Object Tracking Workbench · Idle")

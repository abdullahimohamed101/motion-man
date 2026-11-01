"""The style rules theme config can't express: top bar, cards, type scale, the one lime action.

Tokens: near-black surfaces (void / carbon / obsidian / slate), grey type (paper / mist / fog / ash), one acid-lime
accent for the single primary action per view, hairline borders instead of shadows, 6/12 px radii, a 4 px grid.
Selectors are limited to Streamlit's data-testid attributes and the documented ``st-key-<key>`` classes;
Streamlit is pinned because these names can change between versions.
"""

import base64
import html
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"
FAVICON = str(ASSETS / "favicon.png")
# st.html sanitizes inline <svg> away, so the glyph travels as a data URI.
LOGO_URI = "data:image/svg+xml;base64," + base64.b64encode((ASSETS / "logo.svg").read_bytes()).decode("ascii")

CSS = """
<style>
:root {
  --mm-void: #08090a;
  --mm-carbon: #0f1011;
  --mm-obsidian: #161718;
  --mm-slate: #23252a;
  --mm-graphite: #23252a;
  --mm-smoke: #383b3f;
  --mm-paper: #ffffff;
  --mm-mist: #d0d6e0;
  --mm-fog: #8a8f98;
  --mm-ash: #62666d;
  --mm-lime: #e4f222;
  --mm-lime-hover: #eef66a;
  --mm-lime-pressed: #cdd91f;
  --mm-mono: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
  --mm-hairline: inset 0 0 0 1px var(--mm-graphite);
}

/* Type: Inter features, no weight above 590, headings in paper, body in mist/fog/ash. */
.stApp { font-feature-settings: "cv01", "ss03", "zero"; }
.stApp :is(h1, h2, h3, h4, h5, h6) { color: var(--mm-paper); letter-spacing: -0.012em; }
.stApp h2 { letter-spacing: -0.022em; }
.stApp :is(h4, h5, h6) { letter-spacing: -0.006em; }
.stApp :is(b, strong) { font-weight: 590; }
.stApp :is(code, kbd, pre) { font-family: var(--mm-mono); font-variant-numeric: tabular-nums; }
.stApp code { color: var(--mm-mist); border: 0; }

/* Page frame: no Streamlit chrome, one content width, 24 px gutters on a 4 px grid. */
[data-testid="stHeader"] { display: none; }
[data-testid="stMainBlockContainer"] { max-width: 1280px; padding: 0 24px 48px; }
[data-testid="stHeaderActionElements"] { display: none; }  /* headings are labels, not anchors */
[data-testid="stCaptionContainer"] { color: var(--mm-fog); }
[data-testid="stCaptionContainer"] p { font-size: 13px; line-height: 1.5; }
[data-testid="stWidgetLabel"] p { font-size: 12px; line-height: 1.4; color: var(--mm-fog); }
[data-testid="stMarkdownContainer"] a { text-decoration: none; }

/* Top bar: sticky, carbon, hairline bottom border, bleeds to the gutters. Logo + wordmark left, controls right. */
[data-testid="stLayoutWrapper"]:has(> .st-key-topbar) {
  position: sticky; top: 0; z-index: 99;  /* the wrapper, not the block: a sticky needs room inside its parent */
  margin: 0 -24px; padding: 12px 24px;
  background: var(--mm-carbon);
  /* Full bleed: the content stays on the 1280 px grid while the fill and the hairline reach both window edges. */
  box-shadow: 0 0 0 100vmax var(--mm-carbon), 0 1px 0 100vmax var(--mm-graphite);
  clip-path: inset(0 -100vmax -1px);
}
@media (max-width: 1000px) {  /* the toggle keeps its help tooltip; a clipped "L…" label reads as a glitch */
  .st-key-topbar [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] { display: none; }
}
.st-key-topbar [data-testid="stHorizontalBlock"] { gap: 8px; }
.st-key-topbar button { padding-left: 10px; padding-right: 10px; }
@media (max-width: 1150px) { .mm-brand-source, .mm-brand-sep { display: none; } }
.mm-brand { display: flex; align-items: center; gap: 10px; height: 40px; }
.mm-brand img { width: 24px; height: 24px; flex: none; display: block; }
.mm-brand-name { font-size: 16px; font-weight: 510; color: var(--mm-paper); letter-spacing: -0.011em; }
.mm-brand-sep { width: 1px; height: 16px; background: var(--mm-smoke); margin: 0 2px; }
.mm-brand-source {
  font-size: 13px; color: var(--mm-fog); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.st-key-topbar [data-testid="stSelectbox"] [data-baseweb="select"] > div { background: var(--mm-obsidian); }
.st-key-topbar [data-testid="stCheckbox"] { margin-top: 2px; }
.st-key-topbar [data-testid="stCheckbox"] label { gap: 8px; }
.st-key-topbar [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p { font-size: 13px; color: var(--mm-mist); }
.st-key-topbar [data-testid="stMarkdownContainer"]:has(> p > span[class*="-badge"]) { text-align: center; }
.st-key-topbar span[class*="-badge"] {
  display: inline-flex; align-items: center; height: 32px; padding: 0 12px;
  border-radius: 9999px; font-size: 12px; background: rgba(255, 255, 255, 0.05); white-space: nowrap;
}

/* Page title row: title left, metadata right, below the top bar. */
.st-key-pagehead { padding: 24px 0 4px; }
.st-key-pagehead h2, .st-key-pagehead h3 { padding: 0; margin: 0; }
.st-key-pagehead [data-testid="stCaptionContainer"] { padding-top: 4px; }

/* Cards: carbon, 12 px radius, hairline inset border, 24 px padding, no shadow. */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {
  background: var(--mm-carbon);
  border: 0; border-radius: 12px;
  box-shadow: var(--mm-hairline);
  padding: 24px;
}
/* Nested blocks never get a second frame; the card title row sits flush at the top. */
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] {
  background: transparent; box-shadow: none; padding: 0;
}
[data-testid="stVerticalBlockBorderWrapper"] h4 { padding: 0 0 4px; margin: 0; font-size: 16px; }

/* Expanders: hairline frame, obsidian when open, headers in mist at 14 px. */
[data-testid="stExpander"] details {
  border: 1px solid var(--mm-graphite); border-radius: 6px; background: transparent;
}
[data-testid="stExpander"] details[open] { background: rgba(255, 255, 255, 0.02); }
[data-testid="stExpander"] summary { color: var(--mm-mist); font-size: 14px; }
[data-testid="stExpander"] summary:hover { color: var(--mm-paper); }

/* Tabs: an underline-only strip; the active tab is paper with a lime indicator. */
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid var(--mm-graphite); }
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }
[data-testid="stTabs"] button[role="tab"] { padding: 10px 0; color: var(--mm-fog); }
[data-testid="stTabs"] button[role="tab"] p { font-size: 14px; font-weight: 510; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--mm-paper); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: var(--mm-lime); height: 2px; }
[data-testid="stTabs"] [data-baseweb="tab-panel"] { padding-top: 24px; }

/* The one lime action per view: dark text on lime, visible hover and pressed states, never bold. */
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primary"] p { color: var(--mm-void) !important; font-weight: 510; }
[data-testid="stBaseButton-primary"]:not(:disabled):hover {
  background: var(--mm-lime-hover); border-color: var(--mm-lime-hover);
}
[data-testid="stBaseButton-primary"]:not(:disabled):active {
  background: var(--mm-lime-pressed); border-color: var(--mm-lime-pressed);
}
[data-testid="stBaseButton-primary"]:disabled { background: var(--mm-obsidian); border-color: var(--mm-graphite); }
[data-testid="stBaseButton-primary"]:disabled p { color: var(--mm-ash) !important; }

/* Ghost buttons: transparent, hairline, mist text; slate tint on hover. */
[data-testid="stBaseButton-secondary"] {
  background: transparent; border: 1px solid var(--mm-graphite); color: var(--mm-mist);
}
[data-testid="stBaseButton-secondary"]:not(:disabled):hover {
  background: var(--mm-slate); border-color: var(--mm-smoke); color: var(--mm-paper);
}
[data-testid="stBaseButton-secondary"] p { font-size: 13px; }
.stApp button { font-weight: 400; }

/* Segmented controls: hairline pills; the selected segment is paper on slate (lime is reserved for actions). */
[data-testid="stButtonGroup"] button { border-color: var(--mm-graphite); color: var(--mm-fog); }
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[aria-pressed="true"] {
  background: var(--mm-slate); border-color: var(--mm-smoke) !important; color: var(--mm-paper) !important;
}
[data-testid="stButtonGroup"] button[aria-checked="true"] p,
[data-testid="stButtonGroup"] button[aria-pressed="true"] p { color: var(--mm-paper); }

/* Inputs: faint fill, hairline, mist text; the focus border lightens rather than glows. */
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"],
[data-testid="stFileUploaderDropzone"] {
  background: rgba(255, 255, 255, 0.02); border-color: rgba(255, 255, 255, 0.08);
}
[data-baseweb="select"] > div:focus-within, [data-baseweb="input"]:focus-within { border-color: var(--mm-mist); }
[data-testid="stFileUploaderDropzone"] { border: 1px dashed var(--mm-smoke); border-radius: 6px; }

/* Keyboard focus: 2 px lime outline with 2 px offset, never removed for appearance. */
.stApp :is(button, input, select, textarea, a, [role="radio"], [role="slider"], [role="tab"]):focus-visible {
  outline: 2px solid var(--mm-lime) !important;
  outline-offset: 2px;
}

/* Metrics: cards with a fog label and a mono, tabular value in paper. */
[data-testid="stMetric"] {
  background: var(--mm-carbon); border: 0; border-radius: 12px; box-shadow: var(--mm-hairline); padding: 16px 20px;
}
[data-testid="stMetricLabel"] p { font-size: 12px; color: var(--mm-fog); }
[data-testid="stMetricValue"] {
  font-family: var(--mm-mono); font-variant-numeric: tabular-nums; font-weight: 510; color: var(--mm-paper);
  letter-spacing: -0.012em;
}

/* Data tables and images sit flush in their card with hairline frames; image captions align left like everything. */
[data-testid="stDataFrame"] { border: 1px solid var(--mm-graphite); border-radius: 6px; overflow: hidden; }
[data-testid="stImageCaption"] { text-align: left; color: var(--mm-fog); font-size: 13px; padding-top: 8px; }
[data-testid="stImage"] img, [data-testid="stVideo"] video { border-radius: 6px; }

/* Popovers and the progress bar follow the same surfaces. */
[data-baseweb="popover"] > div {
  background: var(--mm-carbon); border: 1px solid var(--mm-graphite); border-radius: 12px;
}
[data-testid="stProgress"] > div > div > div > div { background: var(--mm-lime); }
[data-testid="stProgress"] p { color: var(--mm-fog); font-size: 13px; }
[data-testid="stAlert"] { border-radius: 6px; }
</style>
"""


def brand_html(source: str | None) -> str:
    """The top-bar brand: glyph, wordmark and, when a run is open, its source name after a hairline separator."""
    name = html.escape(source) if source else ""  # filenames are display data (A§10)
    tail = f'<span class="mm-brand-sep"></span><span class="mm-brand-source">{name}</span>' if source else ""
    return (
        f'<div class="mm-brand"><img src="{LOGO_URI}" alt="" width="24" height="24">'
        f'<span class="mm-brand-name">MotionMan</span>{tail}</div>'
    )

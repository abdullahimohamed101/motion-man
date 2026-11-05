"""The style rules theme config can't express: top bar, page header, cards, type scale, action colours.

Tokens: near-black surfaces (void / carbon / obsidian / slate), grey type (paper / mist / fog / ash), a bone fill
for the single primary action per view, acid lime only as a thin signal (focus ring, active tab, progress),
hairline borders instead of shadows, 6/12 px radii, a 4 px grid. Rationale: ``docs/design/ui.md``.
Selectors are limited to Streamlit's data-testid attributes and the documented ``st-key-<key>`` classes;
Streamlit is pinned because these names can change between versions.
"""

import base64
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
  --mm-bone: #e8e9e3;          /* primary action fill; void text on it is 17.3:1 */
  --mm-bone-hover: #f4f5f0;
  --mm-bone-pressed: #d8d9d3;
  --mm-lime: #e4f222;          /* signal only: focus ring, active tab underline, progress bar */
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

/* Top bar: sticky, carbon, hairline bottom border, bleeds to the gutters. Brand left; Library + New video right,
   sized to their labels so nothing truncates at narrow widths. */
[data-testid="stLayoutWrapper"]:has(> .st-key-topbar) {
  position: sticky; top: 0; z-index: 99;  /* the wrapper, not the block: a sticky needs room inside its parent */
  padding: 10px 0;  /* no negative margin: the wrapper's width is fixed, so it only shifted the bar left */
  background: var(--mm-carbon);
  /* Full bleed: the content stays on the 1280 px grid while the fill and the hairline reach both window edges. */
  box-shadow: 0 0 0 100vmax var(--mm-carbon), 0 1px 0 100vmax var(--mm-graphite);
  clip-path: inset(0 -100vmax -1px);
}
.st-key-topbar, .st-key-topbar [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; }
.st-key-topbar [data-testid="stHorizontalBlock"] > [data-testid="stLayoutWrapper"] { flex: 0 0 auto; width: auto; }
.st-key-topbar button { padding-left: 12px; padding-right: 12px; white-space: nowrap; }
.mm-brand { display: flex; align-items: center; gap: 10px; height: 36px; }
.mm-brand img { width: 24px; height: 24px; flex: none; display: block; }
.mm-brand-name { font-size: 16px; font-weight: 510; color: var(--mm-paper); letter-spacing: -0.011em; }
/* Menus (Library, ⋯ details, Export). Popover bodies render in a portal, outside their st-key container, so
   these rules are generic. The Library is a list: one tertiary row per run, left-aligned, the whole row clickable;
   the legacy filter sits under a rule. */
[data-testid="stPopoverBody"] { min-width: 320px; max-width: 480px; max-height: 70vh; padding: 12px; }
[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] { gap: 2px; }
[data-testid="stPopoverBody"] [data-testid="stBaseButton-tertiary"] {
  justify-content: flex-start; padding: 6px 8px; border-radius: 6px;
}
[data-testid="stPopoverBody"] [data-testid="stBaseButton-tertiary"] p { font-size: 13px; color: var(--mm-mist); }
[data-testid="stPopoverBody"] [data-testid="stBaseButton-tertiary"]:hover { background: var(--mm-slate); }
[data-testid="stPopoverBody"] [data-testid="stBaseButton-tertiary"]:hover p { color: var(--mm-paper); }
[data-testid="stPopoverBody"] [data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p {
  font-size: 13px; color: var(--mm-mist);
}
[data-testid="stPopoverBody"] hr { margin: 8px 0; }
[data-testid="stPopoverBody"] [data-testid="stCaptionContainer"] p { font-size: 12px; }
[data-testid="stPopoverBody"] [data-testid="stBaseButton-secondary"] { margin-top: 4px; }

/* Page header: title and status pill on one centre line, the item's actions right; facts underneath. */
.st-key-pagehead { padding: 24px 0 0; }
.st-key-pagehead h2, .st-key-pagehead h3 { padding: 0; margin: 0; }
/* The title row never wraps; the status fragment's wrapper hugs its pill instead of stretching (it is a vertical
   block inside a horizontal one, which Streamlit sizes to stretch by default). */
.st-key-pagehead [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; }
.st-key-pagehead [data-testid="stHorizontalBlock"] > [data-testid="stLayoutWrapper"] { flex: 0 0 auto; width: auto; }
.st-key-pagehead [data-testid="stCaptionContainer"] { padding-top: 4px; }
.st-key-pagehead [data-testid="stCaptionContainer"] p { font-size: 13px; color: var(--mm-fog); }
.st-key-pagehead span[class*="-badge"] {
  display: inline-flex; align-items: center; height: 24px; padding: 0 10px; margin-left: 2px;
  border-radius: 9999px; font-size: 12px; white-space: nowrap;
}
.st-key-pagehead [data-testid="stMarkdownContainer"]:has(> p > span[class*="-badge"]) p { margin: 0; }
/* The overflow (⋯): a 36 px square ghost button, no label. */
.st-key-details [data-testid="stPopoverButton"] { width: 36px; min-width: 36px; padding: 0; justify-content: center; }
.st-key-details [data-testid="stPopoverButton"] [data-testid="stIconMaterial"] { margin: 0; }
.st-key-details [data-testid="stPopoverButton"] div[aria-hidden="true"] { display: none; }  /* no chevron on ⋯ */
.st-key-pagehead h3 { line-height: 1.15; }
/* Streamlit gives a heading's markdown container a -14 px bottom margin; in a centred flex row that pushes the
   title 7 px below the buttons. The heading box must be the heading's real height. */
.st-key-pagehead [data-testid="stMarkdownContainer"]:has(h3) { margin-bottom: 0 !important; }
/* Tabs follow the facts line closely so title → facts → sections read as one block. */
.st-key-pagehead + div [data-testid="stTabs"], [data-testid="stTabs"] { margin-top: 12px; }

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

/* The one primary action per view (Run tracking, Export): bone fill, void text (17.3:1), never bold.
   Popover buttons carry kind="primary" instead of the primary test id, so both forms are covered. */
[data-testid="stBaseButton-primary"], [data-testid="stPopoverButton"][kind="primary"] {
  background: var(--mm-bone); border-color: var(--mm-bone);
}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primary"] p,
[data-testid="stPopoverButton"][kind="primary"], [data-testid="stPopoverButton"][kind="primary"] p,
[data-testid="stPopoverButton"][kind="primary"] [data-testid="stIconMaterial"] {
  color: var(--mm-void) !important; font-weight: 510;
}
/* Multiselect chips take the bone primary as their fill; Streamlit's default white text on it was about 1.2:1. */
[data-testid="stMultiSelectTagsContainer"] [role="group"] > span,
[data-testid="stMultiSelectTagsContainer"] [role="group"] > span * {
  color: var(--mm-void) !important; fill: var(--mm-void);
}
[data-testid="stBaseButton-primary"]:not(:disabled):hover,
[data-testid="stPopoverButton"][kind="primary"]:not(:disabled):hover {
  background: var(--mm-bone-hover); border-color: var(--mm-bone-hover);
}
[data-testid="stBaseButton-primary"]:not(:disabled):active,
[data-testid="stPopoverButton"][kind="primary"]:not(:disabled):active {
  background: var(--mm-bone-pressed); border-color: var(--mm-bone-pressed);
}
[data-testid="stBaseButton-primary"]:disabled { background: var(--mm-obsidian); border-color: var(--mm-graphite); }
[data-testid="stBaseButton-primary"]:disabled p { color: var(--mm-ash) !important; }

/* Ghost buttons (secondary buttons and secondary popovers): transparent, hairline, mist text; slate on hover. */
[data-testid="stBaseButton-secondary"], [data-testid="stPopoverButton"]:not([kind="primary"]) {
  background: transparent; border: 1px solid var(--mm-graphite); color: var(--mm-mist);
}
[data-testid="stBaseButton-secondary"]:not(:disabled):hover,
[data-testid="stPopoverButton"]:not([kind="primary"]):not(:disabled):hover {
  background: var(--mm-slate); border-color: var(--mm-smoke); color: var(--mm-paper);
}
[data-testid="stBaseButton-secondary"] p, [data-testid="stPopoverButton"] p { font-size: 13px; }
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


def brand_html() -> str:
    """The top-bar brand: glyph and wordmark. Item names belong to the page title, never to the bar."""
    return (
        f'<div class="mm-brand"><img src="{LOGO_URI}" alt="" width="24" height="24">'
        f'<span class="mm-brand-name">MotionMan</span></div>'
    )

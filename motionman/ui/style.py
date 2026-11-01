"""The few style rules theme config can't express (plan ui.md D3; tokens from the UI reference).

Selectors are limited to Streamlit's data-testid attributes and the documented ``st-key-<key>`` classes;
Streamlit is pinned because these names can change between versions.
"""

CSS = """
<style>
:root {
  --mm-panel: #07080A;
  --mm-raised: #1B1C1E;
  --mm-border: #363739;
  --mm-text-secondary: #A1A1AA;
  --mm-text-muted: #8B8C91;
  --mm-accent: #A3E635;
  --mm-accent-hover: #B5EF59;
  --mm-accent-pressed: #8CC927;
  --mm-on-accent: #0B1105;
  --mm-mono: "Geist Mono", ui-monospace, Menlo, Consolas, monospace;
  --mm-panel-highlight: inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

/* The one green action: dark text on lime, visible hover and pressed states. */
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primary"] p { color: var(--mm-on-accent) !important; font-weight: 600; }
[data-testid="stBaseButton-primary"]:not(:disabled):hover {
  background: var(--mm-accent-hover);
  border-color: var(--mm-accent-hover);
}
[data-testid="stBaseButton-primary"]:not(:disabled):active {
  background: var(--mm-accent-pressed);
  border-color: var(--mm-accent-pressed);
}
[data-testid="stBaseButton-primary"]:disabled { background: var(--mm-raised); border-color: var(--mm-border); }
[data-testid="stBaseButton-primary"]:disabled p { color: var(--mm-text-muted) !important; }

/* Keyboard focus: 2 px lime outline with 2 px offset, never removed for appearance. */
.stApp :is(button, input, select, textarea, a, [role="radio"], [role="slider"], [role="tab"]):focus-visible {
  outline: 2px solid var(--mm-accent) !important;
  outline-offset: 2px;
}

/* Secondary text in readable grey rather than full white (or the reference's dim #6A6B6C). */
[data-testid="stCaptionContainer"], [data-testid="stMetricLabel"] { color: var(--mm-text-secondary); }

/* Panels: 16 px radius, neutral hairline, faint inset top highlight. */
[data-testid="stMetric"], .st-key-video-panel {
  background: var(--mm-panel);
  border-radius: 16px;
  box-shadow: var(--mm-panel-highlight);
}

/* Technical values in mono with tabular figures. */
[data-testid="stMetricValue"] { font-family: var(--mm-mono); font-variant-numeric: tabular-nums; font-weight: 500; }

/* Section headings are labels, not destinations: drop their anchor links so they aren't keyboard tab stops. */
[data-testid="stHeaderActionElements"] { display: none; }
</style>
"""

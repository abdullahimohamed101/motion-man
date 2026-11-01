# MotionMan UI: navigation and header

How the top of the setup page and the result page is organised, and why. Companion to the theme tokens in
`.streamlit/config.toml` and `motionman/ui/style.py`. Written 2026-11-01 after the owner reviewed PR #10.

## What was wrong

Seen in the running app at 1440 px and at 800 px (the desktop pane's default width).

1. **The clip name appeared three times** (after the wordmark, in the run picker, as the page title). Repetition
   reads as noise and makes it unclear which one is "the title".
2. **The run picker spoke in IDs**: `crosswalk.mp4 · 2026-11-01 19:45 · ed3f02`. A hash fragment and an ISO
   timestamp mean nothing to a viewer; they were only there to keep selectbox labels unique.
3. **"Legacy" toggle with a "?" sat in the global bar.** It is a filter for a developer edge case, not a way to
   move around the product, yet it had the same weight as the main controls.
4. **The status pill floated alone** between the toggle and the buttons; the gaps between groups were uneven
   because five columns with fixed ratios can't align content of different widths.
5. **Export was a large acid-lime block** with pale text. Its button renders as `stPopoverButton`, so the
   dark-text rule written for primary buttons never applied: low contrast and the loudest thing on the page.
6. **A raw run ID in a mono chip led the caption** (`r_20261101T194556Z_ed3f02`) before the facts people want
   (how long, how many frames, what size).
7. **Three loosely related rows** (bar, title block, tabs) with no visual hierarchy between them.
8. **At 800 px every bar control truncated** ("Com…", "Ne…", "E…"): the fixed column ratios had no room.
9. **Setup page**: the bar showed an empty picker, the Legacy toggle and a "Ready" pill with nothing to the
   right; the heading was a sentence ("Choose a video to track objects, or try the demo."), not a page name.

## Patterns borrowed

- **Library vs. item** (Google Photos, Loom, YouTube Studio): the global bar holds *where you are* and *make
  new*; the item page holds the item's title and the item's actions. Nothing about one item lives in the bar.
- **One title, shown once, with status next to it** (Figma file view, Frame.io asset header, Loom): the name
  is the h1 of the page; a small pill right after it carries state ("Complete", "Processing").
- **Human dates** (Google Photos, Apple Photos): "Today, 7:45 PM", "Yesterday, 7:45 PM", "Mon, 7:45 PM",
  "1 Nov, 7:45 PM", "1 Nov 2025". Never an ISO string, never a hash.
- **One primary action, the rest in an overflow** (Material top-app-bar guidance; Loom's Share + "⋯";
  Frame.io's Download/Share + more menu): on an item, the single filled button is the one most people click.
  Everything infrequent sits behind "⋯".
- **Filters live in the library, not the global bar** (YouTube Studio's filter row, Google Photos' Utilities):
  "Show legacy runs" is a library filter.
- **Technical identity in a details sheet** (Loom "Video details", Finder "Get Info"): the run ID, hashes and
  folder are one click away, never in the headline.
- **Content-sized controls, right-aligned group** (Apple HIG toolbars): buttons size to their label and sit
  in one right-hand cluster with a fixed gap, so nothing truncates at narrow widths.

Research note: live web sources confirmed the Material action-ordering/overflow rule and Google Photos' relative
date headings; the Loom, Frame.io, Figma, Descript, CapCut and YouTube Studio patterns are from prior knowledge
of those products (their pages are JS-rendered and couldn't be fetched).

## Information architecture

### Top bar (both pages, sticky, full bleed)
```
[ logo  MotionMan ]                                      [ Library ▾ ]  [ + New video ]
```
- **Brand** left. No source name after it.
- **Library ▾** (popover): saved runs, newest first, one row each: `crosswalk.mp4` · `Today, 7:45 PM` ·
  `1,356 frames`. The open run has a check mark. Legacy runs say `Legacy (unverified)` instead of frames.
  At the bottom, under a rule: the **Show legacy runs** toggle (the old "Legacy" toggle, same key, same help).
- **New video** only on the result page; the setup page *is* "new video".
- Nothing else. No status, no filters, no item data.

### Setup page header
```
Track a video  [Ready]
Choose a video or try the demo. MP4 with a constant frame rate · up to 512 MiB, 10 minutes and 1920×1080
```
The status pill shows Ready / Processing / Error exactly as before (same live refresh).

### Result page header
```
crosswalk.mp4  [Complete]                                          [ ⤴ Export ]  [ ⋯ ]
Today, 7:45 PM · 56 s · 1,356 frames · 1920×1080
Inspect | Performance
```
- **Title** once, h3. **Status** pill right after it, same baseline.
- **Meta line**: human start time, clip duration (frames ÷ fps), frame count, size. Legacy runs add
  "source unverified (legacy run)". No ID.
- **Export** is the one primary action (popover with the same downloads and evidence PNG as before).
- **⋯ Details** popover: Run ID, started (exact UTC), playback fps, source verification, run folder. The
  Performance tab's "Run details" expander keeps everything it had.
- **Tabs** directly under the meta line with 12 px between them, so title → facts → sections read as one block.

## Colour rules for actions

- **Primary action** (Run tracking, Export): **bone** `#E8E9E3` fill, **void** `#08090A` text. Contrast 17.3:1
  (AA needs 4.5:1). Hover `#F4F5F0`, pressed `#D8D9D3`. One per view.
- **Secondary** (New video, Library, First seen …): transparent, graphite hairline, **mist** `#D0D6E0` text on
  void: 13.4:1. Hover: slate fill, paper text.
- **Overflow (⋯)**: same as secondary, icon only, 36 px square.
- **Acid lime** `#E4F222` stays only where it is a *signal*, never a surface: the keyboard focus ring, the active
  tab underline (2 px) and the progress bar. These are thin strokes, so lime's brightness reads as "look here"
  without shouting. It no longer fills any button, badge or selection. `primaryColor` in the theme is bone so
  sliders, checkboxes and selected options follow the calm primary.
- **Events & zones tab**: **Analyze** is the tab's primary only while the drawn regions have no up-to-date
  analysis for this run; once they do it turns into a disabled secondary "Up to date", so the header's Export
  is again the only filled button. Region outlines use mist; the shape being edited uses lime as a thin signal.
- **Status pills** keep Streamlit's semantic colours (green Complete, red Error, grey otherwise) with the label
  always written out; colour never carries the meaning alone.
- Disabled primary: obsidian fill, ash text, as before.

## Spacing and alignment

- Bar: 56 px tall (12 px padding, 32–36 px controls), items vertically centred; the right-hand group is one
  flex cluster with an 8 px gap; controls size to content so nothing truncates down to 760 px.
- Page header: 24 px above the title, 4 px title → meta, 12 px meta → tabs, 24 px tabs → content.
- Title row: title, pill and action buttons share one centre line; the pill has 10 px left margin.
- Headings use the existing type scale (h3 = 20 px / 590); the pill is 12 px, 24 px tall.

## Unchanged

- All functionality and data: runs, records, exports, presets, the Inspect and Performance tabs, legacy runs,
  the live status refresh, the "a finished run opens its result" behaviour, the one-runner-per-process rule.
- Data formats, run IDs, hashes and the pipeline: nothing outside `motionman/ui/` and the theme file changes.
- The full-bleed sticky bar technique, the card grid, the dark surfaces and the type scale.

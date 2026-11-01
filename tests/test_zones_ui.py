"""Events & zones tab, part 1 (plan P3 step 5): drawing and saving regions never analyzes or infers; Analyze writes
one analysis and an identical one is reused. Same world as test_app: real runs, a fake runner, model calls raise."""

from motionman import analysis, regions
from motionman.ui import zones as zones_ui

from . import test_app
from .test_app import open_run, render

world = test_app.world  # the same pytest fixture: real runs, a fake runner, model calls that raise


INPUT_IDS: dict[str, str] = {}


def zones(at, world):
    INPUT_IDS["a"] = analysis.find_run(world["root"], world["a"]).manifest["input_id"]
    return open_run(render(at), world["a"])


def add_region(at, name, kind="Line", coords=None, **extra):
    at.text_input(key="zname").set_value(name)
    at.segmented_control(key="zkind").set_value(kind)
    at = render(at)
    for i, v in enumerate(coords or ()):
        at.number_input(key=f"zc{i}").set_value(v)
    for key, value in extra.items():
        at.number_input(key=key).set_value(value)
    at = render(at)
    at.button(key="zadd").click()
    return render(at)


def draft(at):
    """The editor's draft for run A's video (drafts are kept per video, not per run)."""
    return at.session_state[zones_ui.draft_key(INPUT_IDS["a"])]


def analysis_folders(world):
    base = world["root"].path / "analyses" / world["a"]
    return sorted(p.name for p in base.iterdir()) if base.is_dir() else []


def test_the_tab_starts_empty_and_explains_what_to_draw(world):
    at = zones(None, world)
    assert any("No regions yet" in c.value for c in at.caption)
    assert not any(b.key == "zanalyze" for b in at.button)  # nothing to analyze yet
    assert any(c.value.startswith("Lines point from A to B") for c in at.caption)


def test_adding_regions_updates_the_draft_and_never_analyzes_or_infers(world):
    at = zones(None, world)
    at = add_region(at, "Door", coords=(18.75, 0, 18.75, 100))
    at = add_region(at, "Pen", kind="Rectangle", coords=(21.875, 0, 46.875, 62.5), zdwell=0.5)
    regs = draft(at)["regions"]
    assert [(r["id"], r["kind"]) for r in regs] == [("door", "line"), ("pen", "rect")]
    assert regs[1]["dwell_threshold_s"] == 0.5 and regs[1]["occupancy_threshold"] is None
    assert at.text_input(key="zname").value == ""  # cleared for the next region
    assert analysis_folders(world) == [] and world["fake"].starts == []
    assert any(b.key == "zanalyze" and b.label == "Analyze" for b in at.button)


def test_an_invalid_region_is_refused_with_its_reason(world):
    at = zones(None, world)
    at = add_region(at, "Flat", coords=(50, 50, 50, 50))
    assert draft(at)["regions"] == [] and any("zero length" in e.value for e in at.error)


def test_duplicate_names_get_distinct_ids_and_remove_works(world):
    at = zones(None, world)
    at = add_region(at, "Door")
    at = add_region(at, "Door", coords=(10, 60, 90, 60))
    assert [r["id"] for r in draft(at)["regions"]] == ["door", "door-2"]
    at.button(key="zrm-0").click()
    at = render(at)
    assert [r["id"] for r in draft(at)["regions"]] == ["door-2"]


def test_saving_regions_stores_a_set_but_does_not_analyze(world):
    at = zones(None, world)
    at = add_region(at, "Door", coords=(18.75, 0, 18.75, 100))
    at.button(key="zsave").click()
    at = render(at)
    run = analysis.find_run(world["root"], world["a"])
    saved = regions.list_sets(world["root"], run.manifest["input_id"])
    assert [s.name for s in saved] == ["Regions"] and saved[0].regions[0].id == "door"
    assert analysis_folders(world) == [] and world["fake"].starts == []


def test_analyze_writes_one_analysis_and_an_identical_one_is_reused(world):
    at = zones(None, world)
    at = add_region(at, "Door", coords=(18.75, 0, 18.75, 100))
    at.button(key="zanalyze").click()
    at = render(at)
    (aid,) = analysis_folders(world)
    button = next(b for b in at.button if b.key == "zanalyze")
    assert button.label == "Up to date" and button.disabled
    assert any(f"analysis `{aid}`" in c.value and "1 events" in c.value for c in at.caption)
    # A fresh session that loads the saved set sees the same analysis as up to date: nothing is rebuilt.
    again = zones(None, world)
    set_key = next(s.key for s in again.selectbox if s.key and s.key.startswith("zset-"))
    saved_id = next(o for o in again.selectbox(key=set_key).options if o != "New set")
    again.selectbox(key=set_key).select(saved_id)  # options are shown by name; select by display label
    again = render(again)
    assert next(b for b in again.button if b.key == "zanalyze").label == "Up to date"
    assert analysis_folders(world) == [aid] and world["fake"].starts == []


def test_loading_a_saved_set_restores_its_name_so_its_analysis_is_up_to_date(world):
    at = zones(None, world)
    at = add_region(at, "Door", coords=(18.75, 0, 18.75, 100))
    set_key = next(s.key for s in at.selectbox if s.key and s.key.startswith("zset-"))
    name_key = set_key.replace("zset-", "zsetname-")
    at.text_input(key=name_key).set_value("Doorway")
    at = render(at)
    at.button(key="zanalyze").click()
    at = render(at)
    fresh = zones(None, world)
    fresh.selectbox(key=set_key).select(next(o for o in fresh.selectbox(key=set_key).options if o != "New set"))
    fresh = render(fresh)
    assert fresh.text_input(key=name_key).value == "Doorway"
    assert next(b for b in fresh.button if b.key == "zanalyze").label == "Up to date"


# --- step 6: results ---


def analyzed(world):
    """Run A with a line the object crosses and a zone it enters (dwell alert at 0.1 s), analyzed once."""
    at = zones(None, world)
    at = add_region(at, "Door", coords=(18.75, 0, 18.75, 100))
    at = add_region(at, "Pen", kind="Rectangle", coords=(21.875, 0, 46.875, 62.5), zdwell=0.1)
    at.button(key="zanalyze").click()
    return render(at)


def metrics(at):
    return {m.label: m.value for m in at.metric}


def aid_of(world):
    (aid,) = analysis_folders(world)
    return aid


def test_results_show_region_totals_and_the_occupancy_note(world):
    at = analyzed(world)
    m = metrics(at)
    assert m["Door"] == "1 crossings" and m["Pen"] == "1 entered"
    assert any(c.value.startswith("Occupancy counts who is inside at one moment") for c in at.caption)
    assert any(c.value.startswith("For this run only") for c in at.caption)
    assert any(c.value == "Showing 3 of 3 events. Filters change this view only; the saved totals above stay the same."
               for c in at.caption)  # fmt: skip


def test_events_filter_changes_the_view_never_the_saved_totals(world):
    at = analyzed(world)
    aid = aid_of(world)
    folder = world["root"].path / "analyses" / world["a"] / aid
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    totals = metrics(at)
    at.multiselect(key=f"zf-type-{aid}").set_value(["DWELL_THRESHOLD"])
    at = render(at)
    assert any(c.value.startswith("Showing 1 of 3 events") for c in at.caption)
    at.number_input(key=f"zf-track-{aid}").set_value(99)
    at = render(at)
    assert any(c.value.startswith("Showing 0 of 3 events") for c in at.caption)
    assert metrics(at) == totals and {p.name: p.read_bytes() for p in folder.iterdir()} == before
    assert world["fake"].starts == []


def test_a_bookmark_selects_its_frame_and_show_in_inspect_moves_the_slider(world):
    at = analyzed(world)
    aid = aid_of(world)
    marks = [b for b in at.button if b.key and b.key.startswith(f"zbm-{aid}-")]
    assert [b.label.split(" · ")[0] for b in marks] == ["Peak occupancy", "First crossing", "First entry"]  # by frame
    next(b for b in marks if b.label.startswith("First crossing · Door")).click()
    at = render(at)
    selected = next(c.value for c in at.caption if c.value.startswith("Bookmark: first crossing"))
    frame = int(selected.split("frame ")[1].split(" ")[0])
    at.button(key=f"zgo-{world['a']}").click()
    at = render(at)
    assert at.slider(key=f"frame-{world['a']}").value == frame
    assert at.selectbox(key=f"track-{world['a']}").value == 1  # the crossing's track is now the selected object
    assert any(f"Inspect is now on frame {frame}" in c.value for c in at.caption)


def test_events_exports_join_the_export_menu_once_analyzed(world):
    before = zones(None, world)
    labels = lambda at: [b.label for b in at.get("download_button")]  # noqa: E731
    assert not any("Events" in label for label in labels(before))
    after = analyzed(world)
    assert "Events · one row per event (CSV)" in labels(after)
    assert "Events & zones analysis (JSON)" in labels(after)


# --- step 7: heatmap and motion arrows ---


def test_heatmap_opacity_re_renders_without_touching_saved_files(world):
    at = analyzed(world)
    aid = aid_of(world)
    folder = world["root"].path / "analyses" / world["a"] / aid
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    assert any(c.value.startswith("Image-space observed occupancy:") for c in at.caption)
    at.slider(key=f"zheat-{aid}").set_value(0.2)
    at = render(at)
    at.slider(key=f"zheat-{aid}").set_value(1.0)
    at = render(at)
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before and world["fake"].starts == []


def test_motion_arrows_toggle_draws_from_saved_records_only(world):
    at = zones(None, world)
    at.slider(key=f"frame-{world['a']}").set_value(12)
    at = render(at)
    at.toggle(key=f"arrows-{world['a']}").set_value(True)
    at = render(at)
    assert at.toggle(key=f"arrows-{world['a']}").value and world["fake"].starts == []


# --- review fixes ---


def load_saved(at):
    set_key = next(s.key for s in at.selectbox if s.key and s.key.startswith("zset-"))
    at.selectbox(key=set_key).select(next(o for o in at.selectbox(key=set_key).options if o != "New set"))
    return render(at)


def test_a_loaded_set_can_be_edited_and_the_edit_sticks(world):
    at = zones(None, world)
    at = add_region(at, "Door", coords=(18.75, 0, 18.75, 100))
    at.button(key="zsave").click()
    at = render(at)
    fresh = load_saved(zones(None, world))
    fresh = add_region(fresh, "Pen", kind="Rectangle", coords=(21.875, 0, 46.875, 62.5))
    assert [r["id"] for r in draft(fresh)["regions"]] == ["door", "pen"]  # was reloaded back to ["door"] before
    assert any(c.value.startswith("Edited from") for c in fresh.caption)
    fresh.button(key="zrm-0").click()
    fresh = render(fresh)
    assert [r["id"] for r in draft(fresh)["regions"]] == ["pen"]


def test_export_drops_the_events_once_the_regions_no_longer_match(world):
    at = analyzed(world)
    labels = lambda a: [b.label for b in a.get("download_button")]  # noqa: E731
    assert "Events · one row per event (CSV)" in labels(at)
    at.button(key="zrm-0").click()
    at = render(at)
    assert next(b for b in at.button if b.key == "zanalyze").label == "Analyze"
    assert not any(label.startswith("Events") for label in labels(at))

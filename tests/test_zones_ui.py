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

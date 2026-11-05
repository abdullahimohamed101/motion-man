"""Source time contract shared with the native engine (spec C§8, plan native-g0-g1 step 2). The same fixture is
checked by the C++ test `native/tests/unit/test_source_time.cpp`, so both languages round identically."""

import json
from pathlib import Path

from motionman.records import source_t_us

FIXTURE = Path(__file__).parent / "fixtures" / "source_time.json"


def test_python_matches_every_shared_case():
    cases = json.loads(FIXTURE.read_text())["cases"]
    assert len(cases) > 100
    assert [source_t_us(c["index"], c["fps"]) for c in cases] == [c["t_us"] for c in cases]


def test_exact_halves_round_to_even():
    halves = {(c["index"], c["fps"]): c["t_us"] for c in json.loads(FIXTURE.read_text())["cases"] if c["fps"] > 1e5}
    assert halves == {(1, 2e6): 0, (3, 2e6): 2, (1, 4e5): 2, (3, 4e5): 8, (5, 4e5): 12}

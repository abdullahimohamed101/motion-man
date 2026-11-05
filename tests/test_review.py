"""Review queue candidates (plan P4 step 4, D5): each heuristic, run differences, ranking, the cap and totals."""

from motionman import review


def o(n, tid, box, conf=0.9, cls="person"):
    return {"observation_id": f"x:{n}", "track_id": tid, "class_name": cls, "confidence": conf, "bbox_xyxy": box}


def rec(frame, *objs):
    return {"source_frame_index": frame, "objects": list(objs)}


B = [10, 10, 30, 40]


def kinds(shown):
    return [(c.kind, c.frame) for c in shown]


def test_a_steady_confident_track_raises_nothing():
    shown, totals = review.candidates([rec(f, o(0, 1, [10 + f, 10, 30 + f, 40])) for f in range(10)])
    assert shown == [] and set(totals.values()) == {0}


def test_low_confidence_boxes_are_candidates_with_or_without_an_id():
    shown, totals = review.candidates([rec(0, o(0, 1, B, conf=0.39), o(1, None, B, conf=0.2), o(2, 2, B, conf=0.4))])
    assert kinds(shown) == [("low_confidence", 0), ("low_confidence", 0)] and totals["low_confidence"] == 2
    assert shown[0].detail == "person at 0.39"


def test_a_track_coming_back_after_a_gap_is_a_candidate_at_its_return():
    shown, _ = review.candidates([rec(0, o(0, 7, B)), rec(1), rec(2), rec(3, o(0, 7, B))])
    assert kinds(shown) == [("track_gap", 3)] and shown[0].detail == "#7 person back after 2 frames"


def test_a_box_that_jumps_between_consecutive_frames_is_a_candidate():
    shown, _ = review.candidates([rec(0, o(0, 4, B)), rec(1, o(0, 4, [40, 10, 60, 40]))])
    assert kinds(shown) == [("abrupt_box_change", 1)] and "overlap 0.00" in shown[0].detail


def test_run_differences_lead_each_round_and_name_both_sides():
    car, truck, person = ({"observation_id": i, "class_name": c} for i, c in (("9:0", "car"), ("9:1", "truck"),
                                                                              ("2:3", "person")))  # fmt: skip
    items = [{"frame": 9, "kind": "class_disagreement", "a": car, "b": truck},
             {"frame": 2, "kind": "unmatched_b", "a": None, "b": person}]  # fmt: skip
    shown, totals = review.candidates([rec(0, o(0, 1, B, conf=0.1))], items)
    assert kinds(shown) == [("disagreement", 2), ("low_confidence", 0), ("disagreement", 9)]  # kinds take turns
    assert shown[0].detail == "person only in run B" and shown[2].detail == "car vs truck"
    assert totals["disagreement"] == 2


def test_the_cap_limits_what_is_shown_but_totals_count_everything():
    many = [rec(f, o(0, None, B, conf=0.1)) for f in range(300)]
    shown, totals = review.candidates(many, cap=200)
    assert len(shown) == 200 and totals["low_confidence"] == 300 and shown[0].frame == 0
    assert review.frames_of(shown)[:3] == [0, 1, 2]


def test_a_flood_of_one_kind_cannot_crowd_out_the_others():
    flood = [{"frame": f, "kind": "unmatched_a", "a": {"observation_id": f"{f}:0", "class_name": "car"}, "b": None}
             for f in range(500)]  # fmt: skip
    shown, totals = review.candidates([rec(0, o(0, 1, B, conf=0.1)), rec(1), rec(2, o(0, 1, B))], flood, cap=10)
    assert {c.kind for c in shown} == {"disagreement", "low_confidence", "track_gap"} and totals["disagreement"] == 500

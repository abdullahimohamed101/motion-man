import numpy as np

from motionman.render import NO_ID_COLOR, PALETTE, Trails, annotate, color_for
from motionman.types import Track


def track(tid, x=0.0, y=0.0, w=10.0, h=10.0):
    return Track(tid, 0, "person", 0.9, (x, y, x + w, y + h))


def test_trail_records_box_centre():
    trails = Trails()
    trails.update(0, [track(1, x=10, y=20, w=10, h=20)])
    assert trails.points(1) == [(15, 30)]


def test_trail_is_bounded_to_max_points():
    trails = Trails(max_points=30)
    for i in range(100):
        trails.update(i, [track(1, x=i)])
    pts = trails.points(1)
    assert len(pts) == 30
    assert pts[0] == (75, 5) and pts[-1] == (104, 5)  # the newest 30 centres, oldest first


def test_null_id_gets_no_trail():
    trails = Trails()
    trails.update(0, [track(None), track(None)])
    assert len(trails) == 0


def test_trail_expires_after_max_age_unseen_frames():
    trails = Trails(max_age=60)
    trails.update(0, [track(1)])
    for i in range(1, 60):
        trails.update(i, [])
    assert trails.points(1)  # 59 frames unseen: kept
    trails.update(60, [])
    assert trails.points(1) == []  # 60 frames unseen: dropped
    assert len(trails) == 0


def test_reobserved_id_keeps_its_trail():
    trails = Trails(max_age=60)
    trails.update(0, [track(1, x=0)])
    trails.update(59, [track(1, x=10)])
    trails.update(100, [])  # 41 frames after the last sighting
    assert trails.points(1) == [(5, 5), (15, 5)]


def test_many_ids_stay_bounded():
    trails = Trails(max_points=30, max_age=60)
    for i in range(1000):
        trails.update(i, [track(i)])  # every ID seen once: a fragmenting tracker
    assert len(trails) == 60


def test_color_for_is_deterministic():
    assert color_for(1) == color_for(1 + len(PALETTE))
    assert color_for(None) == NO_ID_COLOR


def test_annotate_draws_on_a_copy_and_keeps_shape():
    image = np.zeros((48, 64, 3), np.uint8)
    trails = Trails()
    for i in range(3):
        trails.update(i, [track(1, x=5 + 5 * i, y=5)])
    out = annotate(image, [track(1, x=15, y=5), track(None, x=40, y=30)], trails)
    assert out.shape == image.shape
    assert out.any()
    assert not image.any()  # the source frame is never modified


def test_annotate_handles_empty_frame():
    image = np.zeros((48, 64, 3), np.uint8)
    out = annotate(image, [], Trails())
    assert not out.any()

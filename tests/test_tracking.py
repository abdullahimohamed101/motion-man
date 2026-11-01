from types import SimpleNamespace

import numpy as np

from motionman.tracking import tracks_from_boxes

NAMES = {0: "person", 2: "car"}


class Boxes(SimpleNamespace):
    """Stand-in for Ultralytics Boxes: numpy arrays share .tolist() with torch tensors."""

    def __len__(self):
        return len(self.xyxy)


def make(xyxy, cls, conf, ids=None):
    return Boxes(
        xyxy=np.array(xyxy, float),
        cls=np.array(cls, float),
        conf=np.array(conf, float),
        id=None if ids is None else np.array(ids, float),
    )


def test_converts_tracked_boxes():
    tracks = tracks_from_boxes(make([[1, 2, 3, 4], [5, 6, 7, 8]], [0, 2], [0.9, 0.5], ids=[7, 8]), NAMES)
    assert [t.track_id for t in tracks] == [7, 8]
    assert [t.class_name for t in tracks] == ["person", "car"]
    assert tracks[0].bbox_xyxy == (1.0, 2.0, 3.0, 4.0)
    assert isinstance(tracks[0].track_id, int) and isinstance(tracks[0].class_id, int)


def test_missing_ids_stay_none():
    tracks = tracks_from_boxes(make([[1, 2, 3, 4]], [0], [0.9], ids=None), NAMES)
    assert tracks[0].track_id is None  # never invented (INV-2)


def test_empty_and_absent_boxes():
    assert tracks_from_boxes(make(np.zeros((0, 4)), [], [], ids=None), NAMES) == []
    assert tracks_from_boxes(None, NAMES) == []


def test_unknown_class_falls_back_to_number():
    assert tracks_from_boxes(make([[0, 0, 1, 1]], [42], [0.3]), NAMES)[0].class_name == "42"

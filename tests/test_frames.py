"""Exact source frames for the inspector (plan P1 step 3, D1)."""

import numpy as np
import pytest

from motionman.inspection import CACHE_FRAMES, FORWARD_STEP_LIMIT, FrameError, FrameSource

from .conftest import write_clip


class SeekableCapture:
    """Fake cv2.VideoCapture: frame i is an image filled with i % 256, timestamped i / fps unless told otherwise.
    ``seek_lands_on`` lets a test make a seek land on the wrong frame, as an unreliable decoder might."""

    def __init__(self, n=600, fps=30.0, seek_lands_on=None, fail_at=None):
        self.n, self.fps, self.pos, self.seek_lands_on, self.fail_at = n, fps, 0, seek_lands_on, fail_at
        self.current = -1

    def isOpened(self):  # noqa: N802 - mirrors the OpenCV API
        return True

    def set(self, prop, value):
        self.pos = self.seek_lands_on(int(value)) if self.seek_lands_on else int(value)
        return True

    def read(self):
        if self.pos >= self.n or self.pos == self.fail_at:
            return False, None
        self.current, self.pos = self.pos, self.pos + 1
        return True, np.full((4, 4, 3), self.current % 256, np.uint8)

    def get(self, prop):
        return self.current * 1000.0 / self.fps

    def release(self):
        pass


def source(n=600, **kw):
    caps = []

    def opener(path):
        caps.append(SeekableCapture(n, **kw))
        return caps[-1]

    return FrameSource("x.mp4", 30.0, n, open_capture=opener), caps


def value(img):
    return int(img[0, 0, 0])


def test_returns_exactly_the_requested_frame_in_any_order():
    fs, _ = source()
    for i in [500, 3, 599, 0, 250, 251, 100]:
        assert value(fs.frame(i)) == i % 256


def test_short_forward_moves_decode_forward_without_seeking():
    fs, _ = source()
    fs.frame(200)
    seeks = fs.seeks
    for i in range(201, 201 + 10):
        fs.frame(i)
    fs.frame(211 + FORWARD_STEP_LIMIT - 1)
    assert fs.seeks == seeks  # stepping forward reused the open reader


def test_backward_and_long_jumps_seek():
    fs, _ = source()
    fs.frame(100)
    fs.frame(50)  # backwards
    fs.frame(51 + FORWARD_STEP_LIMIT + 2)  # too far forward (and not the cached frame 100)
    assert fs.seeks == 3  # 100 (from the start), 50, and the long jump


def test_cache_is_bounded_and_serves_repeats_without_decoding():
    fs, _ = source()
    for i in range(0, 20):
        fs.frame(i * 60 % 600)
    assert len(fs._cache) == CACHE_FRAMES
    decodes = fs.decodes
    fs.frame(19 * 60 % 600)  # most recent: a cache hit
    assert fs.decodes == decodes


def test_a_mispositioned_seek_is_refused_not_shown():
    fs, _ = source(seek_lands_on=lambda i: i + 1)  # decoder lands one frame late
    with pytest.raises(FrameError, match="Couldn't position frame 300 exactly"):
        fs.frame(300)
    assert fs._cap is None  # closed; the next request starts from a known position


def test_decode_failure_and_out_of_range_are_clear():
    fs, _ = source(fail_at=10)
    with pytest.raises(FrameError, match="couldn't decode frame 10"):
        fs.frame(12)
    with pytest.raises(FrameError, match="outside this run"):
        fs.frame(600)


def test_real_clip_frames_match_sequential_decoding(tmp_path):
    import cv2

    clip = write_clip(tmp_path / "c.mp4", 90)
    cap, sequential = cv2.VideoCapture(str(clip)), []
    while (got := cap.read())[0]:
        sequential.append(got[1])
    cap.release()
    fs = FrameSource(clip, 30.0, len(sequential))
    for i in [45, 0, 89, 46, 47, 10, 60]:
        assert np.array_equal(fs.frame(i), sequential[i]), f"frame {i}"
    fs.close()

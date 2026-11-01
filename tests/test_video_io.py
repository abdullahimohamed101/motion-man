import pytest

from motionman import video_io
from motionman.video_io import VideoError, count_frames, open_reader, open_writer, read_frames

from .conftest import FPS, HEIGHT, WIDTH, write_clip


def test_reader_reports_metadata(clip):
    cap, info = open_reader(clip)
    try:
        assert (info.width, info.height, info.fps) == (WIDTH, HEIGHT, FPS)
    finally:
        cap.release()


def test_frames_have_zero_based_index_and_timestamp(clip):
    cap, info = open_reader(clip)
    try:
        frames = list(read_frames(cap, info.fps))
    finally:
        cap.release()
    assert [f.index for f in frames] == list(range(20))
    assert frames[3].timestamp_s == pytest.approx(3 / 30)
    assert frames[0].image.shape == (HEIGHT, WIDTH, 3)


def test_max_frames_stops_early(clip):
    cap, info = open_reader(clip)
    try:
        assert len(list(read_frames(cap, info.fps, max_frames=5))) == 5
    finally:
        cap.release()


def test_missing_source(tmp_path):
    with pytest.raises(VideoError, match="source not found"):
        open_reader(tmp_path / "nope.mp4")


def test_not_a_video(tmp_path):
    path = tmp_path / "notes.mp4"
    path.write_text("not a video")
    with pytest.raises(VideoError, match="cannot open video"):
        open_reader(path)


class FakeCapture:
    """Stands in for cv2.VideoCapture to exercise metadata checks real encoders won't produce."""

    def __init__(self, width, height, fps):
        self.props = {
            video_io.cv2.CAP_PROP_FRAME_WIDTH: width,
            video_io.cv2.CAP_PROP_FRAME_HEIGHT: height,
            video_io.cv2.CAP_PROP_FPS: fps,
            video_io.cv2.CAP_PROP_FRAME_COUNT: 0,
        }
        self.released = False

    def isOpened(self):  # noqa: N802 - mirrors the OpenCV API
        return True

    def get(self, prop):
        return self.props[prop]

    def release(self):
        self.released = True


@pytest.mark.parametrize(
    ("width", "height", "fps", "message"),
    [
        (64, 48, 0.0, "invalid FPS"),
        (64, 48, float("nan"), "invalid FPS"),
        (63, 48, 30.0, "odd dimensions"),
        (64, 47, 30.0, "odd dimensions"),
        (0, 0, 30.0, "invalid dimensions"),
    ],
)
def test_rejects_bad_metadata_and_releases_capture(monkeypatch, tmp_path, width, height, fps, message):
    fake = FakeCapture(width, height, fps)
    monkeypatch.setattr(video_io.cv2, "VideoCapture", lambda path: fake)
    path = tmp_path / "x.mp4"
    path.write_bytes(b"")
    with pytest.raises(VideoError, match=message):
        open_reader(path)
    assert fake.released


def test_writer_keeps_source_dimensions_and_fps(clip, tmp_path):
    cap, info = open_reader(clip)
    out = tmp_path / "out.mp4"
    writer = open_writer(out, info)
    try:
        for frame in read_frames(cap, info.fps):
            writer.write(frame.image)
    finally:
        cap.release()
        writer.release()
    cap2, info2 = open_reader(out)
    cap2.release()
    assert (info2.width, info2.height, info2.fps) == (info.width, info.height, info.fps)
    assert count_frames(out) == 20


def test_writer_that_cannot_open(clip, tmp_path):
    _, info = open_reader(clip)
    with pytest.raises(VideoError, match="cannot open video writer"):
        open_writer(tmp_path / "missing-dir" / "out.mp4", info)


def test_count_frames_decodes_every_frame(tmp_path):
    assert count_frames(write_clip(tmp_path / "seven.mp4", 7)) == 7

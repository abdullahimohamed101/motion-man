import socket
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

WIDTH, HEIGHT, FPS = 64, 48, 30.0

# INV-8: the suite runs offline and never loads the model stack. Enforced, not just intended.
_HEAVY = ("torch", "ultralytics")


def _no_network(*args, **kwargs):
    raise RuntimeError("tests must not use the network (INV-8)")


socket.socket.connect = _no_network  # type: ignore[method-assign]


def pytest_sessionfinish(session, exitstatus):
    loaded = [name for name in _HEAVY if name in sys.modules]
    if loaded:
        print(f"\nINV-8 violated: tests imported {', '.join(loaded)}", file=sys.stderr)
        session.exitstatus = 1


def write_clip(path: Path, n_frames: int, width: int = WIDTH, height: int = HEIGHT, fps: float = FPS) -> Path:
    """Small constant-frame-rate clip with a moving bright square, written with OpenCV."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    for i in range(n_frames):
        frame = np.zeros((height, width, 3), np.uint8)
        x = (i * 3) % (width - 10)
        frame[10:20, x : x + 10] = 255
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def clip(tmp_path) -> Path:
    return write_clip(tmp_path / "clip.mp4", 20)

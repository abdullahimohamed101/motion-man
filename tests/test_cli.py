from pathlib import Path

import pytest

from motionman import cli, pipeline
from motionman.tracking import DeviceError, resolve_device
from motionman.video_io import VideoError

BASE = ["--source", "in.mp4", "--output-dir", "out"]


def test_defaults_match_spec():
    c = cli.parse_args(BASE)
    assert (c.source, c.output_dir) == (Path("in.mp4"), Path("out"))
    assert (c.device, c.model, c.imgsz, c.max_frames) == ("cpu", "yolo11n.pt", 640, None)
    assert (c.conf, c.iou) == (0.25, 0.7)


@pytest.mark.parametrize(
    "extra",
    [
        [],  # missing required flags
        ["--device", "mps"],
        ["--imgsz", "0"],
        ["--imgsz", "650"],  # not a multiple of 32
        ["--imgsz", "abc"],
        ["--max-frames", "0"],
        ["--max-frames", "-5"],
    ],
)
def test_bad_flags_exit_2(extra, capsys):
    argv = extra if extra == [] else BASE + extra
    with pytest.raises(SystemExit) as exc:
        cli.parse_args(argv)
    assert exc.value.code == 2
    assert "error" in capsys.readouterr().err


def _run_raising(monkeypatch, exc):
    def fake_run(config):
        raise exc

    monkeypatch.setattr(pipeline, "run", fake_run)
    return cli.main(BASE)


def test_expected_error_exits_2_with_message(monkeypatch, capsys):
    assert _run_raising(monkeypatch, VideoError("source not found: in.mp4")) == 2
    err = capsys.readouterr().err
    assert "error: source not found: in.mp4" in err and "Traceback" not in err


def test_interrupt_exits_130(monkeypatch, capsys):
    assert _run_raising(monkeypatch, KeyboardInterrupt()) == 130
    assert "interrupted" in capsys.readouterr().err


def test_unexpected_error_exits_1(monkeypatch, capsys):
    assert _run_raising(monkeypatch, RuntimeError("boom")) == 1
    assert "unexpected RuntimeError: boom" in capsys.readouterr().err


def test_success_prints_summary(monkeypatch, capsys):
    report = {
        "processed_frames": 90,
        "timing": {"processing_wall_s": 4.5, "end_to_end_fps": 20.0},
        "device": {"used": "cpu"},
    }
    monkeypatch.setattr(pipeline, "run", lambda config: report)
    assert cli.main(BASE) == 0
    assert "done: 90 frames in 4.5 s (20.0 fps end to end, device cpu)" in capsys.readouterr().out


def test_cpu_never_checks_cuda():
    def boom():
        raise AssertionError("cpu must not import torch or probe CUDA")

    assert resolve_device("cpu", cuda_available=boom) == "cpu"


def test_auto_falls_back_to_cpu_and_logs(caplog):
    caplog.set_level("INFO")
    assert resolve_device("auto", cuda_available=lambda: False) == "cpu"
    assert "auto → cpu" in caplog.text


def test_auto_prefers_cuda_when_available():
    assert resolve_device("auto", cuda_available=lambda: True) == "cuda"


def test_explicit_cuda_without_gpu_is_a_clear_error():
    with pytest.raises(DeviceError, match="CUDA is not available"):
        resolve_device("cuda", cuda_available=lambda: False)

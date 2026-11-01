"""Opt-in resource telemetry for one run (plan P2 D9, MD§12 "Telemetry and doctor").

A daemon thread samples this process's resident memory about once a second into ``telemetry.jsonl``. It never
touches the tracker (INV-1) and holds only a bounded number of small rows. GPU counters are reported only where
they can really be read (NVML with CUDA); everywhere else they are "Unavailable" with a reason, never guessed.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path
from time import monotonic

import numpy as np

INTERVAL_S = 1.0
MAX_SAMPLES = 7_200  # two hours at 1 Hz; far beyond the 10-minute input limit even on a slow device
SCOPE = "from just before the first decode to the end of processing; process-specific resident memory (RSS)"
UNAVAILABLE = "Unavailable"


def process_rss() -> int:
    import psutil  # imported lazily: only telemetry needs it

    return psutil.Process().memory_info().rss


def gpu_probe(device_used: str | None) -> tuple[Callable[[], dict] | None, dict]:
    """(sampler or None, description). NVML values are device-wide, not this process's share."""
    if not device_used or not device_used.startswith("cuda"):
        return None, {"status": UNAVAILABLE, "reason": f"device {device_used or 'unknown'} has no GPU counters"}
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:  # not installed, no driver, or not an NVIDIA device
        return None, {"status": UNAVAILABLE, "reason": f"NVML not usable here ({type(e).__name__})"}

    def sample() -> dict:
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {"gpu_memory_used_bytes": int(mem.used), "gpu_utilization_pct": int(util.gpu)}

    return sample, {"status": "sampled", "source": "NVML", "scope": "device-wide, not process-specific"}


class Telemetry:
    def __init__(
        self,
        path: Path,
        *,
        device_used: str | None = None,
        interval_s: float = INTERVAL_S,
        max_samples: int = MAX_SAMPLES,
        rss: Callable[[], int] = process_rss,
        gpu: tuple[Callable[[], dict] | None, dict] | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.path, self.interval_s, self.max_samples = path, interval_s, max_samples
        self._rss, self._clock = rss, clock
        self._gpu_sample, self._gpu_info = gpu if gpu is not None else gpu_probe(device_used)
        self._rss_values: list[int] = []
        self._rows = 0
        self.truncated = False
        self.error: str | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._file = None
        self._t0 = 0.0

    def start(self) -> None:
        self._file = self.path.open("w", encoding="utf-8")
        self._t0 = self._clock()
        self.sample()  # t = 0, so even a sub-second run has one row
        self._thread = threading.Thread(target=self._loop, name="motionman-telemetry", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.sample()

    def sample(self) -> None:
        with self._lock:
            if self._file is None or self.error is not None:
                return
            if self._rows >= self.max_samples:
                self.truncated = True
                return
            try:
                row = {"t_s": round(self._clock() - self._t0, 3), "rss_bytes": int(self._rss())}
                if self._gpu_sample is not None:
                    row.update(self._gpu_sample())
            except Exception as e:  # telemetry must never fail the run; the summary says it stopped
                self.error = f"{type(e).__name__}: {e}"
                return
            self._file.write(json.dumps(row) + "\n")
            self._rss_values.append(row["rss_bytes"])
            self._rows += 1

    def stop(self) -> dict:
        """Idempotent. Takes a final sample, closes the file and returns the report summary."""
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._thread = None
            self.sample()
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
        return self.summary()

    def summary(self) -> dict:
        values = self._rss_values
        mb = 1024 * 1024
        return {
            "enabled": True,
            "interval_s": self.interval_s,
            "samples": self._rows,
            "scope": SCOPE,
            "rss_mb": {"peak": round(max(values) / mb, 1), "median": round(float(np.median(values)) / mb, 1)}
            if values
            else None,
            "gpu": self._gpu_info,
            "truncated": self.truncated,
            "error": self.error,
        }

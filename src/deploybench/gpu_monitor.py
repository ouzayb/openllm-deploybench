"""Background GPU sampling during benchmarks."""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from deploybench.metrics import summarize_gpu_samples
from deploybench.result_schema import GPUSample, GPUSampleSummary
from deploybench.utils import utc_now_iso

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

try:
    import pynvml
except ImportError:
    pynvml = None  # type: ignore


class GPUMonitor:
    def __init__(self, sample_interval_seconds: float = 0.5) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self._samples: list[GPUSample] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._nvml_initialized = False
        self._warning: str | None = None

    def start(self) -> None:
        self._samples = []
        self._stop_event.clear()
        if pynvml is None:
            self._warning = "pynvml not available; GPU monitoring disabled"
            logger.warning(self._warning)
            return
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
        except Exception as e:
            self._warning = f"pynvml init failed: {e}"
            logger.warning(self._warning)
            return
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self._take_sample()
            self._stop_event.wait(self.sample_interval_seconds)

    def _take_sample(self) -> None:
        if not self._nvml_initialized or pynvml is None:
            return
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                power: float | None = None
                temp: float | None = None
                sm_clock: float | None = None
                mem_clock: float | None = None
                try:
                    power = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
                except pynvml.NVMLError:
                    pass
                try:
                    temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                except pynvml.NVMLError:
                    pass
                try:
                    sm_clock = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
                except pynvml.NVMLError:
                    pass
                try:
                    mem_clock = float(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM))
                except pynvml.NVMLError:
                    pass
                self._samples.append(
                    GPUSample(
                        timestamp=utc_now_iso(),
                        gpu_index=i,
                        memory_used_mb=mem.used / (1024 * 1024),
                        memory_total_mb=mem.total / (1024 * 1024),
                        utilization_gpu_percent=float(util.gpu),
                        utilization_memory_percent=float(util.memory),
                        power_draw_watts=power,
                        temperature_c=temp,
                        sm_clock_mhz=sm_clock,
                        memory_clock_mhz=mem_clock,
                    )
                )
        except Exception as e:
            logger.debug("GPU sample error: %s", e)

    def stop(self) -> list[GPUSample]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._nvml_initialized and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._nvml_initialized = False
        return list(self._samples)

    def summarize(self, samples: list[GPUSample] | None = None) -> GPUSampleSummary:
        s = samples if samples is not None else self._samples
        return summarize_gpu_samples(s, self.sample_interval_seconds)

    @property
    def warning(self) -> str | None:
        return self._warning

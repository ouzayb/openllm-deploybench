"""Background GPU sampling during benchmarks."""

from __future__ import annotations

import logging
import threading
from typing import Any

from deploybench.metrics import summarize_gpu_samples
from deploybench.nvml_helper import get_nvml, nvml_init, nvml_shutdown
from deploybench.result_schema import GPUSample, GPUSampleSummary
from deploybench.utils import utc_now_iso

logger = logging.getLogger(__name__)


class GPUMonitor:
    def __init__(self, sample_interval_seconds: float = 0.5) -> None:
        self.sample_interval_seconds = sample_interval_seconds
        self._samples: list[GPUSample] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._nvml_initialized = False
        self._nvml: Any = None
        self._warning: str | None = None

    def start(self) -> None:
        self._samples = []
        self._stop_event.clear()
        nvml, err = nvml_init()
        if nvml is None or err:
            self._warning = err or "nvidia-ml-py not available; GPU monitoring disabled"
            logger.warning(self._warning)
            return
        self._nvml = nvml
        self._nvml_initialized = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            self._take_sample()
            self._stop_event.wait(self.sample_interval_seconds)

    def _take_sample(self) -> None:
        if not self._nvml_initialized or self._nvml is None:
            return
        nvml = self._nvml
        try:
            count = nvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = nvml.nvmlDeviceGetHandleByIndex(i)
                mem = nvml.nvmlDeviceGetMemoryInfo(handle)
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                power: float | None = None
                temp: float | None = None
                sm_clock: float | None = None
                mem_clock: float | None = None
                nvml_error = getattr(nvml, "NVMLError", Exception)
                try:
                    power = float(nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
                except nvml_error:
                    pass
                try:
                    temp = float(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU))
                except nvml_error:
                    pass
                try:
                    sm_clock = float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM))
                except nvml_error:
                    pass
                try:
                    mem_clock = float(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM))
                except nvml_error:
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
        if self._nvml_initialized:
            nvml_shutdown()
            self._nvml_initialized = False
        return list(self._samples)

    def summarize(self, samples: list[GPUSample] | None = None) -> GPUSampleSummary:
        s = samples if samples is not None else self._samples
        return summarize_gpu_samples(s, self.sample_interval_seconds)

    @property
    def warning(self) -> str | None:
        return self._warning

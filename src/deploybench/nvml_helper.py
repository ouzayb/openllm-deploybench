"""NVML access with nvidia-ml-py (preferred) and graceful fallback."""

from __future__ import annotations

import logging
import warnings
from typing import Any

logger = logging.getLogger(__name__)

_nvml_module: Any = None
_nvml_init_error: str | None = None


def get_nvml():
    """Return NVML module if available (nvidia-ml-py or legacy pynvml)."""
    global _nvml_module
    if _nvml_module is not None:
        return _nvml_module
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*pynvml.*")
        try:
            import pynvml

            _nvml_module = pynvml
            return pynvml
        except ImportError:
            return None


def nvml_init() -> tuple[Any | None, str | None]:
    """Initialize NVML. Returns (module, error_message)."""
    global _nvml_init_error
    nvml = get_nvml()
    if nvml is None:
        _nvml_init_error = (
            "nvidia-ml-py not installed (pip install nvidia-ml-py). "
            "Do not run: pip uninstall pynvml — that removes NVML bindings."
        )
        return None, _nvml_init_error
    try:
        nvml.nvmlInit()
        _nvml_init_error = None
        return nvml, None
    except Exception as e:
        _nvml_init_error = str(e)
        logger.warning("NVML init failed: %s", e)
        return nvml, _nvml_init_error


def nvml_shutdown() -> None:
    nvml = get_nvml()
    if nvml is None:
        return
    try:
        nvml.nvmlShutdown()
    except Exception:
        pass

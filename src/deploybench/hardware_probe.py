"""Hardware fingerprinting for benchmark machines."""

from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil

from deploybench.config import HardwareConfig, load_hardware_config
from deploybench.result_schema import HardwareProbeResult
from deploybench.utils import get_package_versions, utc_now_iso, write_json

logger = logging.getLogger(__name__)

NVIDIA_COMMANDS = [
    ("nvidia_smi", ["nvidia-smi"]),
    ("nvidia_smi_q", ["nvidia-smi", "-q"]),
    ("nvidia_smi_topo", ["nvidia-smi", "topo", "-m"]),
    ("nvidia_smi_nvlink", ["nvidia-smi", "nvlink", "--status"]),
]


def _run_command(cmd: list[str], timeout: int = 60) -> dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "command": " ".join(cmd),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except FileNotFoundError:
        return {"command": " ".join(cmd), "error": "command not found"}
    except subprocess.TimeoutExpired:
        return {"command": " ".join(cmd), "error": "timeout"}
    except Exception as e:
        return {"command": " ".join(cmd), "error": str(e)}


def _collect_pynvml_gpus() -> tuple[list[dict[str, Any]], str | None, str | None]:
    gpus: list[dict[str, Any]] = []
    driver_version: str | None = None
    cuda_version: str | None = None
    try:
        import pynvml

        pynvml.nvmlInit()
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        try:
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            if isinstance(cuda_version, int):
                major = cuda_version // 1000
                minor = (cuda_version % 1000) // 10
                cuda_version = f"{major}.{minor}"
        except Exception:
            cuda_version = None

        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8", errors="replace")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info: dict[str, Any] = {
                "index": i,
                "name": name,
                "uuid": uuid,
                "memory_total_mb": mem.total / (1024 * 1024),
            }
            try:
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                info["power_limit_watts"] = power_limit / 1000.0
            except Exception:
                pass
            try:
                max_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
                info["max_power_limit_watts"] = max_limit[1] / 1000.0
            except Exception:
                pass
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle)
                info["power_draw_watts"] = power / 1000.0
            except Exception:
                pass
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                info["temperature_c"] = temp
            except Exception:
                pass
            try:
                pci = pynvml.nvmlDeviceGetPciInfo(handle)
                info["pci_bus"] = pci.busId.decode() if isinstance(pci.busId, bytes) else pci.busId
            except Exception:
                pass
            try:
                mig = pynvml.nvmlDeviceGetMigMode(handle)
                info["mig_mode"] = mig
            except Exception:
                pass
            gpus.append(info)
        pynvml.nvmlShutdown()
    except ImportError:
        logger.warning("pynvml not installed")
    except Exception as e:
        logger.warning("pynvml GPU collection failed: %s", e)
    return gpus, driver_version, cuda_version


def _parse_cuda_from_smi(stdout: str) -> str | None:
    m = re.search(r"CUDA Version:\s*([\d.]+)", stdout)
    return m.group(1) if m else None


def _disk_info() -> list[dict[str, Any]]:
    disks: list[dict[str, Any]] = []
    for part in psutil.disk_partitions(all=False):
        try:
            usage = shutil.disk_usage(part.mountpoint)
            disks.append(
                {
                    "device": part.device,
                    "mountpoint": part.mountpoint,
                    "fstype": part.fstype,
                    "total_gb": round(usage.total / (1024**3), 2),
                    "free_gb": round(usage.free / (1024**3), 2),
                }
            )
        except PermissionError:
            continue
    return disks


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def probe_hardware(hardware_config: HardwareConfig | None = None) -> HardwareProbeResult:
    raw_outputs: dict[str, Any] = {}
    for name, cmd in NVIDIA_COMMANDS:
        raw_outputs[name] = _run_command(cmd)

    gpus, driver_version, cuda_version = _collect_pynvml_gpus()
    if cuda_version is None and "nvidia_smi" in raw_outputs:
        smi = raw_outputs["nvidia_smi"]
        if "stdout" in smi:
            cuda_version = _parse_cuda_from_smi(smi["stdout"])

    ram = psutil.virtual_memory().total / (1024**3)
    versions = get_package_versions()

    result = HardwareProbeResult(
        timestamp_utc=utc_now_iso(),
        hostname=platform.node(),
        os=f"{platform.system()} {platform.release()}",
        kernel_version=platform.release(),
        python_version=versions.get("python_version", sys.version.split()[0]),
        cpu_model=_cpu_model(),
        cpu_core_count=psutil.cpu_count(logical=True) or 0,
        ram_total_gb=round(ram, 2),
        disk_info=_disk_info(),
        gpu_count=len(gpus),
        gpus=gpus,
        driver_version=driver_version,
        cuda_version=cuda_version,
        raw_outputs=raw_outputs,
    )

    if hardware_config is not None:
        result.machine_id = hardware_config.machine_id
        result.machine_label = hardware_config.machine_label
        result.location_type = hardware_config.location_type
        result.provider = hardware_config.provider
        result.hourly_price_usd = hardware_config.hourly_price_usd
        result.tags = hardware_config.tags
        result.expected_gpus = [g.model_dump() for g in hardware_config.expected_gpus]

        for expected in hardware_config.expected_gpus:
            matched = sum(
                1 for g in gpus if expected.name_contains.lower() in g.get("name", "").lower()
            )
            if matched < expected.count:
                logger.warning(
                    "Expected %d GPU(s) matching '%s', found %d",
                    expected.count,
                    expected.name_contains,
                    matched,
                )

    return result


def run_probe(output: Path | str, hardware_config_path: Path | str | None = None) -> Path:
    hw_config = load_hardware_config(hardware_config_path) if hardware_config_path else None
    result = probe_hardware(hw_config)
    out = Path(output)
    write_json(out, result)
    logger.info("Hardware probe written to %s", out)
    return out

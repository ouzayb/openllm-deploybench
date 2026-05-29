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


def _parse_gpus_from_nvidia_smi_csv(stdout: str) -> list[dict[str, Any]]:
    """Fallback GPU list when NVML Python bindings fail."""
    gpus: list[dict[str, Any]] = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        name = parts[1]
        uuid = parts[2] if len(parts) > 2 else ""
        info: dict[str, Any] = {
            "index": idx,
            "name": name,
            "uuid": uuid,
            "source": "nvidia-smi",
        }
        if len(parts) > 3 and parts[3]:
            try:
                info["memory_total_mb"] = float(parts[3])
            except ValueError:
                pass
        if len(parts) > 4 and parts[4]:
            try:
                info["power_draw_watts"] = float(parts[4])
            except ValueError:
                pass
        if len(parts) > 5 and parts[5]:
            try:
                info["temperature_c"] = float(parts[5])
            except ValueError:
                pass
        gpus.append(info)
    return gpus


def _parse_driver_from_smi(stdout: str) -> str | None:
    m = re.search(r"Driver Version:\s*(\S+)", stdout)
    return m.group(1) if m else None


def _collect_pynvml_gpus() -> tuple[list[dict[str, Any]], str | None, str | None]:
    from deploybench.nvml_helper import nvml_init, nvml_shutdown

    gpus: list[dict[str, Any]] = []
    driver_version: str | None = None
    cuda_version: str | None = None
    nvml, init_err = nvml_init()
    if nvml is None or init_err:
        if init_err:
            logger.warning("pynvml GPU collection failed: %s", init_err)
        return gpus, driver_version, cuda_version
    try:
        driver_version = nvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode("utf-8", errors="replace")
        try:
            cuda_version = nvml.nvmlSystemGetCudaDriverVersion_v2()
            if isinstance(cuda_version, int):
                major = cuda_version // 1000
                minor = (cuda_version % 1000) // 10
                cuda_version = f"{major}.{minor}"
        except Exception:
            cuda_version = None

        count = nvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = nvml.nvmlDeviceGetHandleByIndex(i)
            name = nvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            uuid = nvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8", errors="replace")
            mem = nvml.nvmlDeviceGetMemoryInfo(handle)
            info: dict[str, Any] = {
                "index": i,
                "name": name,
                "uuid": uuid,
                "memory_total_mb": mem.total / (1024 * 1024),
                "source": "nvml",
            }
            try:
                power_limit = nvml.nvmlDeviceGetPowerManagementLimit(handle)
                info["power_limit_watts"] = power_limit / 1000.0
            except Exception:
                pass
            try:
                max_limit = nvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
                info["max_power_limit_watts"] = max_limit[1] / 1000.0
            except Exception:
                pass
            try:
                power = nvml.nvmlDeviceGetPowerUsage(handle)
                info["power_draw_watts"] = power / 1000.0
            except Exception:
                pass
            try:
                temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                info["temperature_c"] = temp
            except Exception:
                pass
            try:
                pci = nvml.nvmlDeviceGetPciInfo(handle)
                info["pci_bus"] = pci.busId.decode() if isinstance(pci.busId, bytes) else pci.busId
            except Exception:
                pass
            try:
                mig = nvml.nvmlDeviceGetMigMode(handle)
                info["mig_mode"] = mig
            except Exception:
                pass
            gpus.append(info)
        nvml_shutdown()
    except Exception as e:
        logger.warning("NVML GPU enumeration failed: %s", e)
        nvml_shutdown()
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

    smi_query = _run_command([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ])
    raw_outputs["nvidia_smi_query"] = smi_query

    if not gpus and smi_query.get("returncode") == 0 and smi_query.get("stdout"):
        gpus = _parse_gpus_from_nvidia_smi_csv(smi_query["stdout"])
        logger.info("GPU list collected via nvidia-smi fallback (%d GPU(s))", len(gpus))

    if "nvidia_smi" in raw_outputs:
        smi = raw_outputs["nvidia_smi"]
        if "stdout" in smi:
            if cuda_version is None:
                cuda_version = _parse_cuda_from_smi(smi["stdout"])
            if driver_version is None:
                driver_version = _parse_driver_from_smi(smi["stdout"])

    if not gpus and "nvidia_smi" in raw_outputs:
        smi_out = raw_outputs["nvidia_smi"].get("stdout", "")
        if "NVML" in smi_out or "Driver/library version mismatch" in smi_out:
            logger.warning(
                "nvidia-smi reports driver/NVML issues. Try: sudo reboot, or reload "
                "the NVIDIA kernel module after a driver update."
            )

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

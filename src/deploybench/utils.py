"""Shared utilities for OpenLLM DeployBench."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_yaml(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at root in {p}")
    return data


def file_hash(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def config_hash_from_paths(*paths: Path | str | None) -> str:
    h = hashlib.sha256()
    for path in paths:
        if path is None:
            continue
        p = Path(path)
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def collect_relevant_env() -> dict[str, str]:
    prefixes = ("CUDA_", "NCCL_", "VLLM_", "HF_", "HUGGING", "TOKENIZERS")
    out: dict[str, str] = {}
    for key, val in os.environ.items():
        if any(key.startswith(p) for p in prefixes):
            out[key] = val
    return out


def append_jsonl(path: Path | str, record: dict[str, Any] | Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(record, "model_dump"):
        data = record.model_dump(mode="json")
    else:
        data = record
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, default=str) + "\n")


def write_json(path: Path | str, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "model_dump"):
        payload = data.model_dump(mode="json")
    else:
        payload = data
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_jsonl_files(directory: Path | str) -> list[Path]:
    d = Path(directory)
    if not d.exists():
        return []
    return sorted(d.rglob("*.jsonl"))


def get_package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    for name, module_name in [
        ("vllm_version", "vllm"),
        ("torch_version", "torch"),
        ("transformers_version", "transformers"),
    ]:
        try:
            mod = __import__(module_name)
            versions[name] = getattr(mod, "__version__", None)
        except ImportError:
            versions[name] = None
    return versions


def ensure_src_on_path() -> None:
    src = PROJECT_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

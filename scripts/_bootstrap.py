"""Add src to path for script execution without install."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)

# Load CUDA env before importing deploybench (nvcc / FlashInfer)
_env_cuda = ROOT / "scripts" / "env.cuda.sh"
if _env_cuda.exists():
    for _line in _env_cuda.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line.startswith("export ") and "=" in _line:
            _k, _, _v = _line[7:].partition("=")
            _k, _v = _k.strip(), _v.strip().strip('"')
            if _k and _v and "${" not in _v:
                os.environ[_k] = _v

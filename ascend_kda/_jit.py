# Copyright (c) 2026 Contributors to ascend-kda-catlass.
# Licensed under the MIT License.
"""Source-based JIT loader for AscendC + CATLASS KDA kernels.

Cache key incorporates:
  1. SHA-256 digest of kernel/header/binding C++ sources;
  2. Detected CANN toolkit version;
  3. Target NPU architecture (default: dav-2201 / Atlas A2);
  4. PyTorch + torch_npu versions.

This makes every compiled artifact reproducible, source-inspectable, and safe
across CANN/PyTorch upgrades (matching the source-JIT contract discussed in
fla-org/flash-linear-attention#1531).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch

_LOADED_OPS = None
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CSRC_DIR = _REPO_ROOT / "csrc"


def _source_digest() -> str:
    h = hashlib.sha256()
    for rel in (
        "CMakeLists.txt",
        "include/ascend_kda_ops.h",
        "kernel/kda_chunk.asc",
        "src/kda_ops_torch.cpp",
    ):
        p = _CSRC_DIR / rel
        h.update(rel.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _detect_cann_version() -> str:
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "")
    if not ascend_home:
        return "unknown"
    for cand in (
        Path(ascend_home) / "ascend_toolkit_install.info",
        Path(ascend_home) / "opp" / "version.info",
        Path(ascend_home) / "version.info",
    ):
        if cand.is_file():
            m = re.search(r"version\s*=\s*([^\s]+)", cand.read_text(errors="ignore"), re.IGNORECASE)
            if m:
                return re.sub(r"[^A-Za-z0-9._-]", "_", m.group(1))
    return "unknown"


def _resolve_catlass_root() -> Path:
    env_root = os.environ.get("CATLASS_ROOT", "").strip()
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if (p / "include" / "catlass" / "catlass.hpp").is_file():
            return p
        raise FileNotFoundError(f"CATLASS_ROOT={p} does not contain include/catlass/catlass.hpp")
    bundled = _REPO_ROOT / "third_party" / "catlass"
    if (bundled / "include" / "catlass" / "catlass.hpp").is_file():
        return bundled
    raise FileNotFoundError(
        "CATLASS headers not found. Set CATLASS_ROOT=/path/to/catlass or clone "
        "https://gitcode.com/cann/catlass into third_party/catlass."
    )


def build_cache_key(arch: str = "dav-2201") -> str:
    torch_ver = re.sub(r"[^A-Za-z0-9._-]", "_", torch.__version__)
    try:
        import torch_npu

        npu_ver = re.sub(r"[^A-Za-z0-9._-]", "_", getattr(torch_npu, "__version__", "unknown"))
    except Exception:
        npu_ver = "unknown"
    cann_ver = _detect_cann_version()
    return f"src-{_source_digest()}_cann-{cann_ver}_{arch}_pt-{torch_ver}_npu-{npu_ver}"


def _build_so_jit(cache_so: Path, arch: str = "dav-2201") -> Path:
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "").strip()
    if not ascend_home:
        raise RuntimeError("ASCEND_HOME_PATH is not set. Source your CANN set_env.sh before importing ascend_kda.")
    catlass_root = _resolve_catlass_root()
    build_dir = cache_so.parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    jobs = str(min(16, os.cpu_count() or 4))

    cmake_cmd = [
        "cmake",
        "-S",
        str(_CSRC_DIR),
        "-B",
        str(build_dir),
        f"-DPython3_EXECUTABLE={sys.executable}",
        f"-DCATLASS_ROOT={catlass_root}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    subprocess.run(cmake_cmd, check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "-j", jobs], check=True)

    built = build_dir / "libascend_kda_ops.so"
    if not built.is_file():
        raise FileNotFoundError(f"Expected built library at {built}")
    tmp_so = cache_so.with_suffix(".so.tmp")
    shutil.copy2(built, tmp_so)
    tmp_so.replace(cache_so)
    return cache_so


def load_kda_ops(force_rebuild: bool = False):
    """Load `torch.ops.ascend_kda_ops`, building via source-JIT if not already cached."""
    global _LOADED_OPS
    if _LOADED_OPS is not None and not force_rebuild:
        return _LOADED_OPS

    import torch_npu  # noqa: F401

    prebuilt = _CSRC_DIR / "libascend_kda_ops.so"
    if prebuilt.is_file() and not force_rebuild and os.environ.get("ASCEND_KDA_FORCE_JIT", "0") != "1":
        so_path = prebuilt
    else:
        cache_root = Path(os.environ.get("ASCEND_KDA_CACHE_DIR", Path.home() / ".cache" / "ascend_kda_jit"))
        key = build_cache_key()
        so_path = cache_root / key / "libascend_kda_ops.so"
        if force_rebuild or not so_path.is_file():
            so_path = _build_so_jit(so_path)

    torch.ops.load_library(str(so_path))
    _LOADED_OPS = torch.ops.ascend_kda_ops
    return _LOADED_OPS

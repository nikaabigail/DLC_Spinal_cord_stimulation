#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_environment.py - environment probe / pre-flight check for the
DLC-Live real-time inference bridge (single_rt_dlc_live_bridge.py).

Run this on a *target* PC to see exactly what is installed correctly and
what still has to be installed/fixed before the live bridge will run.

Design goals:
  * STDLIB ONLY for the harness itself. Every third-party import (torch,
    cv2, gxipy, ...) is guarded so the script runs on a half-set-up box.
  * Reads the pinned freeze (requirements.lock.txt) as the source of truth
    and compares INSTALLED vs PINNED for the critical packages.
  * Prints one clear verdict line per requirement: [ OK ] / [MISSING] /
    [MISMATCH] / [ WARN ], then a summary "N OK / M MISSING / K MISMATCH"
    plus an actionable TODO list.

Recommended invocation (uses the project venv so installed-package checks
are meaningful):
    C:\\dlc_live_env\\Scripts\\python.exe _migration\\check_environment.py

Exit code: 0 if no MISSING and no MISMATCH, else 1.
"""
from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Locations / expected values (mirror the deployed config + memory snapshot)
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent           # ...\_migration
WORK_DIR = SCRIPT_DIR.parent                            # ...\DLC_OBS_Spinal_cord_stimulation
LOCK_FILE = SCRIPT_DIR / "requirements.lock.txt"

EXPECTED_PY = (3, 10)                                   # 3.10.x

# Critical packages to verify against the lock file. import_name maps the
# distribution name (key, as it appears in the freeze) to its importable
# module so we can ALSO confirm it actually imports, not just that the
# version string matches. None => version-only check (do not try to import).
CRITICAL_PKGS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "deeplabcut-live": "dlclive",
    "numpy": "numpy",
    "opencv-python": "cv2",
    "scipy": "scipy",
    "pandas": "pandas",
    "pillow": "PIL",
    "PyYAML": "yaml",
    "ruamel.yaml": "ruamel.yaml",
    "tables": "tables",
    "timm": "timm",
    "huggingface_hub": "huggingface_hub",
    "dlclibrary": "dlclibrary",
    "networkx": "networkx",
    "safetensors": "safetensors",
}

# Expected GPU compute capability: Blackwell sm_120 (RTX 5070 / 5070 Laptop).
EXPECTED_CAPABILITY = (12, 0)

GALAXY_SDK_ROOT = Path(
    os.getenv("DLC_LIVE_GALAXY_SDK_ROOT", r"C:\Program Files\Daheng Imaging\GalaxySDK")
)
GALAXY_PYTHON_SDK = GALAXY_SDK_ROOT / "Samples" / "Python SDK"
GALAXY_GENTL64 = GALAXY_SDK_ROOT / "GenTL" / "Win64"

MODEL_PATH = Path(
    os.getenv(
        "DLC_LIVE_MODEL_PATH",
        r"C:\dlc\project\r_tm_side-og-2024-10-25\exported-models-pytorch"
        r"\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5"
        r"\DLC_r_tm_side_resnet_50_iteration-0_shuffle-5_snapshot-best-380.pt",
    )
)

CAMERA_CONFIG_DIR = Path(os.getenv("DLC_LIVE_GALAXY_CONFIG_PATH_DIR", r"C:\config_daheng"))

# Files that MUST be present in the work dir for the live bridge to run.
DEPLOYED_FILES = [
    "single_rt_dlc_live_bridge.py",
    "dual_rt_dlc_live.py",
    "config_dual_rt_dlc_live.py",
    "config_rt_dlc_live.py",
    "live_profiles.py",
    "run_live_profile.py",
    "rt_dlc_live.py",
    "send_dual_dlc_bridge_test.py",
    "check_online_buffering.py",
    os.path.join("optimization", "run_live_cudagraphs.py"),
    os.path.join("optimization", "ab_compare.py"),
    os.path.join("optimization", "ab_pose_dump.py"),
    os.path.join("optimization", "crop_range.py"),
]

# --------------------------------------------------------------------------
# Tiny report harness (no deps)
# --------------------------------------------------------------------------
OK, MISSING, MISMATCH, WARN = "OK", "MISSING", "MISMATCH", "WARN"
_TAG = {OK: "[  OK  ]", MISSING: "[MISSING]", MISMATCH: "[MISMATCH]", WARN: "[ WARN ]"}

results: list[tuple[str, str, str]] = []  # (status, title, detail)
todos: list[str] = []


def report(status: str, title: str, detail: str = "", todo: str = "") -> None:
    results.append((status, title, detail))
    line = f"{_TAG[status]:<10} {title}"
    if detail:
        line += f"\n           -> {detail}"
    print(line)
    if todo and status in (MISSING, MISMATCH, WARN):
        todos.append(todo)


def section(name: str) -> None:
    print()
    print("=" * 74)
    print(f" {name}")
    print("=" * 74)


# --------------------------------------------------------------------------
# 1. Python version
# --------------------------------------------------------------------------
def check_python() -> None:
    section("1. Python interpreter")
    v = sys.version_info
    detail = f"running {platform.python_version()}  ({sys.executable})"
    if (v.major, v.minor) == EXPECTED_PY:
        report(OK, f"Python == {EXPECTED_PY[0]}.{EXPECTED_PY[1]}.x", detail)
    else:
        report(
            MISMATCH,
            f"Python == {EXPECTED_PY[0]}.{EXPECTED_PY[1]}.x",
            detail + f" -- expected {EXPECTED_PY[0]}.{EXPECTED_PY[1]}.x",
            todo=(
                f"Install/select Python {EXPECTED_PY[0]}.{EXPECTED_PY[1]} and recreate the venv "
                r"(C:\dlc_live_env). The +cu128 torch wheels are built for cp310."
            ),
        )
    # warn if NOT running the venv interpreter (installed-pkg checks would be misleading)
    exe = Path(sys.executable).resolve()
    expected_venv = Path(r"C:\dlc_live_env\Scripts\python.exe")
    if exe != expected_venv.resolve() if expected_venv.exists() else True:
        if "dlc_live_env" not in str(exe).lower():
            report(
                WARN,
                "Running inside the dlc_live_env venv",
                f"current interpreter is {exe}",
                todo=(
                    r"Re-run this check with C:\dlc_live_env\Scripts\python.exe so the "
                    "pip-package versions reflect the live env, not the system Python."
                ),
            )


# --------------------------------------------------------------------------
# 2. Pinned pip packages vs requirements.lock.txt
# --------------------------------------------------------------------------
def _parse_lock() -> dict[str, str]:
    pins: dict[str, str] = {}
    if not LOCK_FILE.exists():
        return pins
    for raw in LOCK_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, _, ver = line.partition("==")
        pins[name.strip()] = ver.strip()
    return pins


def _installed_version(dist_name: str) -> str | None:
    """Distribution version via importlib.metadata (stdlib, no pip call)."""
    try:
        from importlib import metadata as md
    except Exception:  # pragma: no cover
        return None
    # Try the name as-is, then a couple of normalised forms.
    candidates = {dist_name, dist_name.replace(".", "-"), dist_name.replace("-", "_"),
                  dist_name.lower(), dist_name.replace("-", ".")}
    for cand in candidates:
        try:
            return md.version(cand)
        except Exception:
            continue
    return None


def check_packages() -> None:
    section("2. Critical pip packages (installed vs pinned)")
    pins = _parse_lock()
    if not pins:
        report(
            MISSING,
            "requirements.lock.txt readable",
            f"could not read {LOCK_FILE}",
            todo=f"Make sure {LOCK_FILE} is bundled with the migration.",
        )
        return
    report(OK, "requirements.lock.txt readable", f"{len(pins)} pins loaded from {LOCK_FILE.name}")

    for dist, mod in CRITICAL_PKGS.items():
        pinned = pins.get(dist)
        installed = _installed_version(dist)
        title = f"{dist}=={pinned}" if pinned else dist
        if pinned is None:
            report(WARN, dist, "not present in lock file (skipping pin compare)")
            continue
        if installed is None:
            report(
                MISSING,
                title,
                "not installed in this interpreter",
                todo=f"pip install {dist}=={pinned}",
            )
            continue
        if installed == pinned:
            # also try to import the module so we catch broken installs
            note = f"installed {installed}"
            if mod:
                try:
                    importlib.import_module(mod)
                except Exception as exc:  # importable name differs / broken DLLs
                    report(
                        MISMATCH,
                        title,
                        f"version OK ({installed}) but 'import {mod}' failed: {exc}",
                        todo=f"Reinstall {dist}=={pinned}; check its native deps.",
                    )
                    continue
            report(OK, title, note)
        else:
            report(
                MISMATCH,
                title,
                f"installed {installed}  (expected {pinned})",
                todo=f"pip install {dist}=={pinned}   # currently {installed}",
            )


# --------------------------------------------------------------------------
# 3. torch / CUDA / GPU
# --------------------------------------------------------------------------
def check_torch_cuda() -> None:
    section("3. torch CUDA runtime & GPU")
    try:
        import torch  # noqa
    except Exception as exc:
        report(
            MISSING,
            "import torch",
            f"{exc}",
            todo=(
                "Install the CUDA 12.8 wheel: pip install torch==2.10.0+cu128 "
                "torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 "
                "--index-url https://download.pytorch.org/whl/cu128"
            ),
        )
        return

    report(OK, "import torch", f"torch {torch.__version__}")

    build = getattr(torch.version, "cuda", None)
    if build:
        report(OK, "torch bundled CUDA runtime", f"CUDA {build} (wheel-bundled, no Toolkit needed)")
    else:
        report(
            MISMATCH,
            "torch bundled CUDA runtime",
            "this looks like a CPU-only torch build",
            todo="Reinstall the +cu128 CUDA wheel of torch (see above).",
        )

    if not torch.cuda.is_available():
        report(
            MISSING,
            "torch.cuda.is_available()",
            "False -- no usable CUDA GPU visible to torch",
            todo=(
                "Install a recent Blackwell-capable NVIDIA driver (>= the one that reports "
                "CUDA 12.8+). Confirm with nvidia-smi. Then re-test torch.cuda.is_available()."
            ),
        )
        return
    report(OK, "torch.cuda.is_available()", "True")

    try:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
    except Exception as exc:
        report(MISMATCH, "GPU query", f"{exc}")
        return

    report(OK, "GPU device", f"{name}")
    if cap == EXPECTED_CAPABILITY:
        report(OK, f"compute capability == {EXPECTED_CAPABILITY}", f"sm_{cap[0]}{cap[1]} (Blackwell)")
    else:
        report(
            WARN,
            f"compute capability == {EXPECTED_CAPABILITY}",
            f"got sm_{cap[0]}{cap[1]} (expected sm_120 Blackwell)",
            todo=(
                "Target GPU is not sm_120. The model still runs if torch supports this arch, "
                "but performance pins were tuned for Blackwell."
            ),
        )

    # Tiny on-device sanity op (allocates on the GPU, confirms the runtime works).
    try:
        _ = (torch.ones(8, device="cuda") * 2).sum().item()
        report(OK, "CUDA tensor op", "GPU allocation + compute succeeded")
    except Exception as exc:
        report(
            MISMATCH,
            "CUDA tensor op",
            f"GPU op failed: {exc}",
            todo="Driver/runtime mismatch -- update NVIDIA driver to a Blackwell-capable version.",
        )


# --------------------------------------------------------------------------
# 4. NVIDIA driver via nvidia-smi
# --------------------------------------------------------------------------
def check_nvidia_smi() -> None:
    section("4. NVIDIA driver (nvidia-smi)")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
    except FileNotFoundError:
        report(
            MISSING,
            "nvidia-smi on PATH",
            "nvidia-smi not found",
            todo="Install the NVIDIA GPU driver (Blackwell-capable). nvidia-smi ships with it.",
        )
        return
    except Exception as exc:
        report(WARN, "nvidia-smi", f"failed to run: {exc}")
        return

    if out.returncode != 0:
        report(
            WARN,
            "nvidia-smi",
            (out.stderr or out.stdout or "non-zero exit").strip(),
            todo="Check the NVIDIA driver install.",
        )
        return
    report(OK, "nvidia-smi", out.stdout.strip().replace("\n", " | "))


# --------------------------------------------------------------------------
# 5. Daheng Galaxy SDK + gxipy + GenTL env
# --------------------------------------------------------------------------
def check_galaxy_sdk() -> None:
    section("5. Daheng Galaxy SDK + gxipy")
    if GALAXY_SDK_ROOT.exists():
        report(OK, "Galaxy SDK root", str(GALAXY_SDK_ROOT))
    else:
        report(
            MISSING,
            "Galaxy SDK root",
            f"not found: {GALAXY_SDK_ROOT}",
            todo=(
                "Install Daheng Galaxy SDK (1.18.2208.9301 or compatible) from Daheng Imaging. "
                "The installer also sets the GENICAM_* env vars and PATH entries."
            ),
        )

    # gxipy lives inside Samples/Python SDK, not on pip. Add it to sys.path then import.
    if GALAXY_PYTHON_SDK.exists():
        report(OK, "Galaxy Python SDK folder", str(GALAXY_PYTHON_SDK))
        sdk_str = str(GALAXY_PYTHON_SDK)
        if sdk_str not in sys.path:
            sys.path.insert(0, sdk_str)
        # gxipy also needs the GenTL DLL dir; add it so the import succeeds.
        try:
            if GALAXY_GENTL64.exists() and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(GALAXY_GENTL64))
        except Exception:
            pass
        try:
            importlib.import_module("gxipy")
            report(OK, "import gxipy", "imported from Samples/Python SDK")
        except Exception as exc:
            report(
                MISMATCH,
                "import gxipy",
                f"folder present but import failed: {exc}",
                todo=(
                    "gxipy import failed -- ensure the Galaxy runtime DLLs are installed and "
                    "GENICAM_GENTL64_PATH points to GenTL\\Win64."
                ),
            )
    else:
        report(
            MISSING,
            "Galaxy Python SDK folder",
            f"not found: {GALAXY_PYTHON_SDK}",
            todo="Reinstall the Galaxy SDK with the 'Samples/Python SDK' component (provides gxipy).",
        )

    gentl_env = os.getenv("GENICAM_GENTL64_PATH")
    if gentl_env and Path(gentl_env).exists():
        report(OK, "GENICAM_GENTL64_PATH env var", gentl_env)
    elif gentl_env:
        report(
            MISMATCH,
            "GENICAM_GENTL64_PATH env var",
            f"set to a path that does not exist: {gentl_env}",
            todo=r"Point GENICAM_GENTL64_PATH at <GalaxySDK>\GenTL\Win64.",
        )
    else:
        report(
            MISSING,
            "GENICAM_GENTL64_PATH env var",
            "not set (the code re-adds it at runtime, but the SDK installer should set it)",
            todo=(
                "Set GENICAM_GENTL64_PATH = "
                rf"{GALAXY_GENTL64}  (normally set by the Galaxy SDK installer; "
                "reboot after install so system env vars propagate)."
            ),
        )


# --------------------------------------------------------------------------
# 6. Model snapshot
# --------------------------------------------------------------------------
def check_model() -> None:
    section("6. Exported DLC model snapshot (.pt)")
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        report(OK, "model snapshot present", f"{MODEL_PATH}  ({size_mb:.0f} MB)")
    else:
        report(
            MISSING,
            "model snapshot present",
            f"not found: {MODEL_PATH}",
            todo=(
                "Copy the exported model (snapshot-best-380.pt + its folder) to the path above, "
                "or set DLC_LIVE_MODEL_PATH to where you placed it."
            ),
        )


# --------------------------------------------------------------------------
# 7. Camera config dir
# --------------------------------------------------------------------------
def check_camera_configs() -> None:
    section("7. Camera config directory")
    if CAMERA_CONFIG_DIR.exists():
        txts = sorted(p.name for p in CAMERA_CONFIG_DIR.glob("*.txt"))
        if txts:
            report(OK, "camera config dir", f"{CAMERA_CONFIG_DIR}  ({len(txts)} .txt: {', '.join(txts[:4])}{'...' if len(txts) > 4 else ''})")
        else:
            report(
                WARN,
                "camera config dir",
                f"{CAMERA_CONFIG_DIR} exists but has no .txt config files",
                todo="Copy the Daheng GalaxyView feature-config .txt files into C:\\config_daheng.",
            )
    else:
        report(
            MISSING,
            "camera config dir",
            f"not found: {CAMERA_CONFIG_DIR}",
            todo=r"Create C:\config_daheng and copy the camera .txt config(s) into it.",
        )


# --------------------------------------------------------------------------
# 8. Deployed .py files in the work dir
# --------------------------------------------------------------------------
def check_deployed_files() -> None:
    section("8. Deployed scripts in the work dir")
    missing = [f for f in DEPLOYED_FILES if not (WORK_DIR / f).exists()]
    for f in DEPLOYED_FILES:
        if (WORK_DIR / f).exists():
            report(OK, f, "")
        else:
            report(
                MISSING,
                f,
                f"absent under {WORK_DIR}",
                todo=(
                    "git clone gives an OLD/broken tree -- copy the actual WORKING TREE of "
                    f"{WORK_DIR} (missing: {f})."
                ),
            )
    if not missing:
        report(OK, "all deployed scripts present", f"{len(DEPLOYED_FILES)} files under {WORK_DIR}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    print("DLC-Live migration environment check")
    print(f"work dir : {WORK_DIR}")
    print(f"lock file: {LOCK_FILE}")

    check_python()
    check_packages()
    check_torch_cuda()
    check_nvidia_smi()
    check_galaxy_sdk()
    check_model()
    check_camera_configs()
    check_deployed_files()

    n_ok = sum(1 for s, _, _ in results if s == OK)
    n_missing = sum(1 for s, _, _ in results if s == MISSING)
    n_mismatch = sum(1 for s, _, _ in results if s == MISMATCH)
    n_warn = sum(1 for s, _, _ in results if s == WARN)

    section("SUMMARY")
    print(f"  {n_ok} OK / {n_missing} MISSING / {n_mismatch} MISMATCH / {n_warn} WARN")

    if todos:
        print()
        print("TODO (in priority order):")
        for i, t in enumerate(dict.fromkeys(todos), 1):  # dedupe, keep order
            print(f"  {i:>2}. {t}")
    else:
        print()
        print("  No action needed -- environment looks ready for the live bridge.")

    print()
    if n_missing == 0 and n_mismatch == 0:
        print("RESULT: READY (no MISSING / MISMATCH).")
        return 0
    print("RESULT: NOT READY -- resolve the TODO items above, then re-run this check.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""
Non-invasive LIVE test of torch.compile(backend="cudagraphs") on the single-camera
bridge. Monkeypatches dual_rt_dlc_live.run_raw_inference so the model is compiled
right after the first inference, then runs the unmodified single bridge.

Does NOT edit any production file and does NOT touch git. Accuracy-neutral
(cudagraphs replays identical kernels). Use to confirm the LIVE fps gain that the
offline harness predicted (~22->~17ms inference).

Usage (PowerShell, inside dlc_live_env):
  C:\\dlc_live_env\\Scripts\\python.exe optimization\\run_live_cudagraphs.py
  # forwards to: single_rt_dlc_live_bridge --profile single-best --no-display
  # or pass your own args, e.g.:
  C:\\dlc_live_env\\Scripts\\python.exe optimization\\run_live_cudagraphs.py --profile single-best --display

Watch the log for:
  [cudagraphs] compiled runner.model      <- patch took effect
  stage_profile ... inference=~17 ...      <- inference dropped vs ~21 baseline
Press q / Esc (or Ctrl+C) to stop, then check live_fps / inference in the log.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dual_rt_dlc_live as dual

_orig_run_raw_inference = dual.run_raw_inference
_state = {"compiled": False}


def _patched_run_raw_inference(dlc_live, initialized, packet):
    result = _orig_run_raw_inference(dlc_live, initialized, packet)
    if initialized and not _state["compiled"]:
        _state["compiled"] = True
        try:
            import torch
            dlc_live.runner.model = torch.compile(dlc_live.runner.model, backend="cudagraphs")
            print("[cudagraphs] compiled runner.model (backend=cudagraphs)", flush=True)
        except Exception as exc:  # stay eager, run continues
            print(f"[cudagraphs] compile failed, staying eager: {exc}", flush=True)
    return result


dual.run_raw_inference = _patched_run_raw_inference

# Default to the recommended headless single-best run; allow CLI override.
if len(sys.argv) <= 1:
    forwarded = ["--profile", "single-best", "--no-display"]
else:
    forwarded = sys.argv[1:]

import single_rt_dlc_live_bridge as bridge

sys.argv = ["single_rt_dlc_live_bridge.py"] + forwarded
print(f"[cudagraphs] launching single bridge with args: {forwarded}", flush=True)
bridge.main()

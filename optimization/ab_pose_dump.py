"""
A/B pose-dump harness for DLCLive optimization experiments.

Runs the EXACT production inference path (``dual_rt_dlc_live.run_raw_inference``)
over a RECORDED video and dumps per-frame 6-point pose + likelihood + timing to
a CSV. Compare against an FP32 baseline with ``ab_compare.py``.

Isolated from production: imports existing modules, writes only a CSV, does not
modify any config on disk and does not touch git.

Color note: a recorded .avi decoded by OpenCV is BGR, while the live Galaxy
camera delivers RGB with convert2rgb=False. To reproduce the same tensor the
model sees live, the harness feeds the BGR video with convert2rgb=True. Use
--source-color rgb only if your clip is already RGB.

Examples (PowerShell, inside dlc_live_env):
  python optimization\\ab_pose_dump.py --video CLIP.avi --variant fp32 --out optimization\\dump_full.csv
  python optimization\\ab_pose_dump.py --video CLIP.avi --variant fp32 --dynamic "0.5,150" --out optimization\\dump_dyn.csv
  python optimization\\ab_pose_dump.py --video CLIP.avi --variant fp32 --compile --compile-backend cudagraphs --out optimization\\dump_cg.csv
  python optimization\\ab_compare.py optimization\\dump_full.csv optimization\\dump_cg.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DLCLive A/B pose dump over a recorded video.")
    p.add_argument("--video", required=True, help="Path to a recorded .avi/.mp4 (left camera).")
    p.add_argument("--variant", default="fp32", choices=["fp32", "fp16"], help="Model precision.")
    p.add_argument("--crop", default="", help="Static DLCLive CROPPING 'x1,x2,y1,y2'. Empty = none.")
    p.add_argument("--dynamic", default="", help="Dynamic cropping 'thresh,margin' (e.g. '0.5,150'). Empty = off.")
    p.add_argument("--source-color", default="bgr", choices=["bgr", "rgb"],
                   help="Decoded frame color. 'bgr' (normal OpenCV) -> convert2rgb=True.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile(runner.model) after init (accuracy-neutral). Use on FULL frame, not with --dynamic.")
    p.add_argument("--compile-backend", default="cudagraphs",
                   help="torch.compile backend. 'cudagraphs' (no Triton, best for overhead-bound), 'inductor' (needs Triton), 'aot_eager'.")
    p.add_argument("--max-frames", type=int, default=0, help="0 = whole video.")
    p.add_argument("--warmup", type=int, default=5, help="Frames excluded from timing averages.")
    p.add_argument("--out", required=True, help="Output CSV path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import logging
    import numpy as np

    import config_dual_rt_dlc_live as config
    import rt_dlc_live as live
    import dual_rt_dlc_live as dual

    live.config = config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("ab_pose_dump")

    config.USE_VIDEO_FILE = True
    config.VIDEO_FILE_PATH = args.video
    config.PRECISION = "FP16" if args.variant == "fp16" else "FP32"
    config.CONVERT_TO_RGB = (args.source_color == "bgr")
    if args.crop.strip():
        x1, x2, y1, y2 = (int(v) for v in args.crop.split(","))
        config.CROPPING = [x1, x2, y1, y2]
        logger.info("CROPPING set to %s", config.CROPPING)
    if args.dynamic.strip():
        thr, margin = args.dynamic.split(",")
        config.DYNAMIC_CROPPING = (True, float(thr), int(margin))
        logger.info("DYNAMIC_CROPPING set to %s", config.DYNAMIC_CROPPING)

    source = live.VideoFileSource(Path(args.video), target_fps=0.0, skip_if_behind=False)
    source.open()
    ok, first = source.read()
    if not ok or first is None:
        raise SystemExit(f"Cannot read first frame from: {args.video}")

    cropping = live.normalize_cropping_for_frame(getattr(config, "CROPPING", None), first.frame, logger)
    dlc_live = live.build_dlc_live(cropping)
    model_cfg = dlc_live.read_config()
    body_parts = live.extract_bodyparts(model_cfg)
    bodypart_to_idx = {name: i for i, name in enumerate(body_parts)}
    points = list(config.DUAL_USE_POINTS)

    missing = [pt for pt in points if pt not in bodypart_to_idx]
    if missing:
        raise SystemExit(f"USE_POINTS missing from model: {missing}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(out_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    header = ["frame_id"]
    for name in points:
        header += [f"{name}_x", f"{name}_y", f"{name}_l"]
    header += ["preprocess_ms", "model_ms", "infer_ms"]
    writer.writerow(header)

    logger.info("variant=%s precision=%s convert2rgb=%s compile=%s backend=%s crop=%s dynamic=%s",
                args.variant, config.PRECISION, config.CONVERT_TO_RGB, args.compile,
                args.compile_backend, getattr(config, "CROPPING", None), args.dynamic or "off")

    initialized = False
    compiled = False
    orig_model = None      # kept so we can revert if a compile backend fails at runtime
    packet = first
    n = 0
    model_ms_sum = 0.0
    timed = 0
    warmup = int(args.warmup)

    while packet is not None:
        try:
            initialized, pose, pre_ms, model_ms = dual.run_raw_inference(dlc_live, initialized, packet)
        except Exception as exc:
            if compiled and orig_model is not None:
                logger.warning("compiled inference failed (%s); reverting to eager and retrying", exc)
                dlc_live.runner.model = orig_model
                orig_model = None
                initialized, pose, pre_ms, model_ms = dual.run_raw_inference(dlc_live, initialized, packet)
            else:
                raise

        if args.compile and not compiled and initialized:
            import torch
            orig_model = dlc_live.runner.model
            try:
                dlc_live.runner.model = torch.compile(orig_model, backend=args.compile_backend)
                logger.info("torch.compile(backend=%s) applied; excluding next 40 frames as warmup", args.compile_backend)
            except Exception as exc:
                logger.warning("torch.compile call failed (%s); staying eager", exc)
                orig_model = None
            compiled = True
            model_ms_sum = 0.0
            timed = 0
            warmup = n + 40

        arr = dual.pose_to_compact_array(pose, bodypart_to_idx, points)
        row = [int(packet.frame_id)]
        for i in range(len(points)):
            x = float(arr[i, 0])
            y = float(arr[i, 1])
            lk = float(arr[i, 2])
            row.append(f"{x:.3f}" if np.isfinite(x) else "")
            row.append(f"{y:.3f}" if np.isfinite(y) else "")
            row.append(f"{lk:.4f}" if np.isfinite(lk) else "")
        row.append(f"{pre_ms:.3f}")
        row.append(f"{model_ms:.3f}")
        row.append(f"{pre_ms + model_ms:.3f}")
        writer.writerow(row)

        if n >= warmup:
            model_ms_sum += model_ms
            timed += 1
        n += 1
        if args.max_frames and n >= args.max_frames:
            break
        ok, packet = source.read()
        if not ok:
            packet = None

    handle.close()
    source.release()
    try:
        dlc_live.close()
    except Exception:
        pass

    avg_model_ms = model_ms_sum / max(1, timed)
    print(f"[done] variant={args.variant} dynamic={args.dynamic or 'off'} "
          f"compile={args.compile}({args.compile_backend if args.compile else '-'}) "
          f"frames={n} avg_model_ms={avg_model_ms:.2f} out={out_path}")


if __name__ == "__main__":
    main()

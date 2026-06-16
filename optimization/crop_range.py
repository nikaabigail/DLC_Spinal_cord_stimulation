"""
Measure where the tracked points actually land in a pose dump, and propose a
static DLCLive CROPPING box for task A2 (horizontal crop ~2x faster inference,
accuracy-safe because coordinates are restored).

Reads one ab_pose_dump CSV (run on a REPRESENTATIVE session/clip) and prints
per-point and overall x/y ranges of CONFIDENT detections, then a suggested
CROPPING=[x1, x2, y1, y2] with margin, snapped so width is a multiple of --pad
(the model's pad_width_divisor; pass the value from the exported model cfg).

Pure standard library.

Usage:
  python optimization/crop_range.py optimization/dump_fp32.csv --min-lik 0.3 --margin 60 --pad 32 --frame-w 1920 --frame-h 220
"""
from __future__ import annotations

import argparse
import csv
import math


def fnum(value: str):
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Suggest a static crop from a pose dump.")
    ap.add_argument("dump")
    ap.add_argument("--min-lik", type=float, default=0.3, help="Only count points with likelihood >= this.")
    ap.add_argument("--margin", type=int, default=60, help="Margin (px) added around the observed range.")
    ap.add_argument("--pad", type=int, default=1, help="Snap crop WIDTH to a multiple of this (pad_width_divisor).")
    ap.add_argument("--frame-w", type=int, default=1920)
    ap.add_argument("--frame-h", type=int, default=220)
    args = ap.parse_args()

    with open(args.dump, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        cols = list(rows[0].keys()) if rows else []

    point_names = sorted({c[:-2] for c in cols if c.endswith("_x")})
    gx_min, gx_max = math.inf, -math.inf
    gy_min, gy_max = math.inf, -math.inf

    print(f"{'point':12s} {'n':>7s} {'x_min':>8s} {'x_max':>8s} {'y_min':>8s} {'y_max':>8s}")
    for name in point_names:
        xs, ys = [], []
        for r in rows:
            lk = fnum(r.get(f"{name}_l", ""))
            x = fnum(r.get(f"{name}_x", ""))
            y = fnum(r.get(f"{name}_y", ""))
            if x is None or y is None or lk is None or lk < args.min_lik:
                continue
            xs.append(x)
            ys.append(y)
        if not xs:
            print(f"{name:12s} {0:7d} {'-':>8s} {'-':>8s} {'-':>8s} {'-':>8s}")
            continue
        print(f"{name:12s} {len(xs):7d} {min(xs):8.1f} {max(xs):8.1f} {min(ys):8.1f} {max(ys):8.1f}")
        gx_min, gx_max = min(gx_min, min(xs)), max(gx_max, max(xs))
        gy_min, gy_max = min(gy_min, min(ys)), max(gy_max, max(ys))

    if not math.isfinite(gx_min):
        print("\nNo confident points found; lower --min-lik or check the dump.")
        return

    x1 = max(0, int(gx_min) - args.margin)
    x2 = min(args.frame_w, int(math.ceil(gx_max)) + args.margin)
    y1 = max(0, int(gy_min) - args.margin)
    y2 = min(args.frame_h, int(math.ceil(gy_max)) + args.margin)

    # snap width up to a multiple of --pad (keep inside the frame)
    width = x2 - x1
    if args.pad > 1 and width % args.pad != 0:
        new_width = ((width // args.pad) + 1) * args.pad
        x2 = min(args.frame_w, x1 + new_width)
        if x2 - x1 < new_width:  # ran into right edge; shift x1 left
            x1 = max(0, x2 - new_width)

    full_px = args.frame_w * args.frame_h
    crop_px = (x2 - x1) * (y2 - y1)
    speedup = full_px / max(1, crop_px)

    print()
    print(f"observed (lik>={args.min_lik}): x[{gx_min:.0f}..{gx_max:.0f}] y[{gy_min:.0f}..{gy_max:.0f}]")
    print(f"suggested CROPPING = [{x1}, {x2}, {y1}, {y2}]   (DLCLive [x1,x2,y1,y2])")
    print(f"crop size = {x2 - x1} x {y2 - y1} px   (~{speedup:.2f}x fewer pixels vs {args.frame_w}x{args.frame_h})")
    print("Validate: run ab_pose_dump --crop \"{0},{1},{2},{3}\" and ab_compare vs the no-crop baseline.".format(x1, x2, y1, y2))


if __name__ == "__main__":
    main()

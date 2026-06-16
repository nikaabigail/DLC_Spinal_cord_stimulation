"""
Compare two ab_pose_dump CSVs and report whether a change is accuracy-safe.

For each tracked point it computes the Euclidean displacement (px) between the
baseline and candidate over frames where BOTH have a finite (x, y), plus
likelihood deltas and the inference-time speedup.

Accuracy gate: median Δpx of the angle-relevant points must be <= --thresh
(default 1.0 px) to count as "no accuracy loss".

Pure standard library (no numpy) so it runs anywhere.

Usage:
  python optimization/ab_compare.py optimization/dump_fp32.csv optimization/dump_fp16.csv
  python optimization/ab_compare.py base.csv cand.csv --thresh 1.0 --key hl_hip_l,hl_ankle_l,hl_toes_l
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def load(path: str) -> tuple[list[dict], list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def fnum(value: str):
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def avg_col(rows: list[dict], col: str) -> float:
    vals = [fnum(r.get(col, "")) for r in rows]
    vals = [v for v in vals if v is not None]
    return mean(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two DLCLive pose dumps.")
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--thresh", type=float, default=1.0, help="Accuracy gate on median Δpx (key points).")
    ap.add_argument("--key", default="hl_hip_l,hl_ankle_l,hl_toes_l",
                    help="Comma-separated points that must pass the gate (the trigger triplet).")
    args = ap.parse_args()

    base_rows, base_cols = load(args.baseline)
    cand_rows, _ = load(args.candidate)
    cand_by_id = {int(r["frame_id"]): r for r in cand_rows}

    point_names = sorted({c[:-2] for c in base_cols if c.endswith("_x")})
    key_points = [p.strip() for p in args.key.split(",") if p.strip()]

    print(f"baseline : {Path(args.baseline).name}  ({len(base_rows)} frames)")
    print(f"candidate: {Path(args.candidate).name}  ({len(cand_rows)} frames)")
    print()

    # timing
    b_model = avg_col(base_rows, "model_ms")
    c_model = avg_col(cand_rows, "model_ms")
    speedup = (b_model / c_model) if (c_model and math.isfinite(c_model) and c_model > 0) else float("nan")
    print(f"SPEED   avg model_ms: baseline={b_model:.2f}  candidate={c_model:.2f}  speedup x{speedup:.2f}")
    print()

    print(f"ACCURACY (Δpx between matched finite points; gate median<= {args.thresh}px on key points)")
    print(f"{'point':12s} {'n':>6s} {'medΔpx':>8s} {'p95Δpx':>8s} {'maxΔpx':>8s} {'Δlik(mean)':>11s}  gate")
    worst_key = 0.0
    any_fail = False
    for name in point_names:
        d_list: list[float] = []
        dl_list: list[float] = []
        for br in base_rows:
            fid = int(br["frame_id"])
            cr = cand_by_id.get(fid)
            if cr is None:
                continue
            bx, by = fnum(br.get(f"{name}_x", "")), fnum(br.get(f"{name}_y", ""))
            cx, cy = fnum(cr.get(f"{name}_x", "")), fnum(cr.get(f"{name}_y", ""))
            if None in (bx, by, cx, cy):
                continue
            d_list.append(math.hypot(cx - bx, cy - by))
            bl, cl = fnum(br.get(f"{name}_l", "")), fnum(cr.get(f"{name}_l", ""))
            if bl is not None and cl is not None:
                dl_list.append(cl - bl)
        d_sorted = sorted(d_list)
        med = pct(d_sorted, 0.5)
        p95 = pct(d_sorted, 0.95)
        mx = d_sorted[-1] if d_sorted else float("nan")
        dlm = mean(dl_list) if dl_list else float("nan")
        is_key = name in key_points
        gate = ""
        if is_key and d_sorted:
            ok = med <= args.thresh
            gate = "PASS" if ok else "FAIL"
            worst_key = max(worst_key, med)
            any_fail = any_fail or (not ok)
        print(f"{name:12s} {len(d_list):6d} {med:8.2f} {p95:8.2f} {mx:8.2f} {dlm:11.3f}  {gate}")

    print()
    verdict = "NO ACCURACY LOSS" if not any_fail else "ACCURACY REGRESSION"
    print(f"VERDICT: {verdict}  (worst key median Δ = {worst_key:.2f}px, gate {args.thresh}px)")
    if not math.isfinite(speedup):
        print("NOTE: speedup not computable (candidate model_ms missing/zero).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
enhance_yolopose17_kinematics.py

Kinematic enhancement + quality filtering for datasets extracted by
`filtering_yolopose17_sharded.py` (Ultralytics YOLO Pose, COCO-17 keypoints).

Input .npz expected keys:
  - keypoints: (T,K,17,3) or (T,17,3)
      last dim = (x,y,kp_conf)
  - optional frame_indices: (T,)
  - optional meta: JSON string array

Outputs per sample .npz:
  - keypoints: (T', keep_people, 17, 3)  normalized xy + original kp_conf
  - vel:      (T', keep_people, 17, 2)  d(xy)/dt in normalized space
  - acc:      (T', keep_people, 17, 2)  d(vel)/dt_mid in normalized space
  - time:     (T',) float time (frame idx or sequential), fractional after interpolation
  - meta:     JSON record with quality stats + config

What you asked for (implemented)
--------------------------------
- Velocity + acceleration of each keypoint (per joint, per person)
- Sample more frames via --interp_factor (adds intermediate steps)
- Intermediate steps generated using velocity and acceleration:
    constant-acceleration interpolation on each interval
- Normalize the position so the "body center" is at (0,0), with well-defined method:
    center = mid-hip if hips valid else mean of valid joints
  Optional scale + rotation:
    scale by torso length (mid-shoulder to mid-hip), rotate shoulders horizontal

No MediaPipe/OpenGL used here.

Example:
python enhance_data.py \
  --in_dir  ../data \
  --out_dir ../data_kin \
  --norm_mode center_scale_rotate \
  --interp_factor 5 \
  --keep_people 2 \
  --min_score 0.20 \
  --min_valid_joint_frac 0.12 \
  --max_speed_p95 3.0 \
  --max_acc_p95 6.0 \
  --drop_frame0 \
  --report_jsonl ../data_kin_yolo_quality.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

J = 17
# COCO17 indices for normalization
L_SH, R_SH = 5, 6
L_HIP, R_HIP = 11, 12


# -----------------------------
# IO helpers
# -----------------------------
def iter_npz(root: Path) -> List[Path]:
    return sorted(root.rglob("*.npz"))

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def parse_meta(npz_path: Path) -> Dict[str, Any]:
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            if "meta" not in z:
                return {}
            a = z["meta"]
            if a.size == 0:
                return {}
            s = a[0]
            if hasattr(s, "item"):
                s = s.item()
            if isinstance(s, bytes):
                s = s.decode("utf-8", errors="ignore")
            if isinstance(s, str):
                return json.loads(s)
    except Exception:
        return {}
    return {}

def infer_label(npz_path: Path) -> str:
    meta = parse_meta(npz_path)
    lab = str(meta.get("label", meta.get("label_name", ""))).strip().lower()
    if lab and lab not in ("unknown", "none", "null"):
        return lab
    return npz_path.parent.name.strip().lower()


# -----------------------------
# Mask + kinematics
# -----------------------------
def valid_mask(xy: np.ndarray, score: np.ndarray, min_score: float) -> np.ndarray:
    return (
        np.isfinite(xy[..., 0])
        & np.isfinite(xy[..., 1])
        & np.isfinite(score)
        & (score >= float(min_score))
    )

def vel_acc_time(xy: np.ndarray, m: np.ndarray, time: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    xy:   (T,J,2)
    m:    (T,J)
    time: (T,) float, strictly increasing preferred (frame indices ok)

    Returns:
      vel: (T,J,2)  vel[t] = (xy[t]-xy[t-1]) / dt[t]
      acc: (T,J,2)  acc[t] = (vel[t]-vel[t-1]) / dt_mid[t]
      mv:  (T,J) valid mask for vel
      ma:  (T,J) valid mask for acc
    """
    T, Jj, _ = xy.shape
    assert Jj == J

    vel = np.zeros((T, J, 2), np.float32)
    acc = np.zeros((T, J, 2), np.float32)
    mv = np.zeros((T, J), bool)
    ma = np.zeros((T, J), bool)

    if T < 2:
        return vel, acc, mv, ma

    dt = np.diff(time).astype(np.float32)  # (T-1,)
    dt = np.where(np.isfinite(dt) & (dt > 1e-6), dt, 1.0).astype(np.float32)

    # velocity at t>=1
    mv[1:] = m[1:] & m[:-1]
    dxy = xy[1:] - xy[:-1]
    vel[1:] = dxy / dt[:, None, None]
    vel[~mv] = 0.0

    if T < 3:
        return vel, acc, mv, ma

    # acceleration at t>=2
    # time between midpoints of intervals: 0.5*(dt[t-1] + dt[t-2]) for acc at t
    dt_mid = 0.5 * (dt[1:] + dt[:-1])  # (T-2,)
    dt_mid = np.where(np.isfinite(dt_mid) & (dt_mid > 1e-6), dt_mid, 1.0).astype(np.float32)

    ma[2:] = mv[2:] & mv[1:-1]
    dv = vel[2:] - vel[1:-1]
    acc[2:] = dv / dt_mid[:, None, None]
    acc[~ma] = 0.0

    return vel, acc, mv, ma


def upsample_const_accel(
    xy: np.ndarray,
    m: np.ndarray,
    acc: np.ndarray,
    ma: np.ndarray,
    time: np.ndarray,
    factor: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Insert (factor-1) intermediate steps between each time[t] and time[t+1]
    using constant acceleration on each interval.

    For interval [t0, t1], dt = t1 - t0, alpha in [0,1):
      v0 = (x1 - x0)/dt - 0.5*a0*dt
      x(alpha) = x0 + v0*(alpha*dt) + 0.5*a0*(alpha*dt)^2

    If acceleration is invalid at t0 (ma[t0]==False), we use a0=0 (reduces to linear interpolation).
    Only joints with BOTH endpoints valid are filled; others become NaN.
    """
    factor = int(factor)
    if factor <= 1 or xy.shape[0] <= 1:
        return xy, m, time

    T, Jj, _ = xy.shape
    assert Jj == J

    Tout = (T - 1) * factor + 1
    xy_u = np.full((Tout, J, 2), np.nan, np.float32)
    m_u = np.zeros((Tout, J), bool)
    t_u = np.zeros((Tout,), np.float32)

    out_i = 0
    for t in range(T - 1):
        x0 = xy[t]
        x1 = xy[t + 1]
        ok = m[t] & m[t + 1]

        t0 = float(time[t])
        t1 = float(time[t + 1])
        dt = t1 - t0
        if not np.isfinite(dt) or dt <= 1e-6:
            dt = 1.0

        delta = x1 - x0

        # acceleration at t (per joint); if invalid -> 0
        a0 = np.where(ma[t][:, None], acc[t], 0.0).astype(np.float32)
        v0 = (delta / dt) - 0.5 * a0 * dt

        for s in range(factor):
            alpha = s / float(factor)
            tau = alpha * dt
            x = x0 + v0 * tau + 0.5 * a0 * (tau * tau)
            xy_u[out_i] = np.where(ok[:, None], x, np.nan)
            m_u[out_i] = ok
            t_u[out_i] = t0 + tau
            out_i += 1

    xy_u[out_i] = xy[T - 1]
    m_u[out_i] = m[T - 1]
    t_u[out_i] = float(time[T - 1])

    return xy_u, m_u, t_u


# -----------------------------
# Normalization
# -----------------------------
def _safe_center_mean(xy: np.ndarray, m: np.ndarray) -> np.ndarray:
    w = m.astype(np.float32)[:, :, None]  # (T,J,1)
    num = (xy * w).sum(axis=1)            # (T,2)
    den = np.maximum(w.sum(axis=1), 1e-6) # (T,1)
    return num / den

def normalize_xy(
    xy: np.ndarray,
    m: np.ndarray,
    mode: str,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    xy: (T,17,2), m:(T,17)
    modes:
      none
      center
      center_scale
      center_scale_rotate

    Center definition (well-defined):
      if both hips valid: center = 0.5*(L_HIP + R_HIP)
      else: center = mean(valid joints)
    Then x <- x - center.

    Scale:
      torso_len = || mid_shoulder - mid_hip ||
      fallback: RMS radius of joints
      x <- x / torso_len

    Rotate:
      rotate so shoulder vector (L_SH - R_SH) aligns to +x axis
    """
    info: Dict[str, float] = {}
    mode = str(mode).lower()
    if mode == "none":
        return xy, info

    T, Jj, _ = xy.shape
    assert Jj == J

    mean_center = _safe_center_mean(xy, m)
    hip_ok = m[:, L_HIP] & m[:, R_HIP]
    center = np.zeros((T, 2), np.float32)
    center[hip_ok] = 0.5 * (xy[hip_ok, L_HIP] + xy[hip_ok, R_HIP])
    center[~hip_ok] = mean_center[~hip_ok]

    xy_c = xy - center[:, None, :]

    if mode in ("center_scale", "center_scale_rotate"):
        sh_ok = m[:, L_SH] & m[:, R_SH]
        mid_sh = np.zeros((T, 2), np.float32)
        mid_sh[sh_ok] = 0.5 * (xy[sh_ok, L_SH] + xy[sh_ok, R_SH])
        mid_sh[~sh_ok] = mean_center[~sh_ok]

        torso = np.linalg.norm(mid_sh - center, axis=1)  # (T,)

        rad2 = xy_c[..., 0] ** 2 + xy_c[..., 1] ** 2
        rad2 = np.where(m, rad2, np.nan)
        rms = np.sqrt(np.nanmean(rad2, axis=1))
        rms = np.where(np.isfinite(rms), rms, 1.0)

        torso = np.where(np.isfinite(torso) & (torso > eps), torso, rms)
        torso = np.maximum(torso, 1.0)
        xy_c = xy_c / torso[:, None, None]
        info["scale_mean"] = float(np.mean(torso))

    if mode == "center_scale_rotate":
        sh_ok = m[:, L_SH] & m[:, R_SH]
        vec = xy_c[:, L_SH] - xy_c[:, R_SH]
        ang = np.arctan2(vec[:, 1], vec[:, 0])  # (T,)
        ca = np.cos(-ang).astype(np.float32)
        sa = np.sin(-ang).astype(np.float32)

        x = xy_c[..., 0]
        y = xy_c[..., 1]
        xr = x * ca[:, None] - y * sa[:, None]
        yr = x * sa[:, None] + y * ca[:, None]
        xy_r = np.stack([xr, yr], axis=2).astype(np.float32)

        xy_c = np.where(sh_ok[:, None, None], xy_r, xy_c)

    center_after = _safe_center_mean(xy_c, m)
    info["center_abs_mean"] = float(np.mean(np.abs(center_after)))
    info["center_abs_p95"] = float(np.nanpercentile(np.abs(center_after), 95))

    return xy_c.astype(np.float32), info


# -----------------------------
# Track selection + stats
# -----------------------------
def pick_top_tracks(kps: np.ndarray, min_score: float, keep_people: int) -> List[int]:
    """
    Pick top-N tracks by total valid joints across time.
    kps: (T,K,17,3)
    """
    score = kps[..., 2]
    xy = kps[..., 0:2]
    valid = (
        np.isfinite(xy[..., 0])
        & np.isfinite(xy[..., 1])
        & np.isfinite(score)
        & (score >= float(min_score))
    )
    counts = valid.sum(axis=(0, 2))  # (K,)
    order = np.argsort(counts)[::-1]
    out = [int(i) for i in order[: int(keep_people)] if int(counts[i]) > 0]
    if not out and kps.shape[1] > 0:
        out = [0]
    return out

def speed_p95(vel: np.ndarray, mv: np.ndarray) -> float:
    sp = np.sqrt(vel[..., 0] ** 2 + vel[..., 1] ** 2)
    sp = np.where(mv, sp, np.nan)
    if np.all(~np.isfinite(sp)):
        return float("inf")
    return float(np.nanpercentile(sp, 95))

def acc_p95(acc: np.ndarray, ma: np.ndarray) -> float:
    ac = np.sqrt(acc[..., 0] ** 2 + acc[..., 1] ** 2)
    ac = np.where(ma, ac, np.nan)
    if np.all(~np.isfinite(ac)):
        return float("inf")
    return float(np.nanpercentile(ac, 95))


# -----------------------------
# Main processing
# -----------------------------
@dataclass
class ProcCfg:
    norm_mode: str
    interp_factor: int
    keep_people: int
    min_score: float
    min_valid_joint_frac: float
    max_speed_p95: float
    max_acc_p95: float
    drop_frame0: bool


def process_one(npz_path: Path, out_dir: Path, cfg: ProcCfg) -> Tuple[bool, Dict[str, Any], Optional[Path]]:
    lab = infer_label(npz_path)
    meta_in = parse_meta(npz_path)

    with np.load(npz_path, allow_pickle=False) as z:
        kps = z["keypoints"].astype(np.float32)
        frame_indices = z["frame_indices"].astype(np.float32) if "frame_indices" in z else None

    # normalize shapes
    if kps.ndim == 3 and kps.shape[1] == J and kps.shape[2] == 3:
        kps = kps[:, None, :, :]  # (T,1,17,3)
    if kps.ndim != 4 or kps.shape[2] != J or kps.shape[3] != 3:
        return False, {"reason": "bad_shape", "shape": list(kps.shape)}, None

    T = kps.shape[0]
    if frame_indices is None or frame_indices.shape[0] != T:
        time = np.arange(T, dtype=np.float32)
    else:
        time = frame_indices.astype(np.float32)

    # optional drop frame0
    if cfg.drop_frame0 and T >= 2 and float(time[0]) == 0.0:
        kps = kps[1:]
        time = time[1:]
        T = kps.shape[0]

    keep_idx = pick_top_tracks(kps, min_score=cfg.min_score, keep_people=cfg.keep_people)
    if len(keep_idx) == 0:
        return False, {"reason": "no_valid_tracks"}, None

    track_stats: List[Dict[str, Any]] = []
    kept_tracks = 0
    final_kps = None
    final_vel = None
    final_acc = None
    time_out = None

    for out_slot in range(cfg.keep_people):
        if out_slot >= len(keep_idx):
            track_stats.append({"kept": False, "reason": "missing_track"})
            continue

        kk = keep_idx[out_slot]
        kp = kps[:, kk, :, :]  # (T,17,3)
        xy = kp[:, :, 0:2]
        sc = kp[:, :, 2]

        m = valid_mask(xy, sc, min_score=cfg.min_score)
        valid_frac = float(np.mean(m))
        if valid_frac < cfg.min_valid_joint_frac:
            track_stats.append({"kept": False, "reason": "low_valid_frac", "valid_frac": valid_frac})
            continue

        # normalize
        xy_n, norm_info = normalize_xy(xy.astype(np.float32), m, mode=cfg.norm_mode)
        xy_n = np.where(m[:, :, None], xy_n, np.nan)

        # vel/acc (pre-interp)
        v, a, mv, ma = vel_acc_time(np.nan_to_num(xy_n, nan=0.0).astype(np.float32), m, time)
        sp95 = speed_p95(v, mv)
        ac95 = acc_p95(a, ma)

        if sp95 > cfg.max_speed_p95:
            track_stats.append({"kept": False, "reason": "speed_p95", "speed_p95": sp95, "valid_frac": valid_frac})
            continue
        if ac95 > cfg.max_acc_p95:
            track_stats.append({"kept": False, "reason": "acc_p95", "acc_p95": ac95, "valid_frac": valid_frac})
            continue

        # interpolate (upsample)
        xy_i, m_i, t_i = upsample_const_accel(
            np.nan_to_num(xy_n, nan=0.0).astype(np.float32),
            m,
            a.astype(np.float32),
            ma,
            time.astype(np.float32),
            factor=cfg.interp_factor,
        )
        xy_i = np.where(m_i[:, :, None], xy_i, np.nan)

        # recompute vel/acc after interpolation
        v2, a2, mv2, ma2 = vel_acc_time(np.nan_to_num(xy_i, nan=0.0).astype(np.float32), m_i, t_i)

        track_stats.append({
            "kept": True,
            "track_index": int(kk),
            "valid_frac": valid_frac,
            "speed_p95_pre": sp95,
            "acc_p95_pre": ac95,
            "norm": norm_info,
            "interp_factor": int(cfg.interp_factor),
        })

        if kept_tracks == 0:
            Tout = xy_i.shape[0]
            time_out = t_i.astype(np.float32)
            final_kps = np.full((Tout, cfg.keep_people, J, 3), np.nan, dtype=np.float32)
            final_vel = np.zeros((Tout, cfg.keep_people, J, 2), dtype=np.float32)
            final_acc = np.zeros((Tout, cfg.keep_people, J, 2), dtype=np.float32)
        else:
            Tout = int(final_kps.shape[0])  # type: ignore
            # pad/truncate defensively
            if xy_i.shape[0] != Tout:
                if xy_i.shape[0] > Tout:
                    xy_i = xy_i[:Tout]
                    m_i = m_i[:Tout]
                    v2 = v2[:Tout]
                    a2 = a2[:Tout]
                else:
                    pad = Tout - xy_i.shape[0]
                    xy_i = np.pad(xy_i, ((0, pad), (0, 0), (0, 0)), mode="edge")
                    m_i = np.pad(m_i, ((0, pad), (0, 0)), mode="edge")
                    v2 = np.pad(v2, ((0, pad), (0, 0), (0, 0)), mode="constant")
                    a2 = np.pad(a2, ((0, pad), (0, 0), (0, 0)), mode="constant")

        kp_out = np.full((Tout, J, 3), np.nan, dtype=np.float32)
        kp_out[:, :, 0:2] = xy_i

        # preserve kp_conf trend (pad/truncate to Tout)
        sc2 = sc
        if sc2.shape[0] >= Tout:
            sc2 = sc2[:Tout]
        else:
            sc2 = np.pad(sc2, ((0, Tout - sc2.shape[0]), (0, 0)), mode="edge")
        kp_out[:, :, 2] = sc2

        final_kps[:, out_slot, :, :] = kp_out  # type: ignore
        final_vel[:, out_slot, :, :] = v2      # type: ignore
        final_acc[:, out_slot, :, :] = a2      # type: ignore

        kept_tracks += 1

    if kept_tracks == 0 or final_kps is None or final_vel is None or final_acc is None or time_out is None:
        return False, {"reason": "all_tracks_filtered", "track_stats": track_stats}, None

    out_path = out_dir / lab / f"{npz_path.stem}__kin.npz"
    ensure_dir(out_path.parent)

    rec: Dict[str, Any] = {
        "src_npz": str(npz_path),
        "label": lab,
        "layout": "coco17",
        "cfg": {
            "norm_mode": cfg.norm_mode,
            "interp_factor": int(cfg.interp_factor),
            "keep_people": int(cfg.keep_people),
            "min_score": float(cfg.min_score),
            "min_valid_joint_frac": float(cfg.min_valid_joint_frac),
            "max_speed_p95": float(cfg.max_speed_p95),
            "max_acc_p95": float(cfg.max_acc_p95),
            "drop_frame0": bool(cfg.drop_frame0),
        },
        "track_stats": track_stats,
        "meta_in": meta_in,
        "shapes": {
            "keypoints": list(final_kps.shape),
            "vel": list(final_vel.shape),
            "acc": list(final_acc.shape),
        },
    }

    np.savez_compressed(
        out_path,
        keypoints=final_kps,
        vel=final_vel,
        acc=final_acc,
        time=time_out,
        meta=np.array([json.dumps(rec)], dtype=np.string_),
    )

    return True, rec, out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--norm_mode", type=str, default="center_scale_rotate",
                    choices=["none", "center", "center_scale", "center_scale_rotate"])
    ap.add_argument("--interp_factor", type=int, default=1)
    ap.add_argument("--keep_people", type=int, default=2)
    ap.add_argument("--min_score", type=float, default=0.20)
    ap.add_argument("--min_valid_joint_frac", type=float, default=0.10)
    ap.add_argument("--max_speed_p95", type=float, default=3.0)
    ap.add_argument("--max_acc_p95", type=float, default=6.0)

    ap.add_argument("--drop_frame0", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report_jsonl", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    cfg = ProcCfg(
        norm_mode=args.norm_mode,
        interp_factor=max(1, int(args.interp_factor)),
        keep_people=max(1, int(args.keep_people)),
        min_score=float(args.min_score),
        min_valid_joint_frac=float(args.min_valid_joint_frac),
        max_speed_p95=float(args.max_speed_p95),
        max_acc_p95=float(args.max_acc_p95),
        drop_frame0=bool(args.drop_frame0),
    )

    files = iter_npz(in_dir)
    if args.limit and args.limit > 0:
        files = files[: int(args.limit)]
    if not files:
        raise RuntimeError(f"No .npz found under {in_dir}")

    report_f = None
    if args.report_jsonl:
        rp = Path(args.report_jsonl).expanduser().resolve()
        ensure_dir(rp.parent)
        report_f = open(rp, "w", encoding="utf-8")

    kept = 0
    skipped = 0
    for p in tqdm(files, desc="Enhancing"):
        lab = infer_label(p)
        out_path = out_dir / lab / f"{p.stem}__kin.npz"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        ok, rec, outp = process_one(p, out_dir, cfg)
        if ok:
            kept += 1
        else:
            skipped += 1

        if report_f is not None:
            report_f.write(json.dumps({"ok": ok, **rec}) + "\n")
            report_f.flush()

    if report_f is not None:
        report_f.close()

    print(f"[OK] kept={kept} skipped/failed={skipped}")
    print(f"[OK] out_dir={out_dir}")


if __name__ == "__main__":
    main()
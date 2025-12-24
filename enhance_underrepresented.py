#!/usr/bin/env python3
"""
Re-extract underrepresented classes (fear, disgust) with relaxed requirements.

This script:
1. Finds samples that weren't enhanced due to strict quality filters
2. Re-processes them with relaxed thresholds
3. Uses data augmentation to generate more training samples
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

J = 17  # COCO-17 keypoints
L_SH, R_SH = 5, 6
L_HIP, R_HIP = 11, 12


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def valid_mask(xy: np.ndarray, score: np.ndarray, min_score: float) -> np.ndarray:
    return (
        np.isfinite(xy[..., 0])
        & np.isfinite(xy[..., 1])
        & np.isfinite(score)
        & (score >= float(min_score))
    )


def vel_acc_time(xy: np.ndarray, m: np.ndarray, time: np.ndarray):
    T, Jj, _ = xy.shape
    vel = np.zeros((T, J, 2), np.float32)
    acc = np.zeros((T, J, 2), np.float32)
    mv = np.zeros((T, J), bool)
    ma = np.zeros((T, J), bool)

    if T < 2:
        return vel, acc, mv, ma

    dt = np.diff(time).astype(np.float32)
    dt = np.where(np.isfinite(dt) & (dt > 1e-6), dt, 1.0)

    mv[1:] = m[1:] & m[:-1]
    dxy = xy[1:] - xy[:-1]
    vel[1:] = dxy / dt[:, None, None]
    vel[~mv] = 0.0

    if T < 3:
        return vel, acc, mv, ma

    dt_mid = 0.5 * (dt[1:] + dt[:-1])
    dt_mid = np.where(np.isfinite(dt_mid) & (dt_mid > 1e-6), dt_mid, 1.0)

    ma[2:] = mv[2:] & mv[1:-1]
    dv = vel[2:] - vel[1:-1]
    acc[2:] = dv / dt_mid[:, None, None]
    acc[~ma] = 0.0

    return vel, acc, mv, ma


def _safe_center_mean(xy: np.ndarray, m: np.ndarray) -> np.ndarray:
    w = m.astype(np.float32)[:, :, None]
    num = (xy * w).sum(axis=1)
    den = np.maximum(w.sum(axis=1), 1e-6)
    return num / den


def normalize_xy(xy: np.ndarray, m: np.ndarray, mode: str = "center_scale"):
    T, Jj, _ = xy.shape
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

        torso = np.linalg.norm(mid_sh - center, axis=1)
        rad2 = xy_c[..., 0] ** 2 + xy_c[..., 1] ** 2
        rad2 = np.where(m, rad2, np.nan)
        rms = np.sqrt(np.nanmean(rad2, axis=1))
        rms = np.where(np.isfinite(rms), rms, 1.0)
        torso = np.where(np.isfinite(torso) & (torso > 1e-6), torso, rms)
        torso = np.maximum(torso, 1.0)
        xy_c = xy_c / torso[:, None, None]

    return xy_c.astype(np.float32)


def interpolate_missing(xy: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate missing keypoints using neighboring frames."""
    T, J, _ = xy.shape
    xy_out = xy.copy()
    m_out = m.copy()

    for j in range(J):
        valid_idx = np.where(m[:, j])[0]
        if len(valid_idx) < 2:
            continue

        for t in range(T):
            if m[t, j]:
                continue
            # Find nearest valid frames
            before = valid_idx[valid_idx < t]
            after = valid_idx[valid_idx > t]

            if len(before) > 0 and len(after) > 0:
                t0, t1 = before[-1], after[0]
                alpha = (t - t0) / (t1 - t0)
                xy_out[t, j] = (1 - alpha) * xy[t0, j] + alpha * xy[t1, j]
                m_out[t, j] = True
            elif len(before) > 0:
                xy_out[t, j] = xy[before[-1], j]
                m_out[t, j] = True
            elif len(after) > 0:
                xy_out[t, j] = xy[after[0], j]
                m_out[t, j] = True

    return xy_out, m_out


def augment_sequence(kps: np.ndarray, vel: np.ndarray, acc: np.ndarray,
                     time: np.ndarray, aug_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply augmentation to generate more samples."""
    T = kps.shape[0]

    if aug_type == "temporal_crop_start":
        # Crop first 20% of frames
        start = int(T * 0.2)
        if start > 5:
            return kps[start:], vel[start:], acc[start:], time[start:]

    elif aug_type == "temporal_crop_end":
        # Crop last 20% of frames
        end = int(T * 0.8)
        if end > 5:
            return kps[:end], vel[:end], acc[:end], time[:end]

    elif aug_type == "scale_up":
        # Scale positions by 1.1
        kps_out = kps.copy()
        kps_out[..., :2] *= 1.1
        vel_out = vel * 1.1
        acc_out = acc * 1.1
        return kps_out, vel_out, acc_out, time

    elif aug_type == "scale_down":
        # Scale positions by 0.9
        kps_out = kps.copy()
        kps_out[..., :2] *= 0.9
        vel_out = vel * 0.9
        acc_out = acc * 0.9
        return kps_out, vel_out, acc_out, time

    elif aug_type == "noise":
        # Add small Gaussian noise
        kps_out = kps.copy()
        noise = np.random.randn(*kps_out[..., :2].shape).astype(np.float32) * 0.02
        kps_out[..., :2] += noise
        return kps_out, vel, acc, time

    elif aug_type == "flip_horizontal":
        # Flip x coordinates
        kps_out = kps.copy()
        kps_out[..., 0] = -kps_out[..., 0]
        vel_out = vel.copy()
        vel_out[..., 0] = -vel_out[..., 0]
        acc_out = acc.copy()
        acc_out[..., 0] = -acc_out[..., 0]
        return kps_out, vel_out, acc_out, time

    return kps, vel, acc, time


@dataclass
class RelaxedConfig:
    min_score: float = 0.10  # Relaxed from 0.20
    min_valid_joint_frac: float = 0.05  # Relaxed from 0.10
    max_speed_p95: float = 10.0  # Relaxed from 3.0
    max_acc_p95: float = 20.0  # Relaxed from 6.0
    min_frames: int = 3  # Minimum frames to keep
    keep_people: int = 2
    interp_factor: int = 2
    interpolate_missing: bool = True
    generate_augmentations: bool = True


def process_sample_relaxed(npz_path: Path, out_dir: Path, cfg: RelaxedConfig) -> List[Path]:
    """Process a single sample with relaxed requirements. Returns list of output paths."""
    outputs = []
    label = npz_path.parent.name.lower()

    try:
        with np.load(npz_path, allow_pickle=False) as z:
            kps = z["keypoints"].astype(np.float32)
            frame_indices = z.get("frame_indices", None)
            if frame_indices is not None:
                frame_indices = frame_indices.astype(np.float32)
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return []

    # Normalize shape
    if kps.ndim == 3 and kps.shape[1] == J and kps.shape[2] == 3:
        kps = kps[:, None, :, :]  # (T,1,17,3)
    if kps.ndim != 4 or kps.shape[2] != J or kps.shape[3] != 3:
        return []

    T = kps.shape[0]
    if T < cfg.min_frames:
        return []

    time = frame_indices if frame_indices is not None else np.arange(T, dtype=np.float32)

    # Pick best tracks
    score = kps[..., 2]
    xy = kps[..., 0:2]
    valid = valid_mask(xy, score, cfg.min_score)
    counts = valid.sum(axis=(0, 2))
    track_order = np.argsort(counts)[::-1]

    kept_tracks = []
    for kk in track_order[:cfg.keep_people]:
        if counts[kk] == 0:
            continue

        kp = kps[:, kk, :, :]  # (T,17,3)
        xy = kp[:, :, 0:2]
        sc = kp[:, :, 2]

        m = valid_mask(xy, sc, cfg.min_score)
        valid_frac = float(np.mean(m))

        if valid_frac < cfg.min_valid_joint_frac:
            continue

        # Interpolate missing keypoints
        if cfg.interpolate_missing:
            xy_interp, m_interp = interpolate_missing(
                np.nan_to_num(xy, nan=0.0).astype(np.float32), m
            )
        else:
            xy_interp, m_interp = xy, m

        # Normalize
        xy_n = normalize_xy(xy_interp.astype(np.float32), m_interp, mode="center_scale")
        xy_n = np.where(m_interp[:, :, None], xy_n, np.nan)

        # Compute velocity/acceleration
        vel, acc, mv, ma = vel_acc_time(
            np.nan_to_num(xy_n, nan=0.0).astype(np.float32), m_interp, time
        )

        # Check speed/acceleration (relaxed)
        speed = np.sqrt(vel[..., 0]**2 + vel[..., 1]**2)
        sp = np.where(mv, speed, np.nan)
        sp95 = np.nanpercentile(sp, 95) if not np.all(~np.isfinite(sp)) else float('inf')

        ac_mag = np.sqrt(acc[..., 0]**2 + acc[..., 1]**2)
        ac = np.where(ma, ac_mag, np.nan)
        ac95 = np.nanpercentile(ac, 95) if not np.all(~np.isfinite(ac)) else float('inf')

        if sp95 > cfg.max_speed_p95 or ac95 > cfg.max_acc_p95:
            continue

        # Build output array
        kp_out = np.full((T, J, 3), np.nan, dtype=np.float32)
        kp_out[:, :, 0:2] = xy_n
        kp_out[:, :, 2] = sc

        kept_tracks.append({
            "kps": kp_out,
            "vel": vel,
            "acc": acc,
            "time": time,
            "valid_frac": valid_frac
        })

    if not kept_tracks:
        return []

    # Combine tracks
    Tout = T
    final_kps = np.full((Tout, cfg.keep_people, J, 3), np.nan, dtype=np.float32)
    final_vel = np.zeros((Tout, cfg.keep_people, J, 2), dtype=np.float32)
    final_acc = np.zeros((Tout, cfg.keep_people, J, 2), dtype=np.float32)

    for i, track in enumerate(kept_tracks[:cfg.keep_people]):
        final_kps[:, i] = track["kps"]
        final_vel[:, i] = track["vel"]
        final_acc[:, i] = track["acc"]

    time_out = kept_tracks[0]["time"]

    # Save main output
    out_path = out_dir / label / f"{npz_path.stem}__kin.npz"
    ensure_dir(out_path.parent)

    meta = {
        "src_npz": str(npz_path),
        "label": label,
        "relaxed_extraction": True,
        "valid_frac": kept_tracks[0]["valid_frac"]
    }

    np.savez_compressed(
        out_path,
        keypoints=final_kps,
        vel=final_vel,
        acc=final_acc,
        time=time_out,
        meta=np.array([json.dumps(meta)], dtype=np.bytes_)
    )
    outputs.append(out_path)

    # Generate augmented versions
    if cfg.generate_augmentations:
        augmentations = ["scale_up", "scale_down", "noise", "flip_horizontal"]
        if T > 10:
            augmentations.extend(["temporal_crop_start", "temporal_crop_end"])

        for aug_type in augmentations:
            aug_kps, aug_vel, aug_acc, aug_time = augment_sequence(
                final_kps, final_vel, final_acc, time_out, aug_type
            )

            if aug_kps.shape[0] < cfg.min_frames:
                continue

            aug_path = out_dir / label / f"{npz_path.stem}__aug_{aug_type}__kin.npz"
            meta_aug = {**meta, "augmentation": aug_type}

            np.savez_compressed(
                aug_path,
                keypoints=aug_kps,
                vel=aug_vel,
                acc=aug_acc,
                time=aug_time,
                meta=np.array([json.dumps(meta_aug)], dtype=np.bytes_)
            )
            outputs.append(aug_path)

    return outputs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--enhanced_dir", type=str, default="./data_enhanced")
    ap.add_argument("--classes", nargs="+", default=["fear", "disgust"])
    ap.add_argument("--no_augment", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    enhanced_dir = Path(args.enhanced_dir).resolve()
    target_classes = args.classes

    cfg = RelaxedConfig()
    cfg.generate_augmentations = not args.no_augment

    print(f"Target classes: {target_classes}")
    print(f"Relaxed config:")
    print(f"  min_score: {cfg.min_score}")
    print(f"  min_valid_joint_frac: {cfg.min_valid_joint_frac}")
    print(f"  max_speed_p95: {cfg.max_speed_p95}")
    print(f"  max_acc_p95: {cfg.max_acc_p95}")
    print(f"  generate_augmentations: {cfg.generate_augmentations}")

    total_new = 0
    for cls in target_classes:
        cls_src = data_dir / cls
        cls_enh = enhanced_dir / cls

        if not cls_src.exists():
            print(f"Warning: {cls_src} not found")
            continue

        src_files = set(f.stem for f in cls_src.glob("*.npz"))
        enh_files = set(f.stem.replace("__kin", "").split("__aug_")[0] for f in cls_enh.glob("*.npz"))
        missing = src_files - enh_files

        print(f"\n{cls.upper()}: {len(missing)} samples to process")

        new_count = 0
        for stem in tqdm(list(missing), desc=f"Processing {cls}"):
            src_path = cls_src / f"{stem}.npz"
            outputs = process_sample_relaxed(src_path, enhanced_dir, cfg)
            new_count += len(outputs)

        print(f"  Created {new_count} new samples")
        total_new += new_count

    print(f"\nTotal new samples created: {total_new}")

    # Print updated class distribution
    print("\nUpdated class distribution:")
    for cls_dir in sorted(enhanced_dir.iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.glob("*.npz")))
            print(f"  {cls_dir.name}: {count}")


if __name__ == "__main__":
    main()

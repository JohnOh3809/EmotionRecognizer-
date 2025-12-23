#!/usr/bin/env python3
"""
for i in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=0 python filtering.py \
    --data_root ../CAER \
    --out_dir   ../data \
    --pose_weights yolov8n-pose.pt \
    --device cuda:0 \
    --target_fps 40 \
    --max_people 3 \
    --person_conf 0.40 \
    --scene_threshold 27 \
    --skip_frame0 \
    --num_shards 6 --shard_index $i \
    --manifest_name manifest_shard${i}.jsonl \
    > shard${i}.log 2>&1 &
done

End-to-end CAER preprocessing pipeline using **Ultralytics YOLO Pose** (native PyTorch/CUDA; no OpenGL/EGL).

What it does
------------
1) Iterate videos under --data_root (supports .avi/.mp4/.mov/.mkv)
2) Split videos into scenes (PySceneDetect ContentDetector)
3) For each scene:
   - sample a few frames and run YOLO-Pose to estimate #people + confidence
   - reject scenes outside [min_people, max_people] or below person_conf
   - optional camera stability filter (optical flow + affine motion stats)
4) For scenes that pass filters:
   - extract keypoints per frame using YOLO-Pose (COCO-17 layout)
   - keep up to --max_people persons per frame (ranked by box confidence)
   - save per-scene .npz with:
       keypoints: (T, K, 17, 3)  (x, y, kp_conf)
       bboxes:    (T, K, 4)      (x1,y1,x2,y2)
       frame_indices: (T,)
       meta: JSON with provenance + stats
   - write manifest jsonl

Output organization
-------------------
Ignores train/val/test directory structure on disk.
Saves to: out_dir/<label>/*.npz

Parallel processing
-------------------
Use sharding flags to run N instances in parallel without stepping on each other:
  --num_shards N --shard_index i

Example (6 shards, YOLO on GPU):
  for i in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES=0 python filtering_yolopose17_sharded.py ... --num_shards 6 --shard_index $i &
  done

Install
-------
pip install -U numpy opencv-python-headless tqdm scenedetect ultralytics

Notes
-----
- YOLO pose model weights: yolov8n-pose.pt (fast) / yolov8s-pose.pt (better).
- This uses COCO-17 keypoints:
    0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear,
    5 l_sh, 6 r_sh, 7 l_el, 8 r_el, 9 l_wr, 10 r_wr,
    11 l_hip, 12 r_hip, 13 l_kn, 14 r_kn, 15 l_an, 16 r_an
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


# -----------------------------
# Label normalization (CAER-ish)
# -----------------------------
def normalize_label(name: str) -> str:
    s = name.strip().lower()
    mapping = {
        "anger": "angry",
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "neutral": "neutral",
        "sad": "sad",
        "surprise": "surprise",
    }
    return mapping.get(s, s)

def infer_label(video_path: Path) -> str:
    return normalize_label(video_path.parent.name)

def scan_videos(data_root: Path, exts: Tuple[str, ...] = (".avi", ".mp4", ".mov", ".mkv")) -> List[Path]:
    vids: List[Path] = []
    for p in data_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            vids.append(p)
    vids.sort()
    return vids


# -----------------------------
# Scene detection (PySceneDetect)
# -----------------------------
def detect_scenes(video_path: Path, threshold: float) -> List[Tuple[int, int]]:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except Exception as e:
        raise RuntimeError(
            "PySceneDetect not installed. Run: pip install scenedetect[opencv]\n"
            f"Import error: {e}"
        )

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video=video, show_progress=False)

    scene_list = scene_manager.get_scene_list()
    if not scene_list:
        try:
            total_frames = int(video.duration.get_frames())
            return [(0, max(1, total_frames))]
        except Exception:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            return [(0, max(1, total_frames))]

    scenes: List[Tuple[int, int]] = []
    for start_tc, end_tc in scene_list:
        s = int(start_tc.get_frames())
        e = int(end_tc.get_frames())
        if e > s:
            scenes.append((s, e))
    return scenes


# -----------------------------
# Camera stability filter
# -----------------------------
@dataclass
class StabilityStats:
    n_pairs: int
    fail_frac: float
    trans_median: float
    trans_p95: float
    rot_median: float
    rot_p95: float
    scale_median: float
    scale_p95: float


def _estimate_motion_affine(prev_gray: np.ndarray, curr_gray: np.ndarray) -> Optional[Tuple[float, float, float]]:
    p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
    if p0 is None or len(p0) < 20:
        return None

    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
    if p1 is None or st is None:
        return None

    good0 = p0[st.flatten() == 1]
    good1 = p1[st.flatten() == 1]
    if len(good0) < 20:
        return None

    M, _ = cv2.estimateAffinePartial2D(good0, good1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None

    dx = float(M[0, 2])
    dy = float(M[1, 2])
    trans = math.sqrt(dx * dx + dy * dy)

    a = float(M[0, 0])
    c = float(M[1, 0])
    scale = math.sqrt(a * a + c * c)
    scale_delta = abs(scale - 1.0)
    rot_rad = math.atan2(c, a)
    rot_deg = abs(rot_rad * 180.0 / math.pi)

    return trans, rot_deg, scale_delta


def camera_stability_stats(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    stride: int = 5,
    max_pairs: int = 200,
    resize_w: int = 320,
) -> StabilityStats:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return StabilityStats(0, 1.0, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9)

    start_frame = max(0, int(start_frame))
    end_frame = max(start_frame + 1, int(end_frame))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, prev = cap.read()
    if not ok:
        cap.release()
        return StabilityStats(0, 1.0, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9)

    def _prep(frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if w > resize_w:
            nh = int(h * (resize_w / w))
            frame_bgr = cv2.resize(frame_bgr, (resize_w, nh), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    prev_g = _prep(prev)

    motions: List[Tuple[float, float, float]] = []
    fails = 0
    pairs = 0
    current = start_frame

    while True:
        jumped = 0
        while jumped < stride:
            if current + 1 >= end_frame:
                break
            if not cap.grab():
                break
            current += 1
            jumped += 1

        if current + 1 >= end_frame:
            break

        ok, curr = cap.read()
        current += 1
        if not ok:
            break

        curr_g = _prep(curr)
        est = _estimate_motion_affine(prev_g, curr_g)
        pairs += 1
        if est is None:
            fails += 1
        else:
            motions.append(est)

        prev_g = curr_g
        if pairs >= max_pairs:
            break

    cap.release()

    if pairs == 0:
        return StabilityStats(0, 1.0, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9)

    fail_frac = fails / float(pairs)
    if len(motions) == 0:
        return StabilityStats(pairs, fail_frac, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9)

    arr = np.array(motions, dtype=np.float32)
    trans = arr[:, 0]
    rot = arr[:, 1]
    scale = arr[:, 2]

    def _med(x: np.ndarray) -> float:
        return float(np.nanmedian(x))

    def _p95(x: np.ndarray) -> float:
        return float(np.nanpercentile(x, 95))

    return StabilityStats(
        n_pairs=pairs,
        fail_frac=float(fail_frac),
        trans_median=_med(trans),
        trans_p95=_p95(trans),
        rot_median=_med(rot),
        rot_p95=_p95(rot),
        scale_median=_med(scale),
        scale_p95=_p95(scale),
    )


# -----------------------------
# YOLO Pose wrapper
# -----------------------------
COCO17 = 17

class YOLOPose:
    def __init__(self, weights: str, device: str, imgsz: int, conf: float, half: bool = True):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError("ultralytics not installed. Run: pip install ultralytics\n" + str(e))

        self.model = YOLO(weights)
        self.device = str(device)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.half = bool(half)

    def infer(self, frame_bgr: np.ndarray):
        # returns (kps_np (P,17,3), box_xyxy_np (P,4), box_conf_np (P,))
        res = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return None, None, None

        # boxes
        box_xyxy = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)  # (P,4)
        box_conf = res.boxes.conf.detach().cpu().numpy().astype(np.float32)  # (P,)

        # keypoints tensor: prefer .data if present
        kp = getattr(res.keypoints, "data", None)
        if kp is None:
            # fall back to xy + conf
            xy = res.keypoints.xy  # (P,17,2)
            kconf = getattr(res.keypoints, "conf", None)  # (P,17)
            if kconf is None:
                kconf = np.ones((xy.shape[0], xy.shape[1]), dtype=np.float32)
            xy = xy.detach().cpu().numpy().astype(np.float32)
            kconf = kconf.detach().cpu().numpy().astype(np.float32)
            kps = np.concatenate([xy, kconf[..., None]], axis=2).astype(np.float32)
        else:
            kps = kp.detach().cpu().numpy().astype(np.float32)  # (P,17,3) typically

        if kps.ndim != 3 or kps.shape[1] != COCO17:
            # Unexpected layout
            return None, None, None

        return kps, box_xyxy, box_conf


# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sample_indices(start_frame: int, end_frame: int, n: int) -> List[int]:
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if end_frame <= start_frame:
        return []
    if n <= 1:
        return [start_frame + (end_frame - start_frame) // 2]
    span = end_frame - start_frame
    idxs = []
    for i in range(n):
        t = (i + 0.5) / n
        idx = start_frame + int(t * span)
        idx = min(max(idx, start_frame), end_frame - 1)
        idxs.append(idx)
    return sorted(set(idxs))

def read_frame_at(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    if not ok:
        return None
    return frame

def iter_scene_frames(cap: cv2.VideoCapture, start_frame: int, end_frame: int, step: int, skip_frame0: bool) -> Iterable[Tuple[int, np.ndarray]]:
    start_frame = int(start_frame)
    end_frame = int(end_frame)
    if skip_frame0 and start_frame <= 0:
        start_frame = 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    idx = start_frame
    while idx < end_frame:
        ok = cap.grab()
        if not ok:
            break
        if ((idx - start_frame) % step) == 0:
            ok2, frame = cap.retrieve()
            if not ok2:
                break
            yield idx, frame
        idx += 1


# -----------------------------
# Scene-level filters
# -----------------------------
@dataclass
class FilterCfg:
    scene_threshold: float
    min_scene_frames: int
    min_people: int
    max_people: int
    people_sample_frames: int
    person_conf: float

    # stability
    disable_stability: bool
    stable_stride: int
    stable_max_fail_frac: float
    max_trans_p95: float
    max_rot_p95: float
    max_scale_p95: float

    # pose extraction
    target_fps: int
    normalize_xy: bool
    skip_frame0: bool


def scene_passes_people_filter(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    pose_model: YOLOPose,
    cfg: FilterCfg,
) -> Tuple[bool, Dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return False, {"reason": "cap_open_failed"}
    idxs = sample_indices(start_frame, end_frame, cfg.people_sample_frames)
    counts: List[int] = []
    mean_confs: List[float] = []
    for fi in idxs:
        if cfg.skip_frame0 and fi == 0:
            fi = 1
        frame = read_frame_at(cap, fi)
        if frame is None:
            continue
        kps, box_xyxy, box_conf = pose_model.infer(frame)
        if box_conf is None or len(box_conf) == 0:
            counts.append(0)
            mean_confs.append(0.0)
            continue
        keep = box_conf >= float(cfg.person_conf)
        c = int(np.sum(keep))
        counts.append(c)
        mean_confs.append(float(np.mean(box_conf[keep])) if c > 0 else 0.0)
    cap.release()

    if len(counts) == 0:
        return False, {"reason": "people_sampling_failed"}

    med_count = int(np.median(counts))
    med_conf = float(np.median(mean_confs))
    ok = (cfg.min_people <= med_count <= cfg.max_people) and (med_conf >= cfg.person_conf)
    return ok, {
        "sample_frame_indices": idxs,
        "counts": counts,
        "mean_confs": mean_confs,
        "median_count": med_count,
        "median_conf": med_conf,
    }


def scene_passes_stability_filter(video_path: Path, start_frame: int, end_frame: int, cfg: FilterCfg) -> Tuple[bool, Dict]:
    if cfg.disable_stability:
        return True, {"disabled": True}

    stats = camera_stability_stats(
        video_path=video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        stride=cfg.stable_stride,
    )

    ok = (
        (stats.fail_frac <= cfg.stable_max_fail_frac)
        and (stats.trans_p95 <= cfg.max_trans_p95)
        and (stats.rot_p95 <= cfg.max_rot_p95)
        and (stats.scale_p95 <= cfg.max_scale_p95)
    )
    return ok, {
        "n_pairs": stats.n_pairs,
        "fail_frac": stats.fail_frac,
        "trans_p95": stats.trans_p95,
        "rot_p95": stats.rot_p95,
        "scale_p95": stats.scale_p95,
    }


# -----------------------------
# Pose extraction per scene
# -----------------------------

def _bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    a,b: (4,) xyxy
    """
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return float(inter / union)


def extract_scene_pose(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    pose_model: YOLOPose,
    cfg: FilterCfg,
    track_iou_thr: float = 0.30,
    track_max_lost: int = 3,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict]:
    """
    Extracts per-frame poses and assigns detections to *persistent track slots* (0..max_people-1)
    using greedy IoU matching. This greatly reduces "person id swapping" across frames.

    Outputs:
      keypoints (T,K,17,3), bboxes (T,K,4)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        fps = 30.0
    step = max(1, int(round(fps / float(max(1, cfg.target_fps)))))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    keypoints_list: List[np.ndarray] = []
    bboxes_list: List[np.ndarray] = []
    frame_indices: List[int] = []

    # Track state
    track_boxes = np.full((cfg.max_people, 4), np.nan, dtype=np.float32)
    track_lost = np.zeros((cfg.max_people,), dtype=np.int32)

    for fi, frame in iter_scene_frames(cap, start_frame, end_frame, step=step, skip_frame0=cfg.skip_frame0):
        kps, box_xyxy, box_conf = pose_model.infer(frame)

        out_kp = np.full((cfg.max_people, COCO17, 3), np.nan, dtype=np.float32)
        out_bb = np.full((cfg.max_people, 4), np.nan, dtype=np.float32)

        if kps is None or box_xyxy is None or box_conf is None or len(box_conf) == 0:
            # mark all tracks as lost for this frame
            track_lost += 1
            # expire tracks that have been lost too long
            for t in range(cfg.max_people):
                if track_lost[t] > int(track_max_lost):
                    track_boxes[t] = np.nan
            # write outputs
        else:
            # filter by conf
            keep = box_conf >= float(cfg.person_conf)
            kps2 = kps[keep]
            bb2 = box_xyxy[keep]
            conf2 = box_conf[keep]

            if kps2.size == 0:
                track_lost += 1
                for t in range(cfg.max_people):
                    if track_lost[t] > int(track_max_lost):
                        track_boxes[t] = np.nan
            else:
                # process detections in descending confidence
                det_order = np.argsort(conf2)[::-1]

                assigned_tracks = set()
                assigned_dets = set()

                # greedy IoU matching to existing (non-expired) tracks
                for di in det_order:
                    det_box = bb2[di]
                    # compute best IoU among unassigned tracks with valid boxes
                    best_t = -1
                    best_iou = 0.0
                    for t in range(cfg.max_people):
                        if t in assigned_tracks:
                            continue
                        if not np.isfinite(track_boxes[t]).all():
                            continue
                        iou = _bbox_iou_xyxy(det_box, track_boxes[t])
                        if iou > best_iou:
                            best_iou = iou
                            best_t = t
                    if best_t >= 0 and best_iou >= float(track_iou_thr):
                        assigned_tracks.add(best_t)
                        assigned_dets.add(di)
                        out_kp[best_t] = kps2[di]
                        out_bb[best_t] = det_box
                        track_boxes[best_t] = det_box
                        track_lost[best_t] = 0

                # assign remaining detections to empty/expired tracks
                for di in det_order:
                    if di in assigned_dets:
                        continue
                    # find first available track slot (no valid box) or longest-lost track
                    cand = [t for t in range(cfg.max_people) if t not in assigned_tracks]
                    if not cand:
                        break
                    # prefer empty tracks
                    empty = [t for t in cand if not np.isfinite(track_boxes[t]).all()]
                    if empty:
                        t = empty[0]
                    else:
                        # otherwise steal the most-lost track
                        t = int(cand[int(np.argmax(track_lost[cand]))])
                    assigned_tracks.add(t)
                    assigned_dets.add(di)
                    out_kp[t] = kps2[di]
                    out_bb[t] = bb2[di]
                    track_boxes[t] = bb2[di]
                    track_lost[t] = 0

                # increment lost for tracks not assigned this frame
                for t in range(cfg.max_people):
                    if t not in assigned_tracks:
                        track_lost[t] += 1
                        if track_lost[t] > int(track_max_lost):
                            track_boxes[t] = np.nan

        if cfg.normalize_xy and w > 0 and h > 0:
            out_kp[..., 0] = out_kp[..., 0] / float(w)
            out_kp[..., 1] = out_kp[..., 1] / float(h)

        keypoints_list.append(out_kp)
        bboxes_list.append(out_bb)
        frame_indices.append(int(fi))

    cap.release()

    kps_arr = np.stack(keypoints_list, axis=0) if keypoints_list else np.zeros((0, cfg.max_people, COCO17, 3), np.float32)
    bb_arr = np.stack(bboxes_list, axis=0) if bboxes_list else np.zeros((0, cfg.max_people, 4), np.float32)

    meta = {
        "fps": fps,
        "step": step,
        "target_fps": cfg.target_fps,
        "normalize_xy": cfg.normalize_xy,
        "frame_w": w,
        "frame_h": h,
        "layout": "coco17",
        "tracking": {
            "track_iou_thr": float(track_iou_thr),
            "track_max_lost": int(track_max_lost),
        },
    }
    return kps_arr, bb_arr, frame_indices, meta



# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--pose_weights", type=str, default="yolov8n-pose.pt")
    ap.add_argument("--device", type=str, default="cuda:0", help="YOLO device: cpu | cuda:0 etc.")
    ap.add_argument("--imgsz", type=int, default=384)
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO inference confidence threshold (boxes).")
    ap.add_argument("--half", action="store_true", help="Use FP16 on CUDA (faster).")

    # Sharding
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--manifest_name", type=str, default="manifest.jsonl")

    # Scene splitting
    ap.add_argument("--scene_threshold", type=float, default=27.0)
    ap.add_argument("--min_scene_frames", type=int, default=16)

    # People filter
    ap.add_argument("--min_people", type=int, default=1)
    ap.add_argument("--max_people", type=int, default=2)
    ap.add_argument("--person_conf", type=float, default=0.40, help="Min box confidence to count a person / keep pose.")
    ap.add_argument("--people_sample_frames", type=int, default=5)

    # Stability filter
    ap.add_argument("--disable_stability", action="store_true")
    ap.add_argument("--stable_stride", type=int, default=5)
    ap.add_argument("--stable_max_fail_frac", type=float, default=0.25)
    ap.add_argument("--max_trans_p95", type=float, default=6.0)
    ap.add_argument("--max_rot_p95", type=float, default=2.0)
    ap.add_argument("--max_scale_p95", type=float, default=0.03)

    # Pose extraction
    ap.add_argument("--target_fps", type=int, default=12)
    ap.add_argument("--normalize_xy", action="store_true", help="Store xy normalized by frame size.")
    ap.add_argument("--skip_frame0", action="store_true", help="Never use global frame 0 (mpeg4 keyframe artifacts).")

    ap.add_argument("--track_iou_thr", type=float, default=0.30, help="IoU threshold for greedy person tracking across frames.")
    ap.add_argument("--track_max_lost", type=int, default=3, help="How many consecutive missing frames before a track is reset.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ensure_dir(out_dir)

    videos = scan_videos(data_root)
    if not videos:
        raise RuntimeError(f"No videos found under {data_root}")

    num_shards = max(1, int(args.num_shards))
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= num_shards:
        raise RuntimeError(f"--shard_index must be in [0, {num_shards-1}]")

    # shard assignment
    shard_videos = [v for i, v in enumerate(videos) if (i % num_shards) == shard_index]
    print(f"[INFO] shard {shard_index}/{num_shards} videos={len(shard_videos)} total={len(videos)}")

    pose_model = YOLOPose(
        weights=args.pose_weights,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        half=bool(args.half),
    )

    cfg = FilterCfg(
        scene_threshold=float(args.scene_threshold),
        min_scene_frames=int(args.min_scene_frames),
        min_people=int(args.min_people),
        max_people=int(args.max_people),
        people_sample_frames=int(args.people_sample_frames),
        person_conf=float(args.person_conf),
        disable_stability=bool(args.disable_stability),
        stable_stride=int(args.stable_stride),
        stable_max_fail_frac=float(args.stable_max_fail_frac),
        max_trans_p95=float(args.max_trans_p95),
        max_rot_p95=float(args.max_rot_p95),
        max_scale_p95=float(args.max_scale_p95),
        target_fps=int(args.target_fps),
        normalize_xy=bool(args.normalize_xy),
        skip_frame0=bool(args.skip_frame0),
    )

    manifest_path = out_dir / args.manifest_name
    with open(manifest_path, "w", encoding="utf-8") as mf:
        for vp in tqdm(shard_videos, desc=f"Videos shard{shard_index:02d}"):
            label = infer_label(vp)

            try:
                scenes = detect_scenes(vp, threshold=cfg.scene_threshold)
            except Exception as e:
                tqdm.write(f"[WARN] SceneDetect failed on {vp}: {e}")
                continue

            for si, (sf, ef) in enumerate(scenes):
                if (ef - sf) < cfg.min_scene_frames:
                    continue

                ok_people, people_info = scene_passes_people_filter(
                    video_path=vp,
                    start_frame=sf,
                    end_frame=ef,
                    pose_model=pose_model,
                    cfg=cfg,
                )
                if not ok_people:
                    continue

                ok_stable, stable_info = scene_passes_stability_filter(vp, sf, ef, cfg)
                if not ok_stable:
                    continue

                # output file
                scene_name = f"{vp.stem}__scene{si:03d}__f{sf:06d}-{ef:06d}.npz"
                out_path = out_dir / label / scene_name
                ensure_dir(out_path.parent)
                if out_path.exists():
                    continue

                try:
                    kps, bbs, frame_idx, pose_meta = extract_scene_pose(vp, sf, ef, pose_model, cfg, track_iou_thr=float(args.track_iou_thr), track_max_lost=int(args.track_max_lost))
                except Exception as e:
                    tqdm.write(f"[WARN] pose failed {vp} scene {si}: {e}")
                    continue

                rec: Dict = {
                    "video_path": str(vp),
                    "label": label,
                    "scene_index": int(si),
                    "start_frame": int(sf),
                    "end_frame": int(ef),
                    "frame_indices": frame_idx,
                    "people_filter": people_info,
                    "stability": stable_info,
                    "pose_meta": pose_meta,
                    "pose_backend": "ultralytics_yolo_pose",
                    "pose_layout": "coco17",
                    "pose_weights": str(args.pose_weights),
                    "output_npz": str(out_path),
                }

                np.savez_compressed(
                    out_path,
                    keypoints=kps,  # (T,K,17,3)
                    bboxes=bbs,     # (T,K,4)
                    frame_indices=np.array(frame_idx, dtype=np.int32),
                    meta=np.array([json.dumps(rec)], dtype=np.string_),
                )
                mf.write(json.dumps(rec) + "\n")
                mf.flush()

    print("[OK] wrote:", manifest_path)


if __name__ == "__main__":
    main()
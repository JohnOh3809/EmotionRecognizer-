# preprocess_caer_pose.py
"""
End-to-end CAER preprocessing pipeline (MediaPipe or MMPose pose backend):

1) Iterate all CAER videos under --data_root
2) Split each video into scenes using PySceneDetect
3) For each scene:
   - sample a few frames
   - run a person detector (YOLO) to estimate #people + confidence
   - reject scenes outside [min_people, max_people] or below person_conf
   - reject scenes with unstable camera motion (optical-flow + affine motion stats)
4) For scenes that pass filters:
   - run a pose extractor:
       * MediaPipe Pose (crop-per-person via YOLO bboxes), OR
       * MMPose top-down (requires config + checkpoint; uses YOLO bboxes)
   - map the result to OpenPose BODY_25 layout (25 joints) for drop-in compatibility
   - save per-scene .npz with keypoints + metadata
   - write a manifest.jsonl

Output keypoints are (T, K, 25, 3) where last dim is (x, y, score).

Install deps:
  pip install -U numpy opencv-python-headless tqdm scenedetect ultralytics

Pose backend deps:
  # MediaPipe
  pip install mediapipe

  # MMPose (install varies by CUDA/torch; follow official docs)
  # You will also need a top-down pose model config + checkpoint.

Example (MediaPipe):
  python preprocess_caer_pose.py --pose_backend mediapipe --target_fps 10

Example (MMPose):
  python preprocess_caer_pose.py --pose_backend mmpose \
    --mmpose_config /path/to/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
    --mmpose_checkpoint /path/to/hrnet_w32_coco_256x192.pth \
    --mmpose_device cuda
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

from filtering_config import CAER_ROOT, OUTPUT_ROOT, YOLO_WEIGHTS  


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


def infer_split_and_label(video_path: Path) -> Tuple[str, str]:
    parts = [p.lower() for p in video_path.parts]
    split = "unknown"
    for s in ("train", "validation", "val", "test"):
        if s in parts:
            split = "validation" if s == "val" else s
            break
    label = normalize_label(video_path.parent.name)
    return split, label


def scan_videos(data_root: Path, exts: Tuple[str, ...] = (".avi", ".mp4", ".mov", ".mkv")) -> List[Dict]:
    vids: List[Dict] = []
    for p in data_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            split, label = infer_split_and_label(p)
            vids.append({"path": p, "split": split, "label": label})
    return vids


# -----------------------------
# Scene detection (PySceneDetect)
# -----------------------------
def detect_scenes(video_path: Path, threshold: float) -> List[Tuple[int, int]]:
    """
    Returns list of (start_frame, end_frame_exclusive).
    If no scenes detected, returns whole video.
    """
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
# Person detector backend (YOLO)
# -----------------------------
@dataclass
class PersonDet:
    bbox_xyxy: np.ndarray  # (4,) float32, pixels
    conf: float


class PeopleDetector:
    def detect(self, frame_bgr: np.ndarray) -> List[PersonDet]:
        """Return list of person detections in this frame (bbox + conf)."""
        raise NotImplementedError


class YOLOPeopleDetector(PeopleDetector):
    def __init__(self, weights: str, conf: float, imgsz: int = 640):
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "ultralytics not installed. Run: pip install ultralytics\n"
                f"Import error: {e}"
            )
        self.model = YOLO(weights)
        self.conf = float(conf)
        self.imgsz = int(imgsz)

    def detect(self, frame_bgr: np.ndarray) -> List[PersonDet]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=[0],  # person
            verbose=False,
        )
        r0 = results[0]
        if r0.boxes is None or len(r0.boxes) == 0:
            return []
        xyxy = r0.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        confs = r0.boxes.conf.detach().cpu().numpy().astype(np.float32)
        dets: List[PersonDet] = [PersonDet(bbox_xyxy=bb, conf=float(cc)) for bb, cc in zip(xyxy, confs)]
        dets.sort(key=lambda d: d.conf, reverse=True)
        return dets


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


def _estimate_motion_affine(prev_gray: np.ndarray, curr_gray: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
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

    return trans, rot_deg, scale_delta, 1.0


def camera_stability_stats(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    stride: int = 5,
    max_frames: int = 200,
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
            trans, rot_deg, scale_delta, _ = est
            motions.append((trans, rot_deg, scale_delta))

        prev_g = curr_g
        if pairs >= max_frames:
            break

    cap.release()

    if pairs == 0:
        return StabilityStats(0, 1.0, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9)

    fail_frac = fails / float(pairs)

    if len(motions) == 0:
        return StabilityStats(pairs, fail_frac, 1e9, 1e9, 1e9, 1e9, 1e9, 1e9)

    arr = np.array(motions, dtype=np.float32)

    def _med(x: np.ndarray) -> float:
        return float(np.nanmedian(x))

    def _p95(x: np.ndarray) -> float:
        return float(np.nanpercentile(x, 95))

    return StabilityStats(
        n_pairs=pairs,
        fail_frac=float(fail_frac),
        trans_median=_med(arr[:, 0]),
        trans_p95=_p95(arr[:, 0]),
        rot_median=_med(arr[:, 1]),
        rot_p95=_p95(arr[:, 1]),
        scale_median=_med(arr[:, 2]),
        scale_p95=_p95(arr[:, 2]),
    )


# -----------------------------
# Pose extraction backends
#   - Always output OpenPose BODY_25 layout (25 keypoints)
# -----------------------------
BODY25 = 25


def _nan25() -> np.ndarray:
    return np.full((BODY25, 3), np.nan, dtype=np.float32)


def _bbox_expand_and_clip(bbox_xyxy: np.ndarray, w: int, h: int, expand: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy.tolist()]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    bw2 = bw * (1.0 + float(expand))
    bh2 = bh * (1.0 + float(expand))

    nx1 = int(max(0.0, cx - bw2 / 2.0))
    ny1 = int(max(0.0, cy - bh2 / 2.0))
    nx2 = int(min(float(w), cx + bw2 / 2.0))
    ny2 = int(min(float(h), cy + bh2 / 2.0))

    if nx2 <= nx1:
        nx2 = min(w, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(h, ny1 + 1)
    return nx1, ny1, nx2, ny2


def _nanmean_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    stacked = np.stack([a, b], axis=0).astype(np.float32)
    return np.nanmean(stacked, axis=0).astype(np.float32)


def body25_from_mediapipe33(kp33: np.ndarray) -> np.ndarray:
    """
    kp33: (33,3) with (x_px, y_px, score) in full-frame coordinates.
    Returns (25,3) BODY_25 with NaNs for missing.
    """
    out = _nan25()

    MP = {
        "nose": 0,
        "l_eye": 2,
        "r_eye": 5,
        "l_ear": 7,
        "r_ear": 8,
        "l_sh": 11,
        "r_sh": 12,
        "l_el": 13,
        "r_el": 14,
        "l_wr": 15,
        "r_wr": 16,
        "l_hip": 23,
        "r_hip": 24,
        "l_kn": 25,
        "r_kn": 26,
        "l_an": 27,
        "r_an": 28,
        "l_heel": 29,
        "r_heel": 30,
        "l_foot": 31,  # foot index (closest thing to big toe)
        "r_foot": 32,
    }

    out[0] = kp33[MP["nose"]]
    out[1] = _nanmean_rows(kp33[MP["l_sh"]], kp33[MP["r_sh"]])  # neck

    out[2] = kp33[MP["r_sh"]]
    out[3] = kp33[MP["r_el"]]
    out[4] = kp33[MP["r_wr"]]
    out[5] = kp33[MP["l_sh"]]
    out[6] = kp33[MP["l_el"]]
    out[7] = kp33[MP["l_wr"]]

    out[8] = _nanmean_rows(kp33[MP["l_hip"]], kp33[MP["r_hip"]])  # midhip

    out[9] = kp33[MP["r_hip"]]
    out[10] = kp33[MP["r_kn"]]
    out[11] = kp33[MP["r_an"]]
    out[12] = kp33[MP["l_hip"]]
    out[13] = kp33[MP["l_kn"]]
    out[14] = kp33[MP["l_an"]]

    out[15] = kp33[MP["r_eye"]]
    out[16] = kp33[MP["l_eye"]]
    out[17] = kp33[MP["r_ear"]]
    out[18] = kp33[MP["l_ear"]]

    out[19] = kp33[MP["l_foot"]]  # LBigToe (approx)
    # 20 LSmallToe: not available
    out[21] = kp33[MP["l_heel"]]

    out[22] = kp33[MP["r_foot"]]  # RBigToe (approx)
    # 23 RSmallToe: not available
    out[24] = kp33[MP["r_heel"]]

    return out


def body25_from_coco17(kp17: np.ndarray) -> np.ndarray:
    """
    kp17: (17,3) with (x_px, y_px, score).
    COCO order: nose, l_eye, r_eye, l_ear, r_ear, l_sh, r_sh, l_el, r_el, l_wr, r_wr,
                l_hip, r_hip, l_kn, r_kn, l_an, r_an
    Returns BODY_25 (25,3). Feet/toes/heels are NaN because COCO17 doesn't have them.
    """
    out = _nan25()

    out[0] = kp17[0]  # nose

    # neck, midhip
    out[1] = _nanmean_rows(kp17[5], kp17[6])
    out[8] = _nanmean_rows(kp17[11], kp17[12])

    # arms
    out[2] = kp17[6]
    out[3] = kp17[8]
    out[4] = kp17[10]
    out[5] = kp17[5]
    out[6] = kp17[7]
    out[7] = kp17[9]

    # legs
    out[9] = kp17[12]
    out[10] = kp17[14]
    out[11] = kp17[16]
    out[12] = kp17[11]
    out[13] = kp17[13]
    out[14] = kp17[15]

    # face
    out[15] = kp17[2]
    out[16] = kp17[1]
    out[17] = kp17[4]
    out[18] = kp17[3]

    return out


class PoseExtractor:
    backend_name: str = "base"

    def infer_body25(self, frame_bgr: np.ndarray, person_dets: List[PersonDet]) -> np.ndarray:
        """Return (P, 25, 3) aligned with person_dets order."""
        raise NotImplementedError


class MediaPipePoseExtractor(PoseExtractor):
    backend_name = "mediapipe"

    def __init__(
        self,
        static_image_mode: bool = True,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        bbox_expand: float = 0.2,
    ):
        try:
            import mediapipe as mp  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "mediapipe not installed. Run: pip install mediapipe\n"
                f"Import error: {e}"
            )

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=bool(static_image_mode),
            model_complexity=int(model_complexity),
            enable_segmentation=False,
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self.bbox_expand = float(bbox_expand)

    def infer_body25(self, frame_bgr: np.ndarray, person_dets: List[PersonDet]) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if len(person_dets) == 0:
            return np.zeros((0, BODY25, 3), dtype=np.float32)

        out_all: List[np.ndarray] = []
        for det in person_dets:
            x1, y1, x2, y2 = _bbox_expand_and_clip(det.bbox_xyxy, w=w, h=h, expand=self.bbox_expand)
            crop_bgr = frame_bgr[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                out_all.append(_nan25())
                continue

            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            res = self.pose.process(crop_rgb)
            if res.pose_landmarks is None:
                out_all.append(_nan25())
                continue

            ch, cw = crop_bgr.shape[:2]
            kp33 = np.full((33, 3), np.nan, dtype=np.float32)
            for i, lm in enumerate(res.pose_landmarks.landmark):
                kp33[i, 0] = float(lm.x) * float(cw) + float(x1)
                kp33[i, 1] = float(lm.y) * float(ch) + float(y1)
                kp33[i, 2] = float(getattr(lm, "visibility", 0.0))

            out25 = body25_from_mediapipe33(kp33)
            # damp scores by detector confidence
            out25[:, 2] = out25[:, 2] * float(det.conf)
            out_all.append(out25)

        return np.stack(out_all, axis=0).astype(np.float32)


class MMPoseTopDownExtractor(PoseExtractor):
    backend_name = "mmpose"

    def __init__(
        self,
        config_file: str,
        checkpoint_file: str,
        device: str = "cuda",
        bbox_thr: float = 0.0,
    ):
        self.config_file = str(config_file)
        self.checkpoint_file = str(checkpoint_file)
        self.device = str(device)
        self.bbox_thr = float(bbox_thr)

        try:
            from mmpose.apis import init_model  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "mmpose not installed (or missing dependencies). "
                "You usually need torch + mmcv + mmpose.\n"
                f"Import error: {e}"
            )

        self._init_model = init_model

        self._inference = None
        try:
            from mmpose.apis import inference_topdown  # type: ignore

            self._inference = inference_topdown
            self._api = "inference_topdown"
        except Exception:
            try:
                from mmpose.apis import inference_top_down_pose_model  # type: ignore

                self._inference = inference_top_down_pose_model
                self._api = "inference_top_down_pose_model"
            except Exception as e:
                raise RuntimeError(
                    "Could not find an MMPose inference API. Tried inference_topdown and inference_top_down_pose_model.\n"
                    f"Import error: {e}"
                )

        self.model = self._init_model(self.config_file, self.checkpoint_file, device=self.device)

    def infer_body25(self, frame_bgr: np.ndarray, person_dets: List[PersonDet]) -> np.ndarray:
        if len(person_dets) == 0:
            return np.zeros((0, BODY25, 3), dtype=np.float32)

        # Build bbox list, but keep alignment to person_dets.
        keep_mask: List[bool] = []
        bboxes: List[List[float]] = []
        for det in person_dets:
            keep = det.conf >= self.bbox_thr
            keep_mask.append(keep)
            if keep:
                x1, y1, x2, y2 = [float(v) for v in det.bbox_xyxy.tolist()]
                bboxes.append([x1, y1, x2, y2, float(det.conf)])

        if len(bboxes) == 0:
            return np.stack([_nan25() for _ in person_dets], axis=0).astype(np.float32)

        bboxes_arr = np.array(bboxes, dtype=np.float32)

        # ---- Newer MMPose API ----
        if self._api == "inference_topdown":
            results = self._inference(self.model, frame_bgr, bboxes_arr, bbox_format="xyxy")
            try:
                from mmpose.structures import merge_data_samples  # type: ignore

                merged = merge_data_samples(results)
                kpts = np.asarray(merged.pred_instances.keypoints, dtype=np.float32)  # (P,J,2)
                scores = np.asarray(merged.pred_instances.keypoint_scores, dtype=np.float32)  # (P,J)
            except Exception:
                k_list = []
                s_list = []
                for r in results:
                    inst = getattr(r, "pred_instances", None)
                    if inst is None:
                        continue
                    k = np.asarray(inst.keypoints, dtype=np.float32)
                    sc = np.asarray(inst.keypoint_scores, dtype=np.float32)
                    if k.ndim == 3:
                        k = k[0]
                    if sc.ndim == 2:
                        sc = sc[0]
                    k_list.append(k)
                    s_list.append(sc)
                if len(k_list) == 0:
                    return np.stack([_nan25() for _ in person_dets], axis=0).astype(np.float32)
                kpts = np.stack(k_list, axis=0)
                scores = np.stack(s_list, axis=0)

            if kpts.ndim != 3 or scores.ndim != 2:
                return np.stack([_nan25() for _ in person_dets], axis=0).astype(np.float32)

            P, J, _ = kpts.shape
            if J != 17:
                # Unknown keypoint layout; add another mapper here if you use a different dataset.
                return np.stack([_nan25() for _ in person_dets], axis=0).astype(np.float32)

            kpJ3 = np.zeros((P, J, 3), dtype=np.float32)
            kpJ3[:, :, 0:2] = kpts[:, :, 0:2]
            kpJ3[:, :, 2] = scores

            mapped = [body25_from_coco17(kpJ3[i]) for i in range(P)]

        # ---- Older MMPose API ----
        else:
            person_results = [{"bbox": bb[:4], "bbox_score": float(bb[4])} for bb in bboxes_arr]
            pose_results, _ = self._inference(self.model, frame_bgr, person_results, bbox_thr=self.bbox_thr, format="xyxy")

            mapped = []
            for pr in pose_results:
                k_raw = pr.get("keypoints", None)
                if k_raw is None:
                    continue
                k = np.asarray(k_raw, dtype=np.float32)
                if k.ndim != 2 or k.shape[1] < 2:
                    continue
                if k.shape[1] == 2:
                    kk = np.zeros((k.shape[0], 3), dtype=np.float32)
                    kk[:, :2] = k[:, :2]
                    kk[:, 2] = 1.0
                    k = kk
                if k.shape[0] != 17:
                    continue
                mapped.append(body25_from_coco17(k))

            if len(mapped) == 0:
                return np.stack([_nan25() for _ in person_dets], axis=0).astype(np.float32)

        # Align mapped poses back to person_dets using keep_mask
        out_all: List[np.ndarray] = []
        mi = 0
        for keep in keep_mask:
            if not keep:
                out_all.append(_nan25())
                continue
            if mi < len(mapped):
                out_all.append(mapped[mi])
            else:
                out_all.append(_nan25())
            mi += 1

        return np.stack(out_all, axis=0).astype(np.float32)


# -----------------------------
# Core processing
# -----------------------------
@dataclass
class FilterConfig:
    # scene splitting
    scene_threshold: float = 27.0
    min_scene_frames: int = 16

    # person filtering
    min_people: int = 1
    max_people: int = 2
    person_conf: float = 0.40
    people_sample_frames: int = 5

    # camera stability
    stable_stride: int = 5
    stable_max_fail_frac: float = 0.25
    max_trans_p95: float = 6.0
    max_rot_p95: float = 2.0
    max_scale_p95: float = 0.03

    # pose extraction
    target_fps: int = 10
    normalize_xy: bool = False


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


def scene_passes_people_filter(
    cap: cv2.VideoCapture,
    detector: PeopleDetector,
    start_frame: int,
    end_frame: int,
    cfg: FilterConfig,
) -> Tuple[bool, Dict]:
    idxs = sample_indices(start_frame, end_frame, cfg.people_sample_frames)
    counts: List[int] = []
    mean_confs: List[float] = []

    for fi in idxs:
        frame = read_frame_at(cap, fi)
        if frame is None:
            continue
        dets = detector.detect(frame)
        confs = [d.conf for d in dets if d.conf >= cfg.person_conf]
        counts.append(len(confs))
        mean_confs.append(float(np.mean(confs)) if confs else 0.0)

    if len(counts) == 0:
        return False, {"reason": "people_sampling_failed", "counts": [], "mean_confs": []}

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


def scene_passes_stability_filter(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    cfg: FilterConfig,
) -> Tuple[bool, Dict]:
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
        "trans_median": stats.trans_median,
        "trans_p95": stats.trans_p95,
        "rot_median": stats.rot_median,
        "rot_p95": stats.rot_p95,
        "scale_median": stats.scale_median,
        "scale_p95": stats.scale_p95,
    }


def iter_scene_frames(cap: cv2.VideoCapture, start_frame: int, end_frame: int, step: int) -> Iterable[Tuple[int, np.ndarray]]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    idx = int(start_frame)
    while idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start_frame) % step == 0:
            yield idx, frame
        idx += 1


def extract_pose_scene(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    detector: PeopleDetector,
    pose_extractor: PoseExtractor,
    max_people: int,
    person_conf: float,
    target_fps: int,
    normalize_xy: bool,
) -> Tuple[np.ndarray, List[int], Dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-6:
        fps = 30.0

    step = max(1, int(round(fps / float(max(1, target_fps)))))

    keypoints_list: List[np.ndarray] = []
    frame_indices: List[int] = []
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    for fi, frame in iter_scene_frames(cap, start_frame, end_frame, step=step):
        dets = [d for d in detector.detect(frame) if d.conf >= person_conf]
        dets = dets[:max_people]

        if len(dets) == 0:
            out = np.full((max_people, BODY25, 3), np.nan, dtype=np.float32)
        else:
            kp = pose_extractor.infer_body25(frame, dets)  # (P,25,3)
            if kp is None or kp.size == 0:
                out = np.full((max_people, BODY25, 3), np.nan, dtype=np.float32)
            else:
                person_scores = np.nanmean(kp[:, :, 2], axis=1)
                order = np.argsort(person_scores)[::-1]
                kp = kp[order][:max_people]
                if kp.shape[0] < max_people:
                    pad = np.full((max_people - kp.shape[0], BODY25, 3), np.nan, dtype=np.float32)
                    kp = np.concatenate([kp, pad], axis=0)
                out = kp.astype(np.float32)

        if normalize_xy and w > 0 and h > 0:
            out[:, :, 0] = out[:, :, 0] / float(w)
            out[:, :, 1] = out[:, :, 1] / float(h)

        keypoints_list.append(out)
        frame_indices.append(int(fi))

    cap.release()

    arr = np.stack(keypoints_list, axis=0) if keypoints_list else np.zeros((0, max_people, BODY25, 3), dtype=np.float32)

    meta = {
        "fps": fps,
        "frame_w": w,
        "frame_h": h,
        "step": step,
        "target_fps": target_fps,
        "normalize_xy": normalize_xy,
        "pose_backend": pose_extractor.backend_name,
        "body_layout": "openpose_body25",
    }
    return arr, frame_indices, meta


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def process_all(
    data_root: Path,
    out_dir: Path,
    detector: PeopleDetector,
    pose_extractor: PoseExtractor,
    cfg: FilterConfig,
    dry_run: bool = False,
) -> None:
    data_root = data_root.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    ensure_dir(out_dir)

    manifest_path = out_dir / "manifest.jsonl"

    videos = scan_videos(data_root)
    if not videos:
        raise RuntimeError(f"No videos found under: {data_root}")

    with open(manifest_path, "a", encoding="utf-8") as mf:
        for item in tqdm(videos, desc="Videos"):
            vp: Path = item["path"]
            split: str = item["split"]
            label: str = item["label"]

            try:
                scenes = detect_scenes(vp, threshold=cfg.scene_threshold)
            except Exception as e:
                tqdm.write(f"[WARN] SceneDetect failed on {vp}: {e}")
                continue

            cap = cv2.VideoCapture(str(vp))
            if not cap.isOpened():
                cap.release()
                tqdm.write(f"[WARN] Could not open {vp}")
                continue

            for si, (sf, ef) in enumerate(scenes):
                if (ef - sf) < cfg.min_scene_frames:
                    continue

                ok_people, people_info = scene_passes_people_filter(
                    cap=cap,
                    detector=detector,
                    start_frame=sf,
                    end_frame=ef,
                    cfg=cfg,
                )
                if not ok_people:
                    continue

                ok_stable, stable_info = scene_passes_stability_filter(
                    video_path=vp,
                    start_frame=sf,
                    end_frame=ef,
                    cfg=cfg,
                )
                if not ok_stable:
                    continue

                rel = vp.relative_to(data_root)
                base = rel.with_suffix("").as_posix().replace("/", "__")
                scene_name = f"{base}__scene{si:03d}__f{sf:06d}-{ef:06d}.npz"
                out_path = out_dir / split / label / scene_name
                ensure_dir(out_path.parent)

                if out_path.exists():
                    continue

                record: Dict = {
                    "video_path": str(vp),
                    "split": split,
                    "label": label,
                    "scene_index": si,
                    "start_frame": int(sf),
                    "end_frame": int(ef),
                    "people_filter": people_info,
                    "stability": stable_info,
                    "output_npz": str(out_path),
                    "pose_backend": pose_extractor.backend_name,
                }

                if dry_run:
                    mf.write(json.dumps(record) + "\n")
                    mf.flush()
                    continue

                try:
                    kps, frame_indices, pose_meta = extract_pose_scene(
                        video_path=vp,
                        start_frame=sf,
                        end_frame=ef,
                        detector=detector,
                        pose_extractor=pose_extractor,
                        max_people=cfg.max_people,
                        person_conf=cfg.person_conf,
                        target_fps=cfg.target_fps,
                        normalize_xy=cfg.normalize_xy,
                    )
                except Exception as e:
                    tqdm.write(f"[WARN] Pose extraction failed on {vp} scene {si}: {e}")
                    continue

                record["pose_meta"] = pose_meta
                record["frame_indices"] = frame_indices

                np.savez_compressed(
                    out_path,
                    keypoints=kps,
                    frame_indices=np.array(frame_indices, dtype=np.int32),
                    meta=np.array([json.dumps(record)], dtype=np.string_),
                )

                mf.write(json.dumps(record) + "\n")
                mf.flush()

            cap.release()


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--dry_run", action="store_true")

    ap.add_argument("--detector", type=str, default="yolo", choices=["yolo"])
    ap.add_argument("--yolo_imgsz", type=int, default=640)

    ap.add_argument("--pose_backend", type=str, default="mediapipe", choices=["mediapipe", "mmpose"])

    # MediaPipe options
    ap.add_argument(
        "--mp_video_mode",
        action="store_true",
        help="Use MediaPipe Pose tracking (static_image_mode=False). Default is static mode, which is safer when cropping per-person.",
    )
    ap.add_argument("--mp_model_complexity", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--mp_min_det_conf", type=float, default=0.5)
    ap.add_argument("--mp_min_track_conf", type=float, default=0.5)
    ap.add_argument("--mp_bbox_expand", type=float, default=0.2)

    # MMPose options
    ap.add_argument("--mmpose_config", type=str, default="")
    ap.add_argument("--mmpose_checkpoint", type=str, default="")
    ap.add_argument("--mmpose_device", type=str, default="cuda")
    ap.add_argument("--mmpose_bbox_thr", type=float, default=0.0)

    # Filters
    ap.add_argument("--scene_threshold", type=float, default=27.0)
    ap.add_argument("--min_scene_frames", type=int, default=16)

    ap.add_argument("--min_people", type=int, default=1)
    ap.add_argument("--max_people", type=int, default=2)
    ap.add_argument("--person_conf", type=float, default=0.40)
    ap.add_argument("--people_sample_frames", type=int, default=5)

    ap.add_argument("--stable_stride", type=int, default=5)
    ap.add_argument("--stable_max_fail_frac", type=float, default=0.25)
    ap.add_argument("--max_trans_p95", type=float, default=6.0)
    ap.add_argument("--max_rot_p95", type=float, default=2.0)
    ap.add_argument("--max_scale_p95", type=float, default=0.03)

    ap.add_argument("--target_fps", type=int, default=10)
    ap.add_argument("--normalize_xy", action="store_true")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.detector == "yolo":
        detector = YOLOPeopleDetector(
            weights=str(YOLO_WEIGHTS),
            conf=args.person_conf,
            imgsz=args.yolo_imgsz,
        )
    else:
        raise RuntimeError(f"Unknown detector: {args.detector}")

    if args.pose_backend == "mediapipe":
        pose_extractor = MediaPipePoseExtractor(
            static_image_mode=not bool(args.mp_video_mode),
            model_complexity=int(args.mp_model_complexity),
            min_detection_confidence=float(args.mp_min_det_conf),
            min_tracking_confidence=float(args.mp_min_track_conf),
            bbox_expand=float(args.mp_bbox_expand),
        )
    else:
        if not args.mmpose_config or not args.mmpose_checkpoint:
            raise RuntimeError(
                "MMPose backend selected but --mmpose_config and --mmpose_checkpoint were not provided."
            )
        pose_extractor = MMPoseTopDownExtractor(
            config_file=args.mmpose_config,
            checkpoint_file=args.mmpose_checkpoint,
            device=args.mmpose_device,
            bbox_thr=float(args.mmpose_bbox_thr),
        )

    cfg = FilterConfig(
        scene_threshold=args.scene_threshold,
        min_scene_frames=args.min_scene_frames,
        min_people=args.min_people,
        max_people=args.max_people,
        person_conf=args.person_conf,
        people_sample_frames=args.people_sample_frames,
        stable_stride=args.stable_stride,
        stable_max_fail_frac=args.stable_max_fail_frac,
        max_trans_p95=args.max_trans_p95,
        max_rot_p95=args.max_rot_p95,
        max_scale_p95=args.max_scale_p95,
        target_fps=args.target_fps,
        normalize_xy=args.normalize_xy,
    )

    print(CAER_ROOT)
    print(OUTPUT_ROOT)

    process_all(
        data_root=CAER_ROOT,
        out_dir=OUTPUT_ROOT,
        detector=detector,
        pose_extractor=pose_extractor,
        cfg=cfg,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

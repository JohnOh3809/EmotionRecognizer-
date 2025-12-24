#!/usr/bin/env python3
"""
YOLOv11 + DeepFace Live Emotion Recognition Demo

Uses webcam + DeepFace for emotion classification + YOLOv11 for pose keypoints.
DeepFace provides emotion labels, YOLOv11 provides pose data.

Usage:
    python YOLODEMO.py
    python YOLODEMO.py --yolo_model yolo11n-pose.pt
    python YOLODEMO.py --camera 0

Controls:
    q - Quit
    r - Reset pose buffer
    s - Save screenshot
    d - Save data (with DeepFace emotion label + YOLOv11 pose data)
"""

import argparse
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    exit(1)

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    print("WARNING: DeepFace not available. Please install:")
    print("  pip install deepface")
    print("  Note: DeepFace requires TensorFlow, which may not support Python 3.14")
    print("  Consider using Python 3.11 or 3.12 for full DeepFace support")
    DEEPFACE_AVAILABLE = False

# Emotion classes (8 classes including 'other')
# DeepFace uses: angry, disgust, fear, happy, sad, surprise, neutral
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "other"]

# Map DeepFace emotions to our CLASSES (DeepFace doesn't have 'other')
DEEPFACE_TO_CLASSES = {
    "angry": "angry",
    "disgust": "disgust", 
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral"
}

# Emotion colors (BGR)
EMOTION_COLORS = {
    "angry": (0, 0, 255),      # Red
    "disgust": (0, 128, 0),    # Dark Green
    "fear": (128, 0, 128),     # Purple
    "happy": (0, 255, 255),    # Yellow
    "neutral": (128, 128, 128), # Gray
    "sad": (255, 0, 0),        # Blue
    "surprise": (0, 165, 255), # Orange
    "other": (100, 100, 100),  # Dark Gray
}

# COCO-17 keypoint connections for drawing skeleton
COCO_SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # Head
    [5, 6],  # Shoulders
    [5, 7], [7, 9],  # Left arm
    [6, 8], [8, 10],  # Right arm
    [5, 11], [6, 12],  # Torso
    [11, 13], [13, 15],  # Left leg
    [12, 14], [14, 16],  # Right leg
]

# COCO-17 keypoint names
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


class BiLSTMEmotionClassifier(nn.Module):
    """BiLSTM model matching the training architecture."""

    def __init__(self, in_dim: int = 238, hidden: int = 128, num_layers: int = 1,
                 num_classes: int = 7, dropout: float = 0.3):
        super().__init__()

        # Input projection
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False),
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        # Project input
        x = self.proj(x)

        # BiLSTM
        lstm_out, _ = self.lstm(x)

        # Attention pooling
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)

        return self.head(context)


class PreActResidualBlock(nn.Module):
    """Pre-activation residual block."""
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.linear1(torch.relu(self.norm1(x)))
        out = self.dropout(out)
        out = self.linear2(torch.relu(self.norm2(out)))
        return x + out


class AttentionPooling(nn.Module):
    """Attention-weighted pooling over time."""
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x, mask=None):
        # x: (B, T, D)
        scores = self.attn(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)  # (B, T)
        return (x * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)


class DeepResidualPreact(nn.Module):
    """Best performing model: DeepResidual with pre-activation blocks."""
    def __init__(self, input_dim: int = 238, hidden: int = 256,
                 num_blocks: int = 4, dropout: float = 0.3, num_classes: int = 8):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            PreActResidualBlock(hidden, dropout) for _ in range(num_blocks)
        ])

        # Attention pooling
        self.pool = AttentionPooling(hidden)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        # x: (B, T, D)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        return self.classifier(x)


class PoseBuffer:
    """Buffer to accumulate pose sequences for temporal modeling."""

    def __init__(self, seq_len: int = 64, num_people: int = 2):
        self.seq_len = seq_len
        self.num_people = num_people
        self.buffer = deque(maxlen=seq_len)
        self.prev_keypoints = None
        self.prev_time = None

    def reset(self):
        self.buffer.clear()
        self.prev_keypoints = None
        self.prev_time = None

    def add_frame(self, keypoints: np.ndarray, timestamp: float):
        """
        Add a frame's keypoints to the buffer.
        keypoints: (num_people, 17, 3) - x, y, confidence
        """
        # Ensure correct shape
        if keypoints.shape[0] < self.num_people:
            pad = np.zeros((self.num_people - keypoints.shape[0], 17, 3), dtype=np.float32)
            keypoints = np.concatenate([keypoints, pad], axis=0)
        elif keypoints.shape[0] > self.num_people:
            keypoints = keypoints[:self.num_people]

        # Compute velocity
        if self.prev_keypoints is not None and self.prev_time is not None:
            dt = max(timestamp - self.prev_time, 1e-6)
            vel = (keypoints[:, :, :2] - self.prev_keypoints[:, :, :2]) / dt
        else:
            vel = np.zeros((self.num_people, 17, 2), dtype=np.float32)

        # Compute acceleration (from velocity change)
        if len(self.buffer) > 0:
            prev_vel = self.buffer[-1]["vel"]
            dt = max(timestamp - self.prev_time, 1e-6) if self.prev_time else 1.0
            acc = (vel - prev_vel) / dt
        else:
            acc = np.zeros((self.num_people, 17, 2), dtype=np.float32)

        # Store frame
        self.buffer.append({
            "keypoints": keypoints.copy(),
            "vel": vel,
            "acc": acc,
            "timestamp": timestamp
        })

        self.prev_keypoints = keypoints.copy()
        self.prev_time = timestamp

    def get_features(self) -> np.ndarray:
        """Get feature tensor for model input."""
        if len(self.buffer) == 0:
            return np.zeros((self.seq_len, 238), dtype=np.float32)

        # Stack all frames
        frames = []
        for frame in self.buffer:
            kp = frame["keypoints"]  # (P, 17, 3)
            vel = frame["vel"]       # (P, 17, 2)
            acc = frame["acc"]       # (P, 17, 2)

            # Concatenate features: (P, 17, 7)
            feat = np.concatenate([
                kp.reshape(self.num_people, -1),
                vel.reshape(self.num_people, -1),
                acc.reshape(self.num_people, -1)
            ], axis=-1)

            # Flatten people: (P * 17 * 7) = 238
            feat = feat.reshape(-1)
            frames.append(feat)

        frames = np.array(frames, dtype=np.float32)

        # Pad to seq_len if needed
        if len(frames) < self.seq_len:
            pad = np.zeros((self.seq_len - len(frames), 238), dtype=np.float32)
            frames = np.concatenate([frames, pad], axis=0)

        # Replace NaN
        frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)

        return frames

    def is_ready(self) -> bool:
        """Check if buffer has enough frames for prediction."""
        return len(self.buffer) >= 10  # Minimum frames for prediction


class EmotionPredictor:
    """Wrapper for emotion prediction model."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load checkpoint to detect architecture
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get state dict
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Detect model architecture from state dict keys
        if "input_proj.0.weight" in state_dict:
            # DeepResidualPreact model
            print("Detected DeepResidualPreact architecture")
            self.model = DeepResidualPreact(
                input_dim=238,
                hidden=256,
                num_blocks=4,
                dropout=0.0,  # No dropout at inference
                num_classes=len(CLASSES)
            ).to(self.device)
        else:
            # BiLSTM model (default)
            print("Detected BiLSTM architecture")
            self.model = BiLSTMEmotionClassifier(
                in_dim=238,
                hidden=128,
                num_layers=1,
                num_classes=len(CLASSES),
                dropout=0.0  # No dropout at inference
            ).to(self.device)

        # Load weights
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Loaded model from {model_path}")

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> tuple:
        """
        Predict emotion from features.
        Returns: (emotion_label, confidence, all_probs)
        """
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = np.argmax(probs)
        pred_label = CLASSES[pred_idx]
        confidence = probs[pred_idx]

        return pred_label, confidence, probs


class YOLODemo:
    """Main YOLOv11 + DeepFace live demo class."""

    def __init__(self, yolo_model: str = "yolo11n-pose.pt", 
                 video_source: int = 0, device: str = "cuda"):
        self.pose_buffer = PoseBuffer(seq_len=64, num_people=2)

        # Initialize YOLOv11 pose model
        print(f"Loading YOLO model: {yolo_model}")
        self.yolo_model = YOLO(yolo_model)
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"YOLO using device: {self.device}")
        
        # Initialize DeepFace (will download models on first use)
        if not DEEPFACE_AVAILABLE:
            print("ERROR: DeepFace is not available. Cannot run demo.")
            print("Please install DeepFace or use Python 3.11/3.12")
            exit(1)
        print("Initializing DeepFace...")
        print("(First run will download models - this may take a moment)")

        # Video source
        self.video_source = video_source
        self.cap = None

        # DeepFace emotion state
        self.deepface_emotion = "neutral"
        self.deepface_confidence = 0.0
        self.deepface_all_emotions = {}
        self.last_deepface_time = 0
        self.deepface_interval = 0.5  # Run DeepFace every 500ms (it's slower)
        
        # Display state
        self.fps = 0
        self.frame_count = 0
        
        # Data saving - organized by emotion folders
        self.data_save_dir = Path("collected_data")
        self.data_save_dir.mkdir(exist_ok=True)
        # Create emotion-specific folders
        for emotion in CLASSES:
            (self.data_save_dir / emotion).mkdir(exist_ok=True)
        self.save_count = 0
        self.last_save_time = 0
        self.save_cooldown = 2.0  # Minimum seconds between saves

    def normalize_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Normalize keypoints: center at hip, scale by torso length.
        This matches the training data preprocessing.
        keypoints: (P, 17, 3) - x, y, confidence
        Returns: (P, 17, 3) - normalized
        """
        normalized = keypoints.copy()
        
        for person_idx in range(keypoints.shape[0]):
            kp = keypoints[person_idx]  # (17, 3)
            xy = kp[:, :2]  # (17, 2)
            conf = kp[:, 2]  # (17,)
            
            # Find valid joints (confidence > 0.1)
            valid = conf > 0.1
            
            if valid.sum() < 3:  # Not enough valid joints
                continue
            
            # COCO-17 keypoint indices
            L_HIP, R_HIP = 11, 12
            L_SHOULDER, R_SHOULDER = 5, 6
            
            # Center at mid-hip if both hips are valid, else use mean of valid joints
            if valid[L_HIP] and valid[R_HIP]:
                center = 0.5 * (xy[L_HIP] + xy[R_HIP])
            else:
                center = np.mean(xy[valid], axis=0)
            
            # Scale by torso length (shoulder to hip distance)
            if valid[L_SHOULDER] and valid[R_SHOULDER] and valid[L_HIP] and valid[R_HIP]:
                mid_shoulder = 0.5 * (xy[L_SHOULDER] + xy[R_SHOULDER])
                mid_hip = 0.5 * (xy[L_HIP] + xy[R_HIP])
                torso_len = np.linalg.norm(mid_shoulder - mid_hip)
            else:
                # Fallback: use RMS radius of valid joints
                centered_xy = xy[valid] - center
                rms = np.sqrt(np.mean(np.sum(centered_xy ** 2, axis=1)))
                torso_len = max(rms, 1.0)
            
            # Ensure minimum scale
            torso_len = max(torso_len, 1.0)
            
            # Normalize: center and scale
            normalized[person_idx, :, :2] = (xy - center) / torso_len
            normalized[person_idx, :, 2] = conf  # Keep confidence unchanged
        
        return normalized

    def extract_keypoints(self, results, frame_shape) -> np.ndarray:
        """Extract COCO-format keypoints from YOLO results."""
        h, w = frame_shape[:2]

        if results.keypoints is None or len(results.keypoints) == 0:
            return np.zeros((1, 17, 3), dtype=np.float32)

        # Get keypoints from YOLO results
        kp_data = results.keypoints.data  # (P, 17, 3) or (P, 17, 2)
        
        if kp_data is None:
            # Fallback to xy and conf
            kp_xy = results.keypoints.xy  # (P, 17, 2)
            kp_conf = getattr(results.keypoints, "conf", None)  # (P, 17)
            
            if kp_conf is None:
                kp_conf = np.ones((kp_xy.shape[0], kp_xy.shape[1]), dtype=np.float32)
            
            kp_xy = kp_xy.cpu().numpy().astype(np.float32)
            kp_conf = kp_conf.cpu().numpy().astype(np.float32) if hasattr(kp_conf, 'cpu') else kp_conf
            
            # Stack to (P, 17, 3)
            keypoints = np.concatenate([kp_xy, kp_conf[..., None]], axis=2).astype(np.float32)
        else:
            keypoints = kp_data.cpu().numpy().astype(np.float32)
            
            # Ensure shape is (P, 17, 3)
            if keypoints.shape[2] == 2:
                # Add confidence if missing
                conf = np.ones((keypoints.shape[0], keypoints.shape[1], 1), dtype=np.float32)
                keypoints = np.concatenate([keypoints, conf], axis=2)

        # Limit to max 2 people
        if keypoints.shape[0] > 2:
            keypoints = keypoints[:2]
        elif keypoints.shape[0] == 0:
            keypoints = np.zeros((1, 17, 3), dtype=np.float32)

        # Normalize keypoints to match training data format
        keypoints = self.normalize_keypoints(keypoints)

        return keypoints

    def extract_raw_keypoints(self, results, frame_shape) -> np.ndarray:
        """Extract raw keypoints (pixel coordinates) for drawing."""
        if results.keypoints is None or len(results.keypoints) == 0:
            return np.zeros((1, 17, 3), dtype=np.float32)

        # Get keypoints from YOLO results
        kp_data = results.keypoints.data
        
        if kp_data is None:
            kp_xy = results.keypoints.xy.cpu().numpy()
            kp_conf = getattr(results.keypoints, "conf", None)
            if kp_conf is None:
                kp_conf = np.ones((kp_xy.shape[0], kp_xy.shape[1]), dtype=np.float32)
            else:
                kp_conf = kp_conf.cpu().numpy()
            kp_data = np.concatenate([kp_xy, kp_conf[..., None]], axis=2)
        else:
            kp_data = kp_data.cpu().numpy()
            if kp_data.shape[2] == 2:
                conf = np.ones((kp_data.shape[0], kp_data.shape[1], 1), dtype=np.float32)
                kp_data = np.concatenate([kp_data, conf], axis=2)

        # Limit to max 2 people
        if kp_data.shape[0] > 2:
            kp_data = kp_data[:2]
        elif kp_data.shape[0] == 0:
            kp_data = np.zeros((1, 17, 3), dtype=np.float32)

        return kp_data.astype(np.float32)

    def draw_skeleton(self, frame, results, raw_keypoints=None):
        """Draw pose skeleton on frame using YOLO keypoints."""
        if raw_keypoints is not None:
            kp_data = raw_keypoints
        elif results.keypoints is None or len(results.keypoints) == 0:
            return
        else:
            # Get keypoints from results if raw_keypoints not provided
            kp_data = self.extract_raw_keypoints(results, frame.shape)

        # Draw skeleton for each person
        for person_idx in range(min(kp_data.shape[0], 2)):  # Max 2 people
            keypoints = kp_data[person_idx]  # (17, 3)
            
            # Draw keypoints
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.3:  # Only draw if confidence > threshold
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), 1)

            # Draw skeleton connections
            for connection in COCO_SKELETON:
                pt1_idx, pt2_idx = connection
                if pt1_idx < len(keypoints) and pt2_idx < len(keypoints):
                    x1, y1, c1 = keypoints[pt1_idx]
                    x2, y2, c2 = keypoints[pt2_idx]
                    if c1 > 0.3 and c2 > 0.3:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (0, 255, 0), 2)

    def analyze_emotion_deepface(self, frame):
        """Analyze emotion using DeepFace."""
        try:
            # DeepFace expects RGB format
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Analyze emotion
            result = DeepFace.analyze(
                img_path=rgb_frame,
                actions=['emotion'],
                enforce_detection=False,  # Don't fail if no face detected
                silent=True
            )
            
            # Handle both single dict and list of dicts
            if isinstance(result, list):
                result = result[0]
            
            # Extract emotion predictions
            if 'emotion' in result:
                emotions = result['emotion']
                # Get dominant emotion
                dominant_emotion = max(emotions.items(), key=lambda x: x[1])
                emotion_name = dominant_emotion[0].lower()
                emotion_conf = dominant_emotion[1] / 100.0  # Convert to 0-1 range
                
                # Map to our CLASSES
                label = DEEPFACE_TO_CLASSES.get(emotion_name, "neutral")
                
                return label, emotion_conf, emotions
            else:
                return "neutral", 0.0, {}
        except Exception as e:
            # If DeepFace fails (no face, etc.), return neutral
            return "neutral", 0.0, {}

    def save_pose_data(self, emotion_label: str = None):
        """Save current pose buffer data to npz file with DeepFace emotion label."""
        if len(self.pose_buffer.buffer) < 10:
            print("Not enough data in buffer to save (need at least 10 frames)")
            return False
        
        current_time = time.time()
        if current_time - self.last_save_time < self.save_cooldown:
            print(f"Please wait {self.save_cooldown - (current_time - self.last_save_time):.1f}s before saving again")
            return False
        
        # Get data from buffer
        T = len(self.pose_buffer.buffer)
        num_people = self.pose_buffer.num_people
        
        # Extract keypoints, velocities, and accelerations
        keypoints = np.zeros((T, num_people, 17, 3), dtype=np.float32)
        velocities = np.zeros((T, num_people, 17, 2), dtype=np.float32)
        accelerations = np.zeros((T, num_people, 17, 2), dtype=np.float32)
        timestamps = np.zeros(T, dtype=np.float32)
        
        # Extract timestamps from buffer
        if T > 0 and "timestamp" in self.pose_buffer.buffer[0]:
            # Use actual timestamps from buffer
            start_time = self.pose_buffer.buffer[0]["timestamp"]
            for i, frame_data in enumerate(self.pose_buffer.buffer):
                keypoints[i] = frame_data["keypoints"]
                velocities[i] = frame_data["vel"]
                accelerations[i] = frame_data["acc"]
                if "timestamp" in frame_data:
                    timestamps[i] = frame_data["timestamp"] - start_time
                else:
                    timestamps[i] = i * 0.033  # Fallback to estimated
        else:
            # Fallback: use relative timestamps
            for i, frame_data in enumerate(self.pose_buffer.buffer):
                keypoints[i] = frame_data["keypoints"]
                velocities[i] = frame_data["vel"]
                accelerations[i] = frame_data["acc"]
                timestamps[i] = i * 0.033  # ~30fps default
        
        # Use DeepFace emotion as label (or provided label)
        label = emotion_label if emotion_label else self.deepface_emotion
        
        # Ensure label is valid (fallback to 'other' if not in CLASSES)
        if label not in CLASSES:
            label = "other"
        
        # Create emotion-specific folder path
        emotion_folder = self.data_save_dir / label
        emotion_folder.mkdir(exist_ok=True)
        
        # Create metadata
        meta = {
            "label": label,
            "timestamp": datetime.now().isoformat(),
            "num_frames": T,
            "num_people": num_people,
            "emotion": self.deepface_emotion,  # DeepFace emotion
            "emotion_confidence": float(self.deepface_confidence),
            "deepface_emotions": self.deepface_all_emotions,  # All DeepFace emotion scores
            "source": "live_camera",
            "pose_backend": "yolov11",
            "emotion_backend": "deepface",
            "pose_layout": "coco17"
        }
        
        # Save to npz file in emotion-specific folder
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pose_{timestamp_str}_{self.save_count:04d}.npz"
        filepath = emotion_folder / filename
        
        np.savez_compressed(
            filepath,
            keypoints=keypoints,
            vel=velocities,
            acc=accelerations,
            time=timestamps,
            meta=np.array([json.dumps(meta)], dtype=np.bytes_)
        )
        
        self.save_count += 1
        self.last_save_time = current_time
        print(f"Saved pose data to {filepath} (label: {label}, frames: {T})")
        return True

    def draw_ui(self, frame, current_time=None):
        """Draw UI overlay on frame."""
        h, w = frame.shape[:2]
        
        if current_time is None:
            current_time = time.time()

        # Semi-transparent background for UI
        overlay = frame.copy()

        # DeepFace Emotion display box (left side)
        color = EMOTION_COLORS.get(self.deepface_emotion, (128, 128, 128))
        cv2.rectangle(overlay, (10, 10), (350, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # DeepFace Emotion text
        cv2.putText(frame, "DeepFace Emotion:",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"{self.deepface_emotion.upper()}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Confidence: {self.deepface_confidence*100:.1f}%",
                    (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}",
                    (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # YOLOv11 Pose info (right side)
        cv2.putText(frame, "YOLOv11: Pose Detection",
                    (w - 250, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # DeepFace emotion probability bars (if available)
        if self.deepface_all_emotions:
            bar_width = 150
            bar_height = 15
            bar_x = w - bar_width - 20
            bar_y_start = 50
            
            # Sort emotions by confidence
            sorted_emotions = sorted(
                self.deepface_all_emotions.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:7]  # Top 7 emotions
            
            for i, (emotion_name, prob) in enumerate(sorted_emotions):
                emotion_lower = emotion_name.lower()
                mapped_emotion = DEEPFACE_TO_CLASSES.get(emotion_lower, "other")
                y = bar_y_start + i * (bar_height + 3)
                color = EMOTION_COLORS.get(mapped_emotion, (128, 128, 128))
                
                # Normalize probability (DeepFace gives 0-100)
                prob_normalized = prob / 100.0 if prob > 1.0 else prob

                # Background bar
                cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + bar_height), (50, 50, 50), -1)

                # Probability bar
                fill_width = int(bar_width * prob_normalized)
                cv2.rectangle(frame, (bar_x, y), (bar_x + fill_width, y + bar_height), color, -1)

                # Label
                cv2.putText(frame, f"{mapped_emotion[:4]}: {prob:.0f}%",
                            (bar_x - 90, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Buffer status
        buffer_pct = min(100, len(self.pose_buffer.buffer) / self.pose_buffer.seq_len * 100)
        cv2.putText(frame, f"Buffer: {buffer_pct:.0f}%",
                    (w - 100, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Controls hint
        cv2.putText(frame, "q:Quit  r:Reset  s:Screenshot  d:Save Data",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Save status
        if current_time - self.last_save_time < 1.0:
            cv2.putText(frame, "Data Saved!",
                        (w - 150, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def run(self):
        """Main demo loop."""
        # Open video source
        if isinstance(self.video_source, str):
            self.cap = cv2.VideoCapture(self.video_source)
        else:
            self.cap = cv2.VideoCapture(self.video_source)

        if not self.cap.isOpened():
            print(f"Error: Could not open video source {self.video_source}")
            return

        print("\nYOLOv11 + DeepFace Live Demo Started!")
        print("DeepFace: Emotion classification from face")
        print("YOLOv11: Pose keypoint detection")
        print("Data: DeepFace emotion labels + YOLOv11 pose data")
        print("Controls: q=Quit, r=Reset buffer, s=Screenshot, d=Save Data")
        print(f"Data will be saved to: {self.data_save_dir.absolute()}")
        print(f"Data organized in emotion folders: {', '.join(CLASSES)}")
        print("-" * 40)

        fps_time = time.time()
        fps_count = 0

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    if isinstance(self.video_source, str):
                        # Video file ended, loop
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("Error: Could not read frame")
                        break

                # Flip for mirror effect (webcam only)
                if not isinstance(self.video_source, str):
                    frame = cv2.flip(frame, 1)

                current_time = time.time()

                # Process pose with YOLO
                results = self.yolo_model.predict(
                    source=frame,
                    conf=0.25,
                    device=self.device,
                    verbose=False,
                    imgsz=640
                )[0]

                # Extract keypoints (normalized for model)
                keypoints = self.extract_keypoints(results, frame.shape)
                
                # Extract raw keypoints for drawing (pixel coordinates)
                raw_keypoints = self.extract_raw_keypoints(results, frame.shape)
                
                # Buffer normalized keypoints (YOLOv11 pose data)
                self.pose_buffer.add_frame(keypoints, current_time)

                # Run DeepFace emotion analysis periodically (it's slower, so less frequent)
                if (current_time - self.last_deepface_time) > self.deepface_interval:
                    emotion, conf, all_emotions = self.analyze_emotion_deepface(frame)
                    self.deepface_emotion = emotion
                    self.deepface_confidence = conf
                    self.deepface_all_emotions = all_emotions
                    self.last_deepface_time = current_time

                # Draw skeleton using raw pixel coordinates
                self.draw_skeleton(frame, results, raw_keypoints)

                # Draw UI
                self.draw_ui(frame, current_time)

                # Calculate FPS
                fps_count += 1
                if current_time - fps_time >= 1.0:
                    self.fps = fps_count / (current_time - fps_time)
                    fps_count = 0
                    fps_time = current_time

                # Resize frame to make window bigger (scale up by 1.5x)
                display_frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

                # Display
                cv2.imshow("YOLOv11 Emotion Recognition Demo", display_frame)

                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.pose_buffer.reset()
                    print("Buffer reset")
                elif key == ord('s'):
                    filename = f"screenshot_{int(time.time())}.png"
                    cv2.imwrite(filename, frame)
                    print(f"Saved screenshot: {filename}")
                elif key == ord('d'):
                    # Save pose data
                    self.save_pose_data()

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("\nDemo ended.")


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 + DeepFace Live Emotion Recognition Demo")
    parser.add_argument("--yolo_model", type=str, default="yolo11n-pose.pt",
                        help="YOLO pose model (yolo11n-pose.pt, yolo11s-pose.pt, etc.)")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file (default: webcam)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for inference (cuda or cpu)")
    args = parser.parse_args()

    # Determine video source
    video_source = args.video if args.video else args.camera

    # Run demo
    demo = YOLODemo(args.yolo_model, video_source, args.device)
    demo.run()


if __name__ == "__main__":
    main()


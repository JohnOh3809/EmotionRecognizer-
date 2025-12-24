#!/usr/bin/env python3
"""
Live Emotion Recognition Demo

Uses webcam + MediaPipe pose estimation + BiLSTM model to predict emotions in real-time.

Usage:
    python3 live_demo.py
    python3 live_demo.py --model_path runs/bilstm_20251223_220438_small_1layer_h128_mixup/best.pt
    python3 live_demo.py --video path/to/video.mp4

Controls:
    q - Quit
    r - Reset pose buffer
    s - Save screenshot
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

try:
    import mediapipe as mp
except ImportError:
    print("Please install mediapipe: pip install mediapipe")
    exit(1)

# Emotion classes (8 classes including 'other')
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise", "other"]

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

# MediaPipe to COCO keypoint mapping (17 keypoints)
# MediaPipe has 33 landmarks, we map to COCO 17
MP_TO_COCO = {
    0: 0,    # nose
    2: 1,    # left_eye (inner) -> left_eye
    5: 2,    # right_eye (inner) -> right_eye
    7: 3,    # left_ear
    8: 4,    # right_ear
    11: 5,   # left_shoulder
    12: 6,   # right_shoulder
    13: 7,   # left_elbow
    14: 8,   # right_elbow
    15: 9,   # left_wrist
    16: 10,  # right_wrist
    23: 11,  # left_hip
    24: 12,  # right_hip
    25: 13,  # left_knee
    26: 14,  # right_knee
    27: 15,  # left_ankle
    28: 16,  # right_ankle
}


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
            "acc": acc
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

        # Load model
        self.model = BiLSTMEmotionClassifier(
            in_dim=238,
            hidden=128,
            num_layers=1,
            num_classes=len(CLASSES),
            dropout=0.0  # No dropout at inference
        ).to(self.device)

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

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


class LiveDemo:
    """Main live demo class."""

    def __init__(self, model_path: str, video_source: int = 0):
        self.predictor = EmotionPredictor(model_path)
        self.pose_buffer = PoseBuffer(seq_len=64, num_people=2)

        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Video source
        self.video_source = video_source
        self.cap = None

        # State
        self.current_emotion = "neutral"
        self.current_confidence = 0.0
        self.current_probs = np.zeros(len(CLASSES))
        self.fps = 0
        self.frame_count = 0
        self.last_prediction_time = 0
        self.prediction_interval = 0.1  # Predict every 100ms

    def extract_keypoints(self, results, frame_shape) -> np.ndarray:
        """Extract COCO-format keypoints from MediaPipe results."""
        h, w = frame_shape[:2]

        if not results.pose_landmarks:
            return np.zeros((1, 17, 3), dtype=np.float32)

        landmarks = results.pose_landmarks.landmark
        keypoints = np.zeros((1, 17, 3), dtype=np.float32)

        for mp_idx, coco_idx in MP_TO_COCO.items():
            if mp_idx < len(landmarks):
                lm = landmarks[mp_idx]
                keypoints[0, coco_idx, 0] = lm.x * w
                keypoints[0, coco_idx, 1] = lm.y * h
                keypoints[0, coco_idx, 2] = lm.visibility

        return keypoints

    def draw_skeleton(self, frame, results):
        """Draw pose skeleton on frame."""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

    def draw_ui(self, frame):
        """Draw UI overlay on frame."""
        h, w = frame.shape[:2]

        # Semi-transparent background for UI
        overlay = frame.copy()

        # Emotion display box
        color = EMOTION_COLORS.get(self.current_emotion, (128, 128, 128))
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Emotion text
        cv2.putText(frame, f"Emotion: {self.current_emotion.upper()}",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Confidence: {self.current_confidence*100:.1f}%",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Probability bars
        bar_width = 150
        bar_height = 20
        bar_x = w - bar_width - 20
        bar_y_start = 20

        for i, (cls, prob) in enumerate(zip(CLASSES, self.current_probs)):
            y = bar_y_start + i * (bar_height + 5)
            color = EMOTION_COLORS.get(cls, (128, 128, 128))

            # Background bar
            cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + bar_height), (50, 50, 50), -1)

            # Probability bar
            fill_width = int(bar_width * prob)
            cv2.rectangle(frame, (bar_x, y), (bar_x + fill_width, y + bar_height), color, -1)

            # Label
            cv2.putText(frame, f"{cls[:3]}: {prob*100:.0f}%",
                        (bar_x - 80, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Buffer status
        buffer_pct = min(100, len(self.pose_buffer.buffer) / self.pose_buffer.seq_len * 100)
        cv2.putText(frame, f"Buffer: {buffer_pct:.0f}%",
                    (w - 100, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Controls hint
        cv2.putText(frame, "q:Quit  r:Reset  s:Screenshot",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

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

        print("\nLive Demo Started!")
        print("Controls: q=Quit, r=Reset buffer, s=Screenshot")
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

                # Process pose
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)

                # Extract and buffer keypoints
                keypoints = self.extract_keypoints(results, frame.shape)
                self.pose_buffer.add_frame(keypoints, current_time)

                # Run prediction periodically
                if (current_time - self.last_prediction_time) > self.prediction_interval:
                    if self.pose_buffer.is_ready():
                        features = self.pose_buffer.get_features()
                        emotion, conf, probs = self.predictor.predict(features)
                        self.current_emotion = emotion
                        self.current_confidence = conf
                        self.current_probs = probs
                    self.last_prediction_time = current_time

                # Draw skeleton
                self.draw_skeleton(frame, results)

                # Draw UI
                self.draw_ui(frame)

                # Calculate FPS
                fps_count += 1
                if current_time - fps_time >= 1.0:
                    self.fps = fps_count / (current_time - fps_time)
                    fps_count = 0
                    fps_time = current_time

                # Display
                cv2.imshow("Emotion Recognition Demo", frame)

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

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.pose.close()
            print("\nDemo ended.")


def main():
    parser = argparse.ArgumentParser(description="Live Emotion Recognition Demo")
    parser.add_argument("--model_path", type=str,
                        default="runs/bilstm_20251223_220438_small_1layer_h128_mixup/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to video file (default: webcam)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index (default: 0)")
    args = parser.parse_args()

    # Determine video source
    video_source = args.video if args.video else args.camera

    # Check model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("\nAvailable models:")
        for p in Path("runs").glob("**/best.pt"):
            print(f"  {p}")
        return

    # Run demo
    demo = LiveDemo(str(model_path), video_source)
    demo.run()


if __name__ == "__main__":
    main()

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from spoofgate_cdcn_model import load_checkpoint


MODEL_PATH = Path("models/spoofgate_cdcn/spoofgate_cdcn.pt")
TARGET_FRAMES = 7
MIN_FRAMES = 5
MAX_FRAMES = 10


@dataclass
class LivenessResult:
    decision: str
    reason: str
    median_score: float
    min_score: float
    max_score: float
    frames: int
    last_score: float


class CDCNLivenessGate:
    def __init__(self, threshold=None, margin=0.04, clear_spoof_score=0.20, cpu=True):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing CDCN model: {MODEL_PATH}. Run train_cdcn_spoofgate.py first."
            )

        self.device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
        self.model, trained_threshold, self.image_size, self.checkpoint = load_checkpoint(
            MODEL_PATH,
            self.device,
        )
        self.threshold = float(trained_threshold if threshold is None else threshold)
        self.margin = float(margin)
        self.clear_spoof_score = float(clear_spoof_score)
        self.scores = []
        self.last_box = None

    def reset(self):
        self.scores = []
        self.last_box = None

    def _box_shifted(self, box):
        if self.last_box is None:
            return False

        x, y, w, h = box
        lx, ly, lw, lh = self.last_box
        center = np.array([x + w / 2.0, y + h / 2.0])
        last_center = np.array([lx + lw / 2.0, ly + lh / 2.0])
        center_shift = float(np.linalg.norm(center - last_center))
        size_shift = abs(w * h - lw * lh) / max(1.0, float(lw * lh))
        return center_shift > max(45.0, 0.35 * max(w, h)) or size_shift > 0.55

    def _crop_face(self, frame, box, pad_ratio=0.16):
        x, y, w, h = box
        size = max(w, h) * (1.0 + 2.0 * pad_ratio)
        cx = x + w / 2.0
        cy = y + h / 2.0
        x1 = max(0, int(cx - size / 2.0))
        y1 = max(0, int(cy - size / 2.0))
        x2 = min(frame.shape[1], int(cx + size / 2.0))
        y2 = min(frame.shape[0], int(cy + size / 2.0))
        return frame[y1:y2, x1:x2]

    def _quality_ok(self, face_crop):
        if face_crop.size == 0:
            return False

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return 35 <= brightness <= 230 and sharpness >= 16

    def _preprocess(self, face_crop):
        resized = cv2.resize(
            face_crop,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - 0.5) / 0.5
        chw = np.transpose(rgb, (2, 0, 1))[None, ...]
        return torch.from_numpy(chw).float().to(self.device)

    @torch.no_grad()
    def _score(self, face_crop):
        tensor = self._preprocess(face_crop)
        logit, _live_map = self.model(tensor)
        return float(torch.sigmoid(logit.view(-1))[0].detach().cpu().item())

    def _result(self, decision, reason):
        if self.scores:
            median_score = float(np.median(self.scores))
            min_score = float(np.min(self.scores))
            max_score = float(np.max(self.scores))
            last_score = float(self.scores[-1])
        else:
            median_score = min_score = max_score = last_score = 0.0

        return LivenessResult(
            decision=decision,
            reason=reason,
            median_score=median_score,
            min_score=min_score,
            max_score=max_score,
            frames=len(self.scores),
            last_score=last_score,
        )

    def update(self, frame, box):
        if self._box_shifted(box):
            self.reset()

        self.last_box = tuple(box)
        face_crop = self._crop_face(frame, box)

        if not self._quality_ok(face_crop):
            return self._result("PENDING", "LIVENESS_WAITING_FOR_CLEAR_FACE")

        score = self._score(face_crop)
        self.scores.append(score)
        self.scores = self.scores[-MAX_FRAMES:]

        if len(self.scores) < MIN_FRAMES:
            return self._result("PENDING", "LIVENESS_COLLECTING")

        median_score = float(np.median(self.scores))
        live_frames = sum(score >= self.threshold for score in self.scores)
        strong_live = sum(score >= self.threshold + self.margin for score in self.scores)
        clear_spoof_frames = sum(score <= self.clear_spoof_score for score in self.scores)

        if median_score <= self.clear_spoof_score and clear_spoof_frames >= MIN_FRAMES:
            return self._result("BLOCK", "CDCN_CLEAR_SPOOF")

        if median_score >= self.threshold and live_frames >= MIN_FRAMES and strong_live >= 2:
            return self._result("ALLOW", "CDCN_LIVE_FACE")

        if len(self.scores) < MAX_FRAMES:
            return self._result("PENDING", "LIVENESS_RECHECKING")

        if median_score >= self.threshold - self.margin and max(self.scores) >= self.threshold + self.margin:
            return self._result("ALLOW", "CDCN_BORDERLINE_LIVE_FACE")

        if len(self.scores) >= MAX_FRAMES:
            return self._result("BLOCK", "CDCN_UNCERTAIN")

        return self._result("PENDING", "LIVENESS_COLLECTING")

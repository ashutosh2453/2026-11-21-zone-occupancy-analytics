"""Person detection with a COCO-pretrained torchvision model (BSD-3 licensed code and weights)."""

from __future__ import annotations

from typing import Protocol

import numpy as np

PERSON = 1  # COCO category id for "person" in torchvision detection models


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (boxes (N,4) x1y1x2y2 in pixels, scores (N,))."""


class TorchvisionPersonDetector:
    """Faster R-CNN with a MobileNetV3 backbone: a good accuracy/speed balance on CPU.

    Weights (~75 MB) are downloaded once by torchvision into the torch hub cache.
    """

    def __init__(self, score_threshold: float = 0.6, min_box_area: float = 400.0, device: str = "cpu") -> None:
        import torch
        from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn

        self.torch = torch
        self.device = torch.device(device)
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        self.model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, box_score_thresh=score_threshold).eval().to(self.device)
        self.score_threshold = score_threshold
        self.min_box_area = min_box_area

    def detect(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        tensor = self.torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
        with self.torch.inference_mode():
            out = self.model([tensor])[0]
        keep = (out["labels"] == PERSON) & (out["scores"] >= self.score_threshold)
        boxes = out["boxes"][keep].cpu().numpy()
        scores = out["scores"][keep].cpu().numpy()
        if len(boxes):
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            boxes, scores = boxes[area >= self.min_box_area], scores[area >= self.min_box_area]
        return boxes.reshape(-1, 4), scores


class ScriptedDetector:
    """Replays precomputed detections per frame; used in tests and for offline re-analysis."""

    def __init__(self, per_frame: list[list[tuple[float, float, float, float]]]) -> None:
        self.per_frame = per_frame
        self.i = 0

    def detect(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        boxes = self.per_frame[self.i] if self.i < len(self.per_frame) else []
        self.i += 1
        arr = np.array(boxes, dtype=float).reshape(-1, 4)
        return arr, np.full(len(arr), 0.99)

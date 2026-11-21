"""Multi-object tracker in the style of SORT (Bewley et al., 2016).

Each track carries a constant-velocity Kalman filter over [cx, cy, s, r] (centre,
area, aspect ratio). Every frame: predict all tracks, match detections to predictions
with the Hungarian algorithm on IoU, update matched tracks, start new tracks for
unmatched detections and drop tracks that have been missing too long. A track is only
reported once it has been seen `min_hits` times, which suppresses one-frame false
positives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4) in x1,y1,x2,y2 format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def box_to_z(box: np.ndarray) -> np.ndarray:
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] + w / 2, box[1] + h / 2, w * h, w / max(h, 1e-6)], dtype=float)


def x_to_box(x: np.ndarray) -> np.ndarray:
    s, r = max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class KalmanBoxFilter:
    """Constant-velocity Kalman filter; state = [cx, cy, s, r, vcx, vcy, vs]."""

    def __init__(self, box: np.ndarray) -> None:
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)
        self.R = np.diag([1.0, 1.0, 10.0, 0.01])
        self.Q = np.diag([1.0, 1.0, 1.0, 1e-4, 0.5, 0.5, 1e-2])
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e3, 1e3, 1e3])
        self.x = np.zeros(7)
        self.x[:4] = box_to_z(box)

    def predict(self) -> np.ndarray:
        if self.x[2] + self.x[6] <= 0:  # area must stay positive
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return x_to_box(self.x)

    def update(self, box: np.ndarray) -> None:
        y = box_to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    @property
    def box(self) -> np.ndarray:
        return x_to_box(self.x)


@dataclass
class Track:
    id: int
    kf: KalmanBoxFilter
    hits: int = 1
    misses: int = 0
    age: int = 1
    score: float = 0.0
    history: list[tuple[int, float, float]] = field(default_factory=list)  # (frame, cx, cy) of observed positions

    @property
    def box(self) -> np.ndarray:
        return self.kf.box

    @property
    def centroid(self) -> tuple[float, float]:
        b = self.box
        return float((b[0] + b[2]) / 2), float(b[3])  # bottom-centre: where the person stands


@dataclass
class TrackOutput:
    id: int
    box: tuple[float, float, float, float]
    score: float
    foot: tuple[float, float]


class Tracker:
    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 15, min_hits: int = 3) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame = 0

    def update(self, boxes: np.ndarray, scores: np.ndarray | None = None) -> list[TrackOutput]:
        """Advance one frame with detections (N,4). Returns confirmed tracks seen this frame."""
        self.frame += 1
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        scores = np.ones(len(boxes)) if scores is None else np.asarray(scores, dtype=float)
        predicted = np.array([t.kf.predict() for t in self.tracks]).reshape(-1, 4)
        for t in self.tracks:
            t.age += 1

        matches: list[tuple[int, int]] = []
        unmatched_dets = set(range(len(boxes)))
        if len(predicted) and len(boxes):
            iou = iou_matrix(predicted, boxes)
            rows, cols = linear_sum_assignment(-iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_threshold:
                    matches.append((r, c))
                    unmatched_dets.discard(c)

        matched_tracks = set()
        for ti, di in matches:
            t = self.tracks[ti]
            t.kf.update(boxes[di])
            t.hits += 1
            t.misses = 0
            t.score = float(scores[di])
            cx, foot_y = t.centroid
            t.history.append((self.frame, cx, foot_y))
            matched_tracks.add(ti)
        for ti, t in enumerate(self.tracks):
            if ti not in matched_tracks:
                t.misses += 1
        for di in sorted(unmatched_dets):
            t = Track(self._next_id, KalmanBoxFilter(boxes[di]), score=float(scores[di]))
            cx, foot_y = t.centroid
            t.history.append((self.frame, cx, foot_y))
            self.tracks.append(t)
            self._next_id += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        return [TrackOutput(t.id, tuple(float(v) for v in t.box), t.score, t.centroid)
                for t in self.tracks if t.misses == 0 and t.hits >= self.min_hits]

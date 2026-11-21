"""Video pipeline: detect -> track -> analyse -> annotate (optionally anonymised)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .analytics import Analytics, CountLine, Zone
from .detector import Detector
from .tracker import Tracker

MAX_FRAMES = 20_000


class PipelineError(ValueError):
    pass


@dataclass
class SceneConfig:
    zones: list[Zone]
    lines: list[CountLine]

    @classmethod
    def from_dict(cls, data: dict) -> "SceneConfig":
        try:
            zones = [Zone(z["name"], [tuple(map(float, p)) for p in z["polygon"]], z.get("max_occupancy")) for z in data.get("zones", [])]
            lines = [CountLine(l["name"], tuple(map(float, l["a"])), tuple(map(float, l["b"]))) for l in data.get("lines", [])]
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineError(f"invalid scene config: {exc}") from exc
        names = [z.name for z in zones] + [l.name for l in lines]
        if len(names) != len(set(names)):
            raise PipelineError("zone and line names must be unique")
        return cls(zones, lines)


def anonymise(frame: np.ndarray, box) -> None:
    """Pixelate the top third of a person box (head region) in place."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y2 = min(h, y1 + max(1, (y2 - y1) // 3))
    y1 = max(0, y1)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    small = cv2.resize(roi, (max(1, roi.shape[1] // 12), max(1, roi.shape[0] // 12)), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)


def colour(tid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(tid)
    return tuple(int(c) for c in rng.integers(64, 255, 3))


def annotate(frame: np.ndarray, tracks, config: SceneConfig, occupancy: dict, analytics: Analytics, privacy: bool) -> None:
    for z in config.zones:
        pts = np.array(z.polygon, dtype=np.int32)
        over = z.max_occupancy is not None and occupancy.get(z.name, 0) > z.max_occupancy
        cv2.polylines(frame, [pts], True, (0, 0, 255) if over else (0, 200, 255), 2)
        label = f"{z.name}: {occupancy.get(z.name, 0)}" + (f"/{z.max_occupancy}" if z.max_occupancy is not None else "")
        cv2.putText(frame, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    for l in config.lines:
        a, b = tuple(map(int, l.a)), tuple(map(int, l.b))
        cv2.line(frame, a, b, (255, 0, 255), 2)
        c = analytics.line_counts[l.name]
        cv2.putText(frame, f"{l.name} in {c['in']} / out {c['out']}", a, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    for t in tracks:
        if privacy:
            anonymise(frame, t.box)
        x1, y1, x2, y2 = (int(v) for v in t.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour(t.id), 2)
        cv2.putText(frame, f"#{t.id}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour(t.id), 2)


def process_video(src: str | Path, detector: Detector, config: SceneConfig, out_video: str | Path | None = None,
                  privacy: bool = True, detect_every: int = 1, max_frames: int = MAX_FRAMES) -> dict:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise PipelineError("could not open video (unsupported or corrupt file)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        raise PipelineError("video has no frames")
    tracker = Tracker(max_misses=max(3, int(fps // 2)), min_hits=2)
    analytics = Analytics(config.zones, config.lines, fps, (w, h))
    writer = None
    if out_video:
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    started = time.perf_counter()
    i = 0
    boxes, scores = np.zeros((0, 4)), np.zeros(0)
    try:
        while i < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if i % detect_every == 0:
                boxes, scores = detector.detect(frame)
            tracks = tracker.update(boxes, scores)
            occupancy = analytics.update(i, [(t.id, t.foot) for t in tracks])
            if writer is not None:
                annotate(frame, tracks, config, occupancy, analytics, privacy)
                writer.write(frame)
            i += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if i == 0:
        raise PipelineError("video has no readable frames")
    report = analytics.report()
    report["video"] = {"width": w, "height": h, "fps": round(fps, 2), "frames_processed": i,
                       "processing_fps": round(i / max(time.perf_counter() - started, 1e-6), 1)}
    report["heatmap"] = {"cell_px": analytics.cell, "grid": analytics.heatmap.tolist()}
    return report


def save_heatmap_png(report: dict, path: str | Path, background: np.ndarray | None = None) -> None:
    grid = np.array(report["heatmap"]["grid"], dtype=np.float32)
    w, h = report["video"]["width"], report["video"]["height"]
    norm = (255 * grid / grid.max()).astype(np.uint8) if grid.max() > 0 else grid.astype(np.uint8)
    heat = cv2.applyColorMap(cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR), cv2.COLORMAP_JET)
    img = cv2.addWeighted(background, 0.5, heat, 0.5, 0) if background is not None else heat
    cv2.imwrite(str(path), img)


def write_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")

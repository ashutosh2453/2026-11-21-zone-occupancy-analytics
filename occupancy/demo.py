"""Synthetic demo footage: people walking across a scene, with known ground truth.

The "person" sprite is the public-domain NASA astronaut portrait bundled with
scikit-image, so demos and end-to-end tests need no third-party footage. Ground truth
(how many people cross the line in each direction, who stands in the zone) is known
exactly, which makes the end-to-end test a real accuracy check of detector + tracker +
analytics together.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 640, 360, 15

DEMO_SCENE = {
    "zones": [{"name": "checkout", "polygon": [[400, 200], [630, 200], [630, 355], [400, 355]], "max_occupancy": 1}],
    "lines": [{"name": "entrance", "a": [320, 10], "b": [320, 350]}],
}


def _sprite(height: int = 170) -> np.ndarray:
    from skimage import data

    img = data.astronaut()[:, 90:420]  # crop around the person
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    scale = height / bgr.shape[0]
    return cv2.resize(bgr, (int(bgr.shape[1] * scale), height), interpolation=cv2.INTER_AREA)


def _background() -> np.ndarray:
    bg = np.zeros((H, W, 3), dtype=np.uint8)
    bg[:] = (70, 75, 80)
    for x in range(0, W, 40):  # floor tiles
        cv2.line(bg, (x, 0), (x, H), (80, 85, 90), 1)
    for y in range(0, H, 40):
        cv2.line(bg, (0, y), (W, y), (80, 85, 90), 1)
    return bg


def make_demo_video(path: str | Path, seconds: float = 6.0) -> dict:
    """Two people: A walks left -> right (crosses 'entrance' in), B walks right -> left (out) after A.
    A finishes standing inside the 'checkout' zone. Returns the ground truth."""
    sprite = _sprite()
    sh, sw = sprite.shape[:2]
    bg = _background()
    frames = int(seconds * FPS)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    half = frames // 2
    for f in range(frames):
        frame = bg.copy()
        # Person A: x from 20 to 470 during the first half, then stands still (inside checkout zone)
        ax = int(20 + (470 - 20) * min(f / half, 1.0))
        frame[180:180 + sh, ax:ax + sw] = sprite
        # Person B: enters at the second half, walks from right edge to the left
        if f >= half:
            t = (f - half) / max(frames - half - 1, 1)
            bx = int(W - sw - 10 - (W - sw - 30) * t)
            y0 = 10
            frame[y0:y0 + sh, bx:bx + sw] = cv2.flip(sprite, 1)
        writer.write(frame)
    writer.release()
    return {"fps": FPS, "frames": frames, "entrance": {"in": 1, "out": 1}, "checkout_visitors": 1, "unique_people": 2}

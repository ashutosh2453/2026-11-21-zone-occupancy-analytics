"""Command line.

    occupancy demo out/                                   # make a synthetic video with ground truth and analyse it
    occupancy analyze video.mp4 --scene scene.json --out out/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .demo import DEMO_SCENE, make_demo_video
from .detector import TorchvisionPersonDetector
from .pipeline import PipelineError, SceneConfig, process_video, save_heatmap_png, write_report


def _run(video: Path, scene: dict, out: Path, privacy: bool, detect_every: int, threshold: float) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    report = process_video(video, TorchvisionPersonDetector(score_threshold=threshold), SceneConfig.from_dict(scene),
                           out / "annotated.mp4", privacy=privacy, detect_every=detect_every)
    cap = cv2.VideoCapture(str(out / "annotated.mp4"))
    ok, first = cap.read()
    cap.release()
    save_heatmap_png(report, out / "heatmap.png", first if ok else None)
    write_report(report, out / "report.json")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="occupancy")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="generate synthetic footage with known ground truth and analyse it")
    d.add_argument("out", type=Path)
    a = sub.add_parser("analyze", help="analyse a video file")
    a.add_argument("video", type=Path)
    a.add_argument("--scene", type=Path, required=True, help="JSON with zones and lines")
    a.add_argument("--out", type=Path, required=True)
    for s in (d, a):
        s.add_argument("--no-privacy", action="store_true", help="do not pixelate heads in the annotated video")
        s.add_argument("--detect-every", type=int, default=1)
        s.add_argument("--threshold", type=float, default=0.6)
    args = p.parse_args(argv)
    try:
        if args.cmd == "demo":
            args.out.mkdir(parents=True, exist_ok=True)
            truth = make_demo_video(args.out / "demo.mp4")
            report = _run(args.out / "demo.mp4", DEMO_SCENE, args.out, not args.no_privacy, args.detect_every, args.threshold)
            print(json.dumps({"ground_truth": truth, "measured": {
                "entrance": report["lines"]["entrance"], "checkout_visitors": report["zones"]["checkout"]["visitors"],
                "unique_people": report["unique_people"], "processing_fps": report["video"]["processing_fps"]}}, indent=2))
        else:
            scene = json.loads(args.scene.read_text(encoding="utf-8"))
            report = _run(args.video, scene, args.out, not args.no_privacy, args.detect_every, args.threshold)
            print(json.dumps({k: report[k] for k in ("duration_s", "unique_people", "zones", "lines", "alerts")}, indent=2))
    except (PipelineError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

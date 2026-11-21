"""FastAPI service: upload a video and a scene config, get analytics + annotated video + heatmap."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from . import __version__
from .detector import Detector, TorchvisionPersonDetector
from .pipeline import PipelineError, SceneConfig, process_video, save_heatmap_png, write_report

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 100 * 1024 * 1024
RESULT_ID = r"^[0-9a-f]{32}$"


def create_app(detector: Detector | None = None, results_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Zone Occupancy Analytics", version=__version__)
    results = Path(results_dir or os.getenv("RESULTS_DIR", Path(tempfile.gettempdir()) / "occupancy-results"))
    results.mkdir(parents=True, exist_ok=True)
    state: dict = {"detector": detector}

    def get_detector() -> Detector:
        if state["detector"] is None:
            state["detector"] = TorchvisionPersonDetector(score_threshold=float(os.getenv("SCORE_THRESHOLD", "0.6")))
        return state["detector"]

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.post("/analyze")
    async def analyze(video: UploadFile = File(...), scene: str = Form(...), privacy: bool = Form(True),
                      detect_every: int = Form(1)):
        try:
            config = SceneConfig.from_dict(json.loads(scene))
        except json.JSONDecodeError as exc:
            raise HTTPException(422, f"scene is not valid JSON: {exc}") from None
        except PipelineError as exc:
            raise HTTPException(422, str(exc)) from None
        if not 1 <= detect_every <= 10:
            raise HTTPException(422, "detect_every must be between 1 and 10")
        job = uuid.uuid4().hex
        job_dir = results / job
        job_dir.mkdir()
        src = job_dir / ("input" + Path(video.filename or "v.mp4").suffix.lower()[:8])
        size = 0
        with src.open("wb") as out:
            while chunk := await video.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    shutil.rmtree(job_dir, ignore_errors=True)
                    raise HTTPException(413, "video exceeds 100 MB")
                out.write(chunk)
        try:
            report = process_video(src, get_detector(), config, job_dir / "annotated.mp4", privacy=privacy, detect_every=detect_every)
        except PipelineError as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(422, str(exc)) from None
        finally:
            src.unlink(missing_ok=True)  # never keep raw footage
        cap = cv2.VideoCapture(str(job_dir / "annotated.mp4"))
        ok, first = cap.read()
        cap.release()
        save_heatmap_png(report, job_dir / "heatmap.png", first if ok else None)
        write_report(report, job_dir / "report.json")
        report["result_id"] = job
        report["links"] = {"video": f"/results/{job}/annotated.mp4", "heatmap": f"/results/{job}/heatmap.png",
                           "report": f"/results/{job}/report.json"}
        return report

    @app.get("/results/{job}/{name}")
    def result_file(job: str, name: str):
        import re
        if not re.fullmatch(RESULT_ID, job) or name not in ("annotated.mp4", "heatmap.png", "report.json"):
            raise HTTPException(404, "not found")
        path = results / job / name
        if not path.exists():
            raise HTTPException(404, "not found")
        return FileResponse(path, filename=name)

    return app


app = create_app()

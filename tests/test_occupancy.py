import json
import os

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from occupancy.analytics import Analytics, CountLine, Zone, point_in_polygon, segments_intersect, side
from occupancy.api import create_app
from occupancy.demo import DEMO_SCENE, make_demo_video
from occupancy.detector import ScriptedDetector
from occupancy.pipeline import PipelineError, SceneConfig, anonymise, process_video
from occupancy.tracker import KalmanBoxFilter, Tracker, iou_matrix


# ---------------------------------------------------------------- geometry

def test_point_in_polygon():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_polygon((5, 5), sq) and point_in_polygon((10, 5), sq)  # edge counts as inside
    assert not point_in_polygon((11, 5), sq) and not point_in_polygon((-1, -1), sq)
    concave = [(0, 0), (10, 0), (10, 10), (5, 4), (0, 10)]
    assert point_in_polygon((2, 3), concave) and not point_in_polygon((5, 8), concave)


def test_segment_intersection_and_side():
    a, b = (5, 0), (5, 10)
    assert segments_intersect((0, 5), (10, 5), a, b)
    assert not segments_intersect((0, 5), (4, 5), a, b)
    assert not segments_intersect((0, 11), (10, 11), a, b)  # passes beyond the end of the line
    assert side((0, 5), a, b) < 0 < side((10, 5), a, b) or side((0, 5), a, b) > 0 > side((10, 5), a, b)


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)


# ---------------------------------------------------------------- tracker

def walk(start, end, frames, size=(40, 80)):
    xs = np.linspace(start[0], end[0], frames)
    ys = np.linspace(start[1], end[1], frames)
    return [(x, y, x + size[0], y + size[1]) for x, y in zip(xs, ys)]


def test_kalman_predicts_constant_velocity():
    kf = KalmanBoxFilter(np.array([0, 0, 10, 20.0]))
    for i in range(1, 15):
        kf.predict()
        kf.update(np.array([5 * i, 0, 5 * i + 10, 20.0]))
    pred = kf.predict()
    assert abs((pred[0] + pred[2]) / 2 - (5 * 15 + 5)) < 2.0


def test_tracker_keeps_ids_through_short_occlusion_and_separates_people():
    a = walk((0, 100), (400, 100), 40)
    b = walk((400, 300), (0, 300), 40)
    tracker = Tracker(min_hits=2, max_misses=5)
    seen = {}
    for f in range(40):
        dets = [a[f]] + ([] if 18 <= f < 22 else [b[f]])  # B is occluded for 4 frames
        for t in tracker.update(np.array(dets)):
            seen.setdefault(t.id, []).append(f)
    assert len(seen) == 2, f"expected exactly two identities, got {sorted(seen)}"
    assert all(len(v) > 30 for v in seen.values())


def test_tracker_requires_min_hits_and_drops_lost_tracks():
    tracker = Tracker(min_hits=3, max_misses=2)
    assert tracker.update(np.array([[0, 0, 10, 10]])) == []  # tentative
    tracker.update(np.array([[1, 0, 11, 10]]))
    assert [t.id for t in tracker.update(np.array([[2, 0, 12, 10]]))] == [1]
    for _ in range(3):
        tracker.update(np.zeros((0, 4)))
    assert tracker.tracks == []


# ---------------------------------------------------------------- analytics

def test_line_counts_directions_and_debounces_jitter():
    line = CountLine("door", (100, 0), (100, 200))
    an = Analytics([], [line], fps=10, frame_size=(200, 200))
    # track 1 walks left->right; track 2 right->left; track 3 jitters across the line within 0.3 s
    paths = {1: [(80, 100), (95, 100), (105, 100), (120, 100)], 2: [(130, 50), (110, 50), (90, 50), (70, 50)],
             3: [(98, 150), (102, 150), (98, 150), (102, 150)]}
    for f in range(4):
        an.update(f, [(tid, p[f]) for tid, p in paths.items()])
    counts = an.report()["lines"]["door"]
    assert counts["in"] + counts["out"] == 3  # 1, 2 and only the first crossing of 3
    directions = {c["track"]: c["direction"] for c in an.crossings}
    assert directions[1] != directions[2]


def test_zone_occupancy_dwell_and_capacity_alerts():
    zone = Zone("queue", [(0, 0), (100, 0), (100, 100), (0, 100)], max_occupancy=1)
    an = Analytics([zone], [], fps=2, frame_size=(200, 200))
    frames = [[(1, (10, 10))], [(1, (10, 10)), (2, (50, 50))], [(1, (10, 10)), (2, (50, 50))], [(2, (150, 150))],
              [(1, (10, 10)), (3, (20, 20))]]
    for f, tracks in enumerate(frames):
        an.update(f, tracks)
    z = an.report()["zones"]["queue"]
    assert z["peak_occupancy"] == 2 and z["visitors"] == 3
    assert z["max_dwell_s"] == 2.0  # track 1 inside for 4 frames at 2 fps
    assert z["seconds_over_capacity"] == 1.5
    assert [a["time_s"] for a in an.alerts] == [0.5, 2.0]  # two separate over-capacity episodes


def test_invalid_configs():
    with pytest.raises(ValueError):
        Analytics([Zone("x", [(0, 0), (1, 1)])], [], 10, (10, 10))
    with pytest.raises(PipelineError):
        SceneConfig.from_dict({"zones": [{"name": "a"}]})
    with pytest.raises(PipelineError):
        SceneConfig.from_dict({"zones": [{"name": "a", "polygon": [[0, 0], [1, 0], [1, 1]]}],
                               "lines": [{"name": "a", "a": [0, 0], "b": [1, 1]}]})


def test_anonymise_pixelates_only_the_head_region():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (120, 120, 3), dtype=np.uint8)
    before = frame.copy()
    anonymise(frame, (20, 0, 80, 90))
    head, body = frame[0:30, 20:80], frame[40:90, 20:80]
    assert not np.array_equal(head, before[0:30, 20:80])
    assert np.array_equal(body, before[40:90, 20:80])


# ---------------------------------------------------------------- pipeline with scripted detections

def blank_video(path, frames=30, size=(320, 240), fps=10):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for _ in range(frames):
        w.write(np.zeros((size[1], size[0], 3), np.uint8))
    w.release()


def test_pipeline_end_to_end_with_scripted_detector(tmp_path):
    video = tmp_path / "v.mp4"
    blank_video(video)
    boxes = walk((10, 100), (260, 100), 30, size=(40, 80))
    det = ScriptedDetector([[b] for b in boxes])
    scene = SceneConfig.from_dict({"zones": [{"name": "right", "polygon": [[200, 0], [320, 0], [320, 240], [200, 240]]}],
                                   "lines": [{"name": "mid", "a": [160, 0], "b": [160, 240]}]})
    report = process_video(video, det, scene, tmp_path / "out.mp4")
    assert report["unique_people"] == 1
    assert sum(report["lines"]["mid"].values()) == 1
    assert report["zones"]["right"]["visitors"] == 1
    assert (tmp_path / "out.mp4").stat().st_size > 0
    with pytest.raises(PipelineError):
        process_video(tmp_path / "missing.mp4", det, scene)


def test_api_with_scripted_detector(tmp_path):
    video = tmp_path / "v.mp4"
    blank_video(video, frames=20)
    det = ScriptedDetector([[b] for b in walk((10, 100), (260, 100), 20)])
    with TestClient(create_app(detector=det, results_dir=tmp_path / "results")) as c:
        scene = json.dumps({"zones": [], "lines": [{"name": "mid", "a": [160, 0], "b": [160, 240]}]})
        r = c.post("/analyze", files={"video": ("v.mp4", video.read_bytes(), "video/mp4")}, data={"scene": scene})
        assert r.status_code == 200, r.text
        body = r.json()
        assert sum(body["lines"]["mid"].values()) == 1
        assert c.get(body["links"]["heatmap"]).status_code == 200
        assert c.get(f"/results/{body['result_id']}/../../etc/passwd").status_code == 404
        assert c.get("/results/notanid/report.json").status_code == 404
        assert c.post("/analyze", files={"video": ("v.mp4", b"x", "video/mp4")}, data={"scene": "{bad"}).status_code == 422
        assert c.post("/analyze", files={"video": ("v.mp4", b"not a video", "video/mp4")}, data={"scene": scene}).status_code == 422
        assert not list((tmp_path / "results").glob("*/input*"))  # raw uploads are deleted


# ---------------------------------------------------------------- real detector on synthetic footage with ground truth

@pytest.mark.skipif(os.getenv("SKIP_MODEL_TESTS") == "1", reason="model download disabled")
def test_real_detector_matches_demo_ground_truth(tmp_path):
    from occupancy.detector import TorchvisionPersonDetector

    truth = make_demo_video(tmp_path / "demo.mp4")
    report = process_video(tmp_path / "demo.mp4", TorchvisionPersonDetector(), SceneConfig.from_dict(DEMO_SCENE),
                           detect_every=1)
    assert report["lines"]["entrance"] == truth["entrance"]
    assert report["zones"]["checkout"]["visitors"] == truth["checkout_visitors"]
    assert report["unique_people"] == truth["unique_people"]

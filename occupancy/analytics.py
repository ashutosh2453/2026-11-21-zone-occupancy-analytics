"""Zone occupancy, dwell time and directional line-crossing counts from tracks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Point = tuple[float, float]


@dataclass
class Zone:
    name: str
    polygon: list[Point]  # pixel coordinates, at least 3 points
    max_occupancy: int | None = None  # raise an alert above this


@dataclass
class CountLine:
    name: str
    a: Point
    b: Point  # crossing from the left of a->b to the right counts as "in"


def point_in_polygon(p: Point, polygon: list[Point]) -> bool:
    """Ray casting; points on an edge count as inside."""
    x, y = p
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        # on-edge check
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(cross) < 1e-9 and min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9:
            return True
        if (y1 > y) != (y2 > y):
            x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_at_y:
                inside = not inside
    return inside


def side(p: Point, a: Point, b: Point) -> float:
    """Sign of the cross product: >0 left of a->b, <0 right, 0 on the line."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def segments_intersect(p1: Point, p2: Point, a: Point, b: Point) -> bool:
    d1, d2 = side(p1, a, b), side(p2, a, b)
    d3, d4 = side(a, p1, p2), side(b, p1, p2)
    return d1 * d2 < 0 and d3 * d4 < 0


@dataclass
class ZoneStats:
    peak: int = 0
    occupancy_series: list[int] = field(default_factory=list)
    dwell_frames: dict[int, int] = field(default_factory=dict)  # track id -> frames inside
    over_capacity_frames: int = 0
    over_now: bool = False


class Analytics:
    """Feeds on per-frame track positions (foot points) and accumulates statistics."""

    def __init__(self, zones: list[Zone], lines: list[CountLine], fps: float, frame_size: tuple[int, int],
                 heatmap_cell: int = 16) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        for z in zones:
            if len(z.polygon) < 3:
                raise ValueError(f"zone '{z.name}' needs at least 3 points")
        self.zones, self.lines, self.fps = zones, lines, fps
        self.zone_stats = {z.name: ZoneStats() for z in zones}
        self.line_counts = {l.name: {"in": 0, "out": 0} for l in lines}
        self.crossings: list[dict] = []
        self._last_pos: dict[int, tuple[int, Point]] = {}
        self._counted: dict[tuple[str, int], float] = {}  # (line, track) -> last crossing time, debounces jitter
        self.unique_ids: set[int] = set()
        self.frames = 0
        w, h = frame_size
        self.cell = heatmap_cell
        self.heatmap = np.zeros((max(1, h // heatmap_cell + 1), max(1, w // heatmap_cell + 1)), dtype=np.int32)
        self.alerts: list[dict] = []

    def update(self, frame_index: int, tracks: list[tuple[int, Point]]) -> dict:
        """tracks: (track_id, foot_point). Returns this frame's occupancy per zone."""
        self.frames += 1
        occupancy = {}
        for z in self.zones:
            inside = [tid for tid, p in tracks if point_in_polygon(p, z.polygon)]
            st = self.zone_stats[z.name]
            st.occupancy_series.append(len(inside))
            st.peak = max(st.peak, len(inside))
            for tid in inside:
                st.dwell_frames[tid] = st.dwell_frames.get(tid, 0) + 1
            over = z.max_occupancy is not None and len(inside) > z.max_occupancy
            if over:
                st.over_capacity_frames += 1
                if not st.over_now:  # one alert per episode, raised when the limit is first exceeded
                    self.alerts.append({"type": "over_capacity", "zone": z.name, "time_s": round(frame_index / self.fps, 2),
                                        "occupancy": len(inside), "limit": z.max_occupancy})
            st.over_now = over
            occupancy[z.name] = len(inside)

        t_now = frame_index / self.fps
        for tid, p in tracks:
            self.unique_ids.add(tid)
            cy, cx = int(p[1] // self.cell), int(p[0] // self.cell)
            if 0 <= cy < self.heatmap.shape[0] and 0 <= cx < self.heatmap.shape[1]:
                self.heatmap[cy, cx] += 1
            prev = self._last_pos.get(tid)
            if prev is not None:
                _, q = prev
                for line in self.lines:
                    if segments_intersect(q, p, line.a, line.b):
                        key = (line.name, tid)
                        if t_now - self._counted.get(key, -1e9) < 1.0:  # ignore jitter back-and-forth within 1 s
                            continue
                        self._counted[key] = t_now
                        direction = "in" if side(q, line.a, line.b) > 0 else "out"
                        self.line_counts[line.name][direction] += 1
                        self.crossings.append({"line": line.name, "track": tid, "direction": direction, "time_s": round(t_now, 2)})
            self._last_pos[tid] = (frame_index, p)
        return occupancy

    def report(self) -> dict:
        zones = {}
        for name, st in self.zone_stats.items():
            dwell_s = [f / self.fps for f in st.dwell_frames.values()]
            zones[name] = {
                "peak_occupancy": st.peak,
                "average_occupancy": round(float(np.mean(st.occupancy_series)) if st.occupancy_series else 0.0, 2),
                "visitors": len(st.dwell_frames),
                "average_dwell_s": round(float(np.mean(dwell_s)), 2) if dwell_s else 0.0,
                "max_dwell_s": round(float(np.max(dwell_s)), 2) if dwell_s else 0.0,
                "seconds_over_capacity": round(st.over_capacity_frames / self.fps, 2),
            }
        return {
            "frames": self.frames,
            "duration_s": round(self.frames / self.fps, 2),
            "unique_people": len(self.unique_ids),
            "zones": zones,
            "lines": self.line_counts,
            "crossings": self.crossings,
            "alerts": self.alerts,
        }

import numpy as np
from shapely.geometry import Point, Polygon, box


class ROI:
    def __init__(self):
        self.points: list[tuple[int, int]] = []
        self._polygon: Polygon | None = None

    @property
    def valid(self) -> bool:
        return self._polygon is not None and self._polygon.is_valid

    @property
    def area(self) -> float:
        return self._polygon.area if self.valid else 0.0

    def set_points(self, points: list[tuple[int, int]]):
        self.points = list(points)
        if len(self.points) >= 3:
            self._polygon = Polygon(self.points)
        else:
            self._polygon = None

    def add_point(self, x: int, y: int):
        self.points.append((x, y))
        if len(self.points) >= 3:
            self._polygon = Polygon(self.points)

    def clear(self):
        self.points = []
        self._polygon = None

    def contains(self, x: int, y: int) -> bool:
        if not self.valid:
            return False
        return self._polygon.contains(Point(x, y))

    def bbox_overlap(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Fraction of bbox area that intersects the ROI (0.0–1.0)."""
        if not self.valid:
            return 0.0
        bbox_poly = box(x1, y1, x2, y2)
        bbox_area = bbox_poly.area
        if bbox_area == 0:
            return 0.0
        return self._polygon.intersection(bbox_poly).area / bbox_area

    def polygon_array(self) -> np.ndarray | None:
        if not self.points:
            return None
        return np.array(self.points, dtype=np.int32)

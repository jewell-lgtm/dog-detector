import numpy as np
from shapely.geometry import Point, Polygon


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

    def polygon_array(self) -> np.ndarray | None:
        if not self.points:
            return None
        return np.array(self.points, dtype=np.int32)

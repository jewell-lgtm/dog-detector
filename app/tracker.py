import time
from dataclasses import dataclass, field

from app.detector import Detection
from app.roi import ROI


def _test_roi():
    """Return a valid ROI for doctests.

    >>> _test_roi().valid
    True
    """
    roi = ROI()
    roi.set_points([(0, 0), (200, 0), (200, 200), (0, 200)])
    return roi


def _test_det(track_id=1, center=(50, 50)):
    """Return a Detection fully inside the test ROI."""
    return Detection(bbox=(25, 25, 75, 75), center=center, confidence=0.9, track_id=track_id)


@dataclass
class DogTrack:
    """Per-dog tracking state with hysteresis counters."""
    track_id: int
    in_roi_count: int = 0      # consecutive frames detected in ROI
    absent_count: int = 0      # consecutive frames not detected (or not in ROI)
    confirmed: bool = False    # has met enter threshold
    last_seen: float = 0.0


@dataclass
class TrackerState:
    dog_inside: bool = False
    last_change_time: float = 0.0
    enter_count: int = 0
    leave_count: int = 0
    tracks: dict[int, DogTrack] = field(default_factory=dict)


class Tracker:
    def __init__(self, enter_frames: int = 3, leave_frames: int = 5, min_overlap: float = 0.5):
        self.state = TrackerState()
        self.enter_frames = enter_frames
        self.leave_frames = leave_frames
        self.min_overlap = min_overlap

    def update(self, detections: list[Detection], roi: ROI) -> tuple[bool, bool]:
        """Returns (entered, left) booleans.

        Invalid ROI returns immediately:

        >>> Tracker().update([], ROI())
        (False, False)

        Enter hysteresis — need enter_frames consecutive frames in ROI:

        >>> t = Tracker(enter_frames=3, leave_frames=5)
        >>> roi = _test_roi()
        >>> t.update([_test_det()], roi)
        (False, False)
        >>> t.update([_test_det()], roi)
        (False, False)
        >>> t.update([_test_det()], roi)
        (True, False)

        Brief flash — disappears before threshold, no events:

        >>> t = Tracker(enter_frames=3, leave_frames=5)
        >>> roi = _test_roi()
        >>> t.update([_test_det()], roi)
        (False, False)
        >>> t.update([_test_det()], roi)
        (False, False)
        >>> t.update([], roi)
        (False, False)

        Leave hysteresis — must be absent leave_frames to trigger leave:

        >>> t = Tracker(enter_frames=2, leave_frames=3)
        >>> roi = _test_roi()
        >>> t.update([_test_det()], roi)
        (False, False)
        >>> t.update([_test_det()], roi)
        (True, False)
        >>> t.update([], roi)
        (False, False)
        >>> t.update([], roi)
        (False, False)
        >>> t.update([], roi)
        (False, True)
        """
        if not roi.valid:
            return False, False

        # tag detections with in_roi via bbox overlap, collect IDs seen in ROI
        ids_in_roi: set[int] = set()
        for d in detections:
            d.in_roi = roi.bbox_overlap(*d.bbox) >= self.min_overlap
            if d.in_roi and d.track_id is not None:
                ids_in_roi.add(d.track_id)

        now = time.time()

        # update existing tracks + create new ones
        seen_ids: set[int] = set()
        for d in detections:
            if d.track_id is None:
                continue
            seen_ids.add(d.track_id)
            track = self.state.tracks.get(d.track_id)
            if track is None:
                track = DogTrack(track_id=d.track_id)
                self.state.tracks[d.track_id] = track
            track.last_seen = now

            if d.track_id in ids_in_roi:
                track.in_roi_count += 1
                track.absent_count = 0
            else:
                track.absent_count += 1
                track.in_roi_count = 0

        # increment absent count for tracks not seen this frame
        for tid, track in self.state.tracks.items():
            if tid not in seen_ids:
                track.absent_count += 1
                track.in_roi_count = 0

        # apply hysteresis thresholds
        was_inside = self.state.dog_inside
        any_confirmed = False

        for track in list(self.state.tracks.values()):
            if not track.confirmed and track.in_roi_count >= self.enter_frames:
                track.confirmed = True
            elif track.confirmed and track.absent_count >= self.leave_frames:
                track.confirmed = False

            if track.confirmed:
                any_confirmed = True

        # prune stale tracks (absent for 2x leave_frames)
        stale_threshold = self.leave_frames * 2
        stale_ids = [tid for tid, t in self.state.tracks.items()
                     if t.absent_count >= stale_threshold and not t.confirmed]
        for tid in stale_ids:
            del self.state.tracks[tid]

        self.state.dog_inside = any_confirmed

        entered = any_confirmed and not was_inside
        left = not any_confirmed and was_inside

        if entered:
            self.state.last_change_time = now
            self.state.enter_count += 1
        if left:
            self.state.last_change_time = now
            self.state.leave_count += 1

        return entered, left

    def as_dict(self) -> dict:
        return {
            "dog_inside": self.state.dog_inside,
            "last_change_time": self.state.last_change_time,
            "enter_count": self.state.enter_count,
            "leave_count": self.state.leave_count,
            "tracks": {
                tid: {
                    "in_roi_count": t.in_roi_count,
                    "absent_count": t.absent_count,
                    "confirmed": t.confirmed,
                }
                for tid, t in self.state.tracks.items()
            },
        }

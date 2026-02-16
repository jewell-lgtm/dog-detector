import logging
import time

import numpy as np

from app.camera import CameraThread
from app.config import Config
from app.detector import DogDetector
from app.overlay import OverlayPainter
from app.roi import ROI
from app.script_runner import ScriptRunner
from app.state import AppState
from app.tracker import Tracker

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: Config, state: AppState):
        self.config = config
        self.state = state
        self.roi = ROI()
        if config.roi_points:
            self.roi.set_points([tuple(p) for p in config.roi_points])
            self.state.roi_points = self.roi.points
        self.detector = DogDetector(confidence=config.confidence)
        self.tracker = Tracker(
            enter_frames=config.enter_frames,
            leave_frames=config.leave_frames,
            min_overlap=config.min_overlap,
        )
        self.overlay = OverlayPainter()
        self.script_runner = ScriptRunner(cooldown=config.cooldown)

        self._last_detections: list = []
        self._frame_count = 0
        self._inference_count = 0
        self._fps_time = time.time()
        self._fps_frames = 0
        self._inf_time = time.time()
        self._inf_frames = 0

        self._camera = CameraThread(source=config.camera_device, on_frame=self._on_frame)

    def start(self):
        self._camera.start()

    def stop(self):
        self._camera.stop()

    def _on_frame(self, frame: np.ndarray):
        try:
            self._process_frame(frame)
        except Exception:
            logger.exception("pipeline frame processing crashed")

    def _process_frame(self, frame: np.ndarray):
        self._frame_count += 1
        self.state.frame_count = self._frame_count

        # fps calc
        self._fps_frames += 1
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed >= 1.0:
            self.state.timings["camera_fps"] = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_time = now

        # check for web ROI updates
        web_roi = self.state.consume_roi_update()
        if web_roi is not None:
            self.roi.set_points(web_roi)
            self._save_roi()

        # check for web triggers
        trig_enter, trig_leave = self.state.consume_triggers()
        if trig_enter and self.config.enter_script:
            if self.script_runner.run(self.config.enter_script):
                self.state.log_event("MANUAL ENTER TRIGGER")
        if trig_leave and self.config.leave_script:
            if self.script_runner.run(self.config.leave_script):
                self.state.log_event("MANUAL LEAVE TRIGGER")

        # detect every Nth frame
        detections = self._last_detections
        inference_ms = self.state.timings.get("inference_ms", 0)
        if self._frame_count % self.config.inference_interval == 0:
            t0 = time.time()
            detections = self.detector.detect(frame)
            inference_ms = (time.time() - t0) * 1000
            self._last_detections = detections
            self._inference_count += 1
            self.state.inference_count = self._inference_count

            self._inf_frames += 1
            inf_elapsed = now - self._inf_time
            if inf_elapsed >= 1.0:
                self.state.timings["inference_fps"] = self._inf_frames / inf_elapsed
                self._inf_frames = 0
                self._inf_time = now

            # track
            entered, left = self.tracker.update(detections, self.roi)
            if entered:
                self.state.log_event("DOG ENTERED")
                if self.config.enter_script:
                    if self.script_runner.run(self.config.enter_script):
                        self.state.log_event("Fired enter script")
            if left:
                self.state.log_event("DOG LEFT")
                if self.config.leave_script:
                    if self.script_runner.run(self.config.leave_script):
                        self.state.log_event("Fired leave script")

        # draw overlay
        annotated = self.overlay.draw(frame, detections, self.roi, self.tracker.state.dog_inside)
        render_ms = (time.time() - now) * 1000

        # update shared state
        self.state.update_frame(annotated, detections, self.tracker.as_dict(), inference_ms, render_ms)

    def _save_roi(self):
        self.state.roi_points = self.roi.points
        self.config.roi_points = [list(p) for p in self.roi.points]
        self.config.save()

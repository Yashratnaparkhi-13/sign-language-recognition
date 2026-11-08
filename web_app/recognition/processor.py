from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import av
import cv2
from streamlit_webrtc import VideoProcessorBase

from asl_app.hand_detector import HandDetector
from asl_app.model import LandmarkClassifier
from .state import RecognitionState

LOGGER = logging.getLogger(__name__)


class ASLVideoProcessor(VideoProcessorBase):
    def __init__(
        self,
        state: RecognitionState,
        classifier: LandmarkClassifier,
        task_path: Path,
        frame_interval: float = 0.0,
        detector_factory=HandDetector,
    ) -> None:
        self.state = state
        self.classifier = classifier
        self.task_path = task_path
        self.frame_interval = frame_interval
        self._last_processed_at = 0.0
        self._last_label: Optional[str] = None
        self._last_confidence = 0.0
        self._lock = threading.Lock()
        self._detector = None
        try:
            self._detector = detector_factory(task_path)
        except Exception as error:
            self.state.set_error(f"Hand detector unavailable: {error}")
            LOGGER.exception("Could not initialize MediaPipe hand detector")
            raise

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")
        mirrored = cv2.flip(bgr, 1)
        now = time.monotonic()

        if self.frame_interval and now - self._last_processed_at < self.frame_interval:
            return av.VideoFrame.from_ndarray(mirrored, format="bgr24")

        self._last_processed_at = now
        try:
            rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
            landmarks = self._detector.detect(rgb)

            label: Optional[str] = None
            confidence = 0.0
            if landmarks:
                label, confidence = self.classifier.predict(landmarks)
                for landmark in landmarks:
                    x = int(landmark.x * mirrored.shape[1])
                    y = int(landmark.y * mirrored.shape[0])
                    cv2.circle(mirrored, (x, y), 3, (62, 220, 160), -1)

            with self._lock:
                stable, stable_confidence, accepted = self.state.stabilizer.update(
                    label,
                    confidence,
                    bool(landmarks),
                )
                self._last_label = stable
                self._last_confidence = stable_confidence

            self.state.set_prediction(stable, stable_confidence)
            if accepted is not None:
                self.state.accept(accepted)
            self.state.set_error("")
            self.state.mark_frame()
        except Exception as error:
            self.state.set_error(f"Recognition error: {error}")
            LOGGER.exception("WebRTC frame processing failed")

        return av.VideoFrame.from_ndarray(mirrored, format="bgr24")

    def on_ended(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None

from __future__ import annotations

from pathlib import Path
from typing import Any

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


class HandDetector:
    def __init__(self, task_path: Path) -> None:
        if not task_path.is_file():
            raise FileNotFoundError(f"MediaPipe task file not found: {task_path}")

        base_options = mp_python.BaseOptions(model_asset_path=str(task_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb_frame: Any) -> list[Any]:
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        result = self._landmarker.detect(image)
        return result.hand_landmarks[0] if result.hand_landmarks else []

    def close(self) -> None:
        self._landmarker.close()

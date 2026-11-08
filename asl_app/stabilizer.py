from __future__ import annotations

from collections import Counter, deque
from typing import Optional

from .model import EXPECTED_CLASSES


class PredictionStabilizer:
    def __init__(
        self,
        confidence_threshold: float = 0.70,
        history_size: int = 10,
        required_votes: int = 7,
        release_frames: int = 8,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.required_votes = required_votes
        self.release_frames = release_frames
        self.history: deque[Optional[tuple[str, float]]] = deque(
            maxlen=history_size
        )
        self.stable_class: Optional[str] = None
        self.stable_confidence = 0.0
        self._missing_frames = 0

    def update(
        self,
        label: Optional[str],
        confidence: float,
        hand_present: bool,
    ) -> tuple[Optional[str], float, Optional[str]]:
        accepted: Optional[str] = None
        if not hand_present:
            self._missing_frames += 1
            self.history.append(None)
            if self._missing_frames >= self.release_frames:
                self.history.clear()
                self.stable_class = None
                self.stable_confidence = 0.0
            return self.stable_class, self.stable_confidence, accepted

        self._missing_frames = 0
        if label in EXPECTED_CLASSES and confidence >= self.confidence_threshold:
            self.history.append((label, confidence))
        else:
            self.history.append(None)

        votes = Counter(
            item[0] for item in self.history if item is not None
        )
        if votes:
            candidate, count = votes.most_common(1)[0]
            if count >= self.required_votes and candidate != self.stable_class:
                self.stable_class = candidate
                matching_confidences = [
                    item[1]
                    for item in self.history
                    if item is not None and item[0] == candidate
                ]
                self.stable_confidence = sum(matching_confidences) / len(
                    matching_confidences
                )
                accepted = candidate
        return self.stable_class, self.stable_confidence, accepted

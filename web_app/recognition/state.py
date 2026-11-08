from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

from asl_app.sentence_manager import SentenceManager
from asl_app.stabilizer import PredictionStabilizer


@dataclass(frozen=True)
class RecognitionSnapshot:
    current_class: Optional[str]
    confidence: float
    sentence: str
    recent: tuple[str, ...]
    last_completed_word: str
    word_event_id: int
    error: str
    processed_frames: int


class RecognitionState:
    """Thread-safe state shared between WebRTC callbacks and Streamlit."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.stabilizer = PredictionStabilizer()
        self.sentence = SentenceManager()
        self.current_class: Optional[str] = None
        self.confidence = 0.0
        self.recent: deque[str] = deque(maxlen=12)
        self.last_completed_word = ""
        self.word_event_id = 0
        self.error = ""
        self.processed_frames = 0

    def set_prediction(self, label: Optional[str], confidence: float) -> None:
        with self.lock:
            self.current_class = label
            self.confidence = confidence

    def accept(self, label: str) -> None:
        with self.lock:
            update = self.sentence.apply(label)
            self.recent.append(update.action)
            if update.spoken_word:
                self.last_completed_word = update.spoken_word
                self.word_event_id += 1

    def clear(self) -> None:
        with self.lock:
            self.sentence.clear()
            self.recent.clear()
            self.last_completed_word = ""

    def backspace(self) -> None:
        with self.lock:
            self.sentence.text = self.sentence.text[:-1]

    def set_error(self, message: str) -> None:
        with self.lock:
            self.error = message

    def mark_frame(self) -> None:
        with self.lock:
            self.processed_frames += 1

    def snapshot(self) -> RecognitionSnapshot:
        with self.lock:
            return RecognitionSnapshot(
                current_class=self.current_class,
                confidence=self.confidence,
                sentence=self.sentence.text,
                recent=tuple(self.recent),
                last_completed_word=self.last_completed_word,
                word_event_id=self.word_event_id,
                error=self.error,
                processed_frames=self.processed_frames,
            )

from __future__ import annotations

import logging
from queue import Empty, Queue
from typing import Callable, Optional

from PySide6.QtCore import QThread, Signal

LOGGER = logging.getLogger(__name__)


class SpeechWorker(QThread):
    status_changed = Signal(bool, str)
    error_occurred = Signal(str)

    def __init__(
        self,
        engine_factory: Optional[Callable[[], object]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._engine_factory = engine_factory
        self._commands: Queue[tuple[str, str]] = Queue()
        self._shutdown_requested = False

    def speak(self, text: str) -> None:
        if text.strip():
            self._commands.put(("speak", text.strip()))

    def set_muted(self, muted: bool) -> None:
        self._commands.put(("mute", "1" if muted else "0"))

    def shutdown(self) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        while True:
            try:
                self._commands.get_nowait()
            except Empty:
                break
        self._commands.put(("shutdown", ""))

    def run(self) -> None:
        try:
            if self._engine_factory is None:
                import pyttsx3

                engine = pyttsx3.init()
            else:
                engine = self._engine_factory()
            engine.setProperty("rate", 150)
            engine.setProperty("volume", 1.0)
            self.status_changed.emit(True, "Voice ready")
        except Exception as error:
            LOGGER.exception("Could not initialize text-to-speech")
            self.status_changed.emit(False, "Voice unavailable")
            self.error_occurred.emit(f"Text-to-speech unavailable: {error}")
            return

        muted = False
        while True:
            operation, value = self._commands.get()
            if operation == "shutdown":
                engine.stop()
                break
            if operation == "mute":
                muted = value == "1"
                if muted:
                    engine.stop()
                engine.setProperty("volume", 0.0 if muted else 1.0)
                self.status_changed.emit(True, "Voice muted" if muted else "Voice ready")
                continue
            if operation == "speak" and not muted:
                try:
                    engine.say(value)
                    engine.runAndWait()
                except Exception as error:
                    LOGGER.exception("Text-to-speech request failed")
                    self.error_occurred.emit(f"Speech failed: {error}")

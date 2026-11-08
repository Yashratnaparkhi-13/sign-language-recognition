from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .model import EXPECTED_CLASSES


@dataclass(frozen=True)
class SentenceUpdate:
    text: str
    action: str
    spoken_word: Optional[str] = None


class SentenceManager:
    def __init__(self) -> None:
        self.text = ""

    def apply(self, label: str) -> SentenceUpdate:
        if label not in EXPECTED_CLASSES:
            raise ValueError(f"Unsupported model class: {label}")

        if len(label) == 1 and label.isalpha():
            self.text += label
            return SentenceUpdate(self.text, label)

        if label == "del":
            self.text = self.text[:-1]
            return SentenceUpdate(self.text, "DELETE")

        completed_word = None
        if self.text and not self.text.endswith(" "):
            completed_word = self.text.split()[-1]
            self.text += " "
        return SentenceUpdate(self.text, "SPACE", completed_word)

    def clear(self) -> None:
        self.text = ""

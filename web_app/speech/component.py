from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT = components.declare_component(
    "asl_browser_speech",
    path=str(Path(__file__).resolve().parent / "frontend"),
)


def browser_speech(
    *,
    enabled: bool,
    word_event_id: int,
    completed_word: str,
    replay_word_id: int,
    sentence_event_id: int,
    sentence: str,
    key: str = "asl-browser-speech",
):
    return _COMPONENT(
        enabled=enabled,
        word_event_id=word_event_id,
        completed_word=completed_word,
        replay_word_id=replay_word_id,
        sentence_event_id=sentence_event_id,
        sentence=sentence,
        key=key,
        default="idle",
    )

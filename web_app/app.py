from __future__ import annotations

import logging
import os
import sys
import time
from html import escape
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit_webrtc import RTCConfiguration, webrtc_streamer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asl_app.model import LandmarkClassifier
from web_app.recognition.processor import ASLVideoProcessor
from web_app.recognition.state import RecognitionState
from web_app.speech.component import browser_speech

LOGGER = logging.getLogger(__name__)
MODEL_PATH = PROJECT_ROOT / "sign_landmark.pt"
TASK_PATH = PROJECT_ROOT / "hand_landmarker.task"

st.set_page_config(
    page_title="Signa · ASL Recognition",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def load_classifier(checkpoint_path: str) -> LandmarkClassifier:
    return LandmarkClassifier(Path(checkpoint_path))


def initialize_state() -> None:
    defaults = {
        "recognition_state": RecognitionState,
        "voice_enabled": True,
        "replay_word_id": 0,
        "sentence_event_id": 0,
        "speech_status": "idle",
        "camera_desired_state": False,
        "camera_start_time": 0.0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value() if key == "recognition_state" else value


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp {
          background:
            radial-gradient(ellipse at 12% 0%, rgba(32, 99, 174, .18), transparent 34%),
            #080e19;
          color: #e8eef8;
          font-family: 'Avenir Next', 'Segoe UI', system-ui, sans-serif;
        }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 2rem; }
        .hero { display:flex; align-items:center; justify-content:space-between; margin-bottom:1.35rem; }
        .brand { font:800 2rem 'Avenir Next','Segoe UI',sans-serif; letter-spacing:-.06em; color:#f2f7ff; }
        .brand span { color:#56a7ff; }
        .tagline { color:#90a2bb; font-size:.94rem; margin-top:.15rem; }
        .eyebrow { color:#82a8d7; font-size:.72rem; letter-spacing:.14em; font-weight:700; margin-bottom:.55rem; }
        .panel {
          background:linear-gradient(145deg,rgba(18,30,49,.97),rgba(12,21,35,.97));
          border:1px solid #20324c; border-radius:18px; padding:1.2rem 1.35rem;
          box-shadow:0 15px 44px rgba(0,0,0,.18);
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
          background:linear-gradient(145deg,rgba(18,30,49,.97),rgba(12,21,35,.97));
          border-color:#20324c; border-radius:18px;
        }
        .camera-panel { min-height:420px; }
        .sign-card {
          display:flex; align-items:center; justify-content:center; min-height:146px;
          background:linear-gradient(145deg,#142c4d,#10213a);
          border:1px solid #2c5b91; border-radius:15px; color:#76bdff;
          font:800 5.2rem 'Avenir Next','Segoe UI',sans-serif; margin:.4rem 0 .6rem;
        }
        .confidence { color:#a5b8d2; text-align:center; font-size:.9rem; }
        .sentence {
          min-height:110px; white-space:pre-wrap; overflow-wrap:anywhere;
          background:#0a1422; border:1px solid #253a56; border-radius:14px;
          padding:1.15rem; color:#f4f8ff; font:700 1.65rem 'Avenir Next','Segoe UI',sans-serif;
        }
        .history { color:#c8d8ec; font-size:1.08rem; word-break:break-word; }
        .status-row { display:flex; gap:.55rem; flex-wrap:wrap; margin:.7rem 0 1rem; }
        .badge {
          border:1px solid #29405f; border-radius:999px; background:#11223a;
          color:#c3d6ef; padding:.35rem .7rem; font-size:.78rem;
        }
        .error-card { background:#351b25; border:1px solid #743743; border-radius:12px; padding:.8rem 1rem; color:#ffd4d7; }
        .hint { color:#8fa1b8; font-size:.84rem; }
        div.stButton > button {
          background:#142843; border:1px solid #2b4567; border-radius:10px;
          color:#e7f1ff; min-height:2.7rem; font-weight:600;
        }
        div.stButton > button:hover { border-color:#4a98e8; color:white; background:#1b3658; }
        div.stButton > button[kind="primary"] { background:#1769b5; border-color:#2587df; }
        @media (max-width: 720px) {
          .block-container { padding:1rem .75rem; }
          .sign-card { min-height:110px; font-size:4rem; }
          .sentence { font-size:1.3rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def snapshot():
    return st.session_state.recognition_state.snapshot()


def main() -> None:
    initialize_state()
    inject_styles()
    st.markdown(
        """
        <div class="hero">
          <div>
            <div class="brand">sign<span>a</span></div>
            <div class="tagline">ASL alphabet recognition · spell words one sign at a time</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    state: RecognitionState = st.session_state.recognition_state

    try:
        classifier = load_classifier(str(MODEL_PATH))
        st.session_state.model_error = ""
    except Exception as error:
        classifier = None
        st.session_state.model_error = str(error)
        LOGGER.exception("Could not load the ASL model")

    try:
        app_secrets = st.secrets
    except StreamlitSecretNotFoundError:
        app_secrets = {}

    provider_secret_names = (
        "CLOUDFLARE_TURN_KEY_ID",
        "CLOUDFLARE_TURN_KEY_API_TOKEN",
        "HF_TOKEN",
    )
    for name in provider_secret_names:
        secret_value = app_secrets.get(name)
        if secret_value and not os.getenv(name):
            os.environ[name] = str(secret_value)

    turn_secrets = app_secrets.get("turn", {})
    turn_url = os.getenv("ASL_TURN_URL") or turn_secrets.get("url")
    turn_username = os.getenv("ASL_TURN_USERNAME") or turn_secrets.get("username")
    turn_credential = os.getenv("ASL_TURN_CREDENTIAL") or turn_secrets.get(
        "credential"
    )
    manual_turn_values = (turn_url, turn_username, turn_credential)
    manual_turn_configured = all(manual_turn_values)
    manual_turn_incomplete = any(manual_turn_values) and not manual_turn_configured

    cloudflare_credentials = (
        os.getenv("CLOUDFLARE_TURN_KEY_ID")
        or app_secrets.get("CLOUDFLARE_TURN_KEY_ID"),
        os.getenv("CLOUDFLARE_TURN_KEY_API_TOKEN")
        or app_secrets.get("CLOUDFLARE_TURN_KEY_API_TOKEN"),
    )
    hf_token = os.getenv("HF_TOKEN")
    provider_credentials = (cloudflare_credentials, (hf_token,))
    provider_turn_configured = any(all(pair) for pair in provider_credentials)
    provider_turn_incomplete = any(
        any(pair) and not all(pair) for pair in provider_credentials
    )
    turn_configured = manual_turn_configured or provider_turn_configured
    turn_incomplete = not turn_configured and (
        manual_turn_incomplete or provider_turn_incomplete
    )

    rtc_configuration = None
    if manual_turn_configured:
        ice_servers = [
            {
                "urls": [
                    os.getenv("ASL_STUN_URL", "stun:stun.l.google.com:19302")
                ]
            },
            {
                "urls": [turn_url],
                "username": turn_username,
                "credential": turn_credential,
            },
        ]
        rtc_configuration = RTCConfiguration({"iceServers": ice_servers})
    elif not provider_turn_configured:
        custom_stun_url = os.getenv("ASL_STUN_URL")
        if custom_stun_url:
            rtc_configuration = RTCConfiguration(
                {"iceServers": [{"urls": [custom_stun_url]}]}
            )
    if classifier is None:
        st.error(f"Model unavailable: {st.session_state.model_error}")
    if not TASK_PATH.is_file():
        st.error(f"MediaPipe hand-landmarker asset not found: {TASK_PATH}")

    camera_column, current_column = st.columns([1.5, 1], gap="large")
    with camera_column:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">LIVE CAMERA · WEBRTC</div>', unsafe_allow_html=True)
            st.caption("Camera video is processed for recognition and is not saved.")
            if classifier is not None and TASK_PATH.is_file():
                try:
                    context = webrtc_streamer(
                        key="asl-webcam",
                        desired_playing_state=bool(
                            st.session_state.get("camera_desired_state", False)
                        ),
                        video_processor_factory=lambda: ASLVideoProcessor(
                            state,
                            classifier,
                            TASK_PATH,
                            frame_interval=1 / 24,
                        ),
                        media_stream_constraints={
                            "video": {
                                "width": {"ideal": 640},
                                "height": {"ideal": 480},
                                "frameRate": {"ideal": 24, "max": 30},
                            },
                            "audio": False,
                        },
                        rtc_configuration=rtc_configuration,
                        async_processing=False,
                        video_receiver_size=1,
                        media_toggle_controls=False,
                    )
                    st.session_state.webrtc_context = context
                except Exception as error:
                    message = f"WebRTC could not start: {error}"
                    state.set_error(message)
                    LOGGER.exception("WebRTC initialization failed")
                    st.error(message)
            else:
                st.info("Camera is unavailable until both model assets are present.")
    with current_column:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">CURRENT SIGN</div>', unsafe_allow_html=True)
            current_sign_slot = st.empty()
            confidence_slot = st.empty()
        system_status_slot = st.empty()
    text_column, history_column = st.columns([1.5, 1], gap="large")
    with text_column:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">RECOGNIZED TEXT</div>', unsafe_allow_html=True)
            sentence_slot = st.empty()
    with history_column:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">RECENT RECOGNITION</div>', unsafe_allow_html=True)
            history_slot = st.empty()
    error_slot = st.empty()

    @st.fragment(run_every="300ms")
    def live_panel() -> None:
        control_columns = st.columns([1.2, 1.1, 1.1, 1.1, 1.4, 2.3])
        with control_columns[0]:
            if st.button(
                "▶  Start camera",
                type="primary",
                use_container_width=True,
                disabled=(
                    classifier is None
                    or not TASK_PATH.is_file()
                    or st.session_state.camera_desired_state
                ),
            ):
                st.session_state.camera_desired_state = True
                st.session_state.camera_start_time = time.monotonic()
                st.rerun()
        with control_columns[1]:
            if st.button(
                "■  Stop camera",
                use_container_width=True,
                disabled=not st.session_state.camera_desired_state,
            ):
                st.session_state.camera_desired_state = False
                st.session_state.camera_start_time = 0.0
                st.rerun()
        with control_columns[2]:
            if st.button("↤  Delete", use_container_width=True):
                state.backspace()
        with control_columns[3]:
            if st.button("Clear", use_container_width=True):
                state.clear()
        with control_columns[4]:
            if st.button(
                "🔊 Voice ON" if st.session_state.voice_enabled else "🔇 Voice OFF",
                use_container_width=True,
            ):
                st.session_state.voice_enabled = not st.session_state.voice_enabled
        with control_columns[5]:
            replay_columns = st.columns(2)
            with replay_columns[0]:
                if st.button(
                    "↻ Replay word",
                    use_container_width=True,
                    disabled=not bool(snapshot().last_completed_word),
                ):
                    st.session_state.replay_word_id += 1
            with replay_columns[1]:
                if st.button(
                    "▶ Speak sentence",
                    use_container_width=True,
                    disabled=not bool(snapshot().sentence.strip()),
                ):
                    st.session_state.sentence_event_id += 1

        st.markdown(
            '<div class="hint">Hold each gesture until accepted. To repeat the same letter, briefly remove your hand from the frame, then sign it again. The model supports A–Z, SPACE, and DELETE.</div>',
            unsafe_allow_html=True,
        )

        current = snapshot()
        context = st.session_state.get("webrtc_context")
        camera_connected = bool(context and context.state.playing)
        desired = st.session_state.get("camera_desired_state", False)
        camera_status = "CONNECTED" if camera_connected else "DISCONNECTED"
        recognition_status = "ACTIVE" if camera_connected else "STOPPED"
        model_status = "READY" if not st.session_state.model_error else "ERROR"
        voice_status = "ON" if st.session_state.voice_enabled else "OFF"
        if desired and not camera_connected:
            camera_status = (
                "NEGOTIATING"
                if context and context.state.signalling
                else "WAITING"
            )
            recognition_status = "STARTING"
        turn_status = "CONFIGURED" if turn_configured else "NOT CONFIGURED"

        system_status_slot.markdown(
            '<div class="status-row">'
            f'<span class="badge">Camera · {camera_status}</span>'
            f'<span class="badge">Model · {model_status}</span>'
            f'<span class="badge">Recognition · {recognition_status}</span>'
            f'<span class="badge">Voice · {voice_status}</span>'
            f'<span class="badge">TURN · {turn_status}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Frames received and processed: {current.processed_frames}")
        if current.current_class == "space":
            shown_sign = "SPACE"
        elif current.current_class == "del":
            shown_sign = "DELETE"
        else:
            shown_sign = current.current_class or "—"
        current_sign_slot.markdown(
            f'<div class="sign-card">{escape(shown_sign)}</div>',
            unsafe_allow_html=True,
        )
        confidence_text = (
            f"Confidence · {current.confidence:.1%}"
            if current.current_class is not None
            else "Hold a sign steadily to begin"
        )
        confidence_slot.markdown(
            f'<div class="confidence">{confidence_text}</div>',
            unsafe_allow_html=True,
        )
        sentence_slot.markdown(
            f'<div class="sentence">{escape(current.sentence or "Your recognized text will appear here")}</div>',
            unsafe_allow_html=True,
        )
        history = " → ".join(current.recent) if current.recent else "No gestures accepted yet"
        history_slot.markdown(
            f'<div class="history">{escape(history)}</div>',
            unsafe_allow_html=True,
        )
        if current.error:
            error_slot.markdown(
                f'<div class="error-card">{escape(current.error)}</div>',
                unsafe_allow_html=True,
            )
        elif desired and not camera_connected and st.session_state.get(
            "camera_start_time", 0
        ) and time.monotonic() - st.session_state.camera_start_time > 10:
            if turn_incomplete:
                error_slot.warning(
                    "Camera has not connected. TURN settings are incomplete; "
                    "configure url, username, and credential under [turn] in "
                    "Streamlit Secrets."
                )
            elif not turn_configured:
                error_slot.warning(
                    "Camera has not connected and no TURN credentials were "
                    "loaded. Add url, username, and credential under [turn] "
                    "in Streamlit Secrets; STUN alone may be blocked."
                )
            else:
                error_slot.warning(
                    "Camera has not connected although TURN settings were "
                    "loaded. Verify the TURN provider URL, port, and credentials "
                    "in Streamlit Secrets."
                )
        else:
            error_slot.empty()

        status = browser_speech(
            enabled=st.session_state.voice_enabled,
            word_event_id=current.word_event_id,
            completed_word=current.last_completed_word,
            replay_word_id=st.session_state.replay_word_id,
            sentence_event_id=st.session_state.sentence_event_id,
            sentence=current.sentence,
            key="asl-browser-speech",
        )
        st.session_state.speech_status = status or "idle"
        if st.session_state.speech_status == "error":
            st.warning("Speech synthesis is unavailable in this browser.")
        elif st.session_state.speech_status == "speaking":
            st.caption("🔊 Speaking on this device…")

    live_panel()
    st.markdown(
        '<div class="hint">Privacy: camera frames are processed in memory for recognition and are not stored. On public hosting, access the app over HTTPS and configure TURN if a restrictive network blocks WebRTC.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

"""Streamlit page (extra ``ui``): browser microphone in, browser audio out, session-scoped memory.

Run: streamlit run src/calmvoice/app/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from calmvoice import DISCLAIMER
from calmvoice.companion import build_companion
from calmvoice.config import Settings
from calmvoice.memory import SessionMemory
from calmvoice.voice import Audio, ToneTTS


@st.cache_resource
def _settings() -> Settings:
    return Settings.from_env()


def _companion():
    # One companion per browser session. The index is loaded from disk (built once), not re-embedded.
    if "companion" not in st.session_state:
        s = _settings()
        st.session_state.memory = SessionMemory(enabled=st.session_state.get("memory_on", False))
        st.session_state.companion = build_companion(s, memory=st.session_state.memory)
        st.session_state.turns = []
    return st.session_state.companion


@st.cache_resource
def _voice():
    try:
        from calmvoice.voice import GTTSTextToSpeech, WhisperSTT

        return WhisperSTT(), GTTSTextToSpeech()
    except RuntimeError:
        return None, ToneTTS()


st.set_page_config(page_title="calmvoice", page_icon=":speech_balloon:")
st.title("calmvoice")
st.info(DISCLAIMER)
st.caption("Privacy: messages stay in this browser session only. Nothing is written to disk.")

memory_on = st.sidebar.toggle("Remember this conversation (this session only)", value=False, key="memory_on")
companion = _companion()
companion.memory.enabled = memory_on
if st.sidebar.button("Clear conversation"):
    companion.memory.clear()
    st.session_state.turns = []

stt, tts = _voice()
spoken = st.audio_input("Speak (optional)") if stt is not None else None
typed = st.chat_input("Type a message")

message = typed
if spoken is not None and stt is not None:
    message = stt.transcribe(Audio(spoken.getvalue(), spoken.type or "audio/wav"))

if message:
    reply = companion.respond(message)
    st.session_state.turns.append((message, reply))

for user_text, reply in st.session_state.turns:
    with st.chat_message("user"):
        st.write(user_text)
    with st.chat_message("assistant"):
        if reply.route == "crisis":
            st.error(reply.text)
        else:
            st.write(reply.text)
        if reply.sources:
            with st.expander("Sources"):
                for src in reply.sources:
                    st.write(f"[{src.n}] {src.title} ({src.source}, {src.license})")

if st.session_state.turns:
    last = st.session_state.turns[-1][1]
    speech = tts.synthesize(last.text)
    st.audio(speech.data, format=speech.mime)

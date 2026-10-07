"""Voice adapters that work on BYTES, never on server devices.

The browser records the audio and plays the answer. The server only receives audio bytes and returns
audio bytes. This module never opens a microphone or a speaker, so voice works the same way on a
laptop and on a remote server, and it never blocks a worker in a playback loop.
"""

from __future__ import annotations

import io
import math
import struct
import tempfile
import wave
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Audio:
    data: bytes
    mime: str  # "audio/wav", "audio/mpeg", ...


class SpeechToText(Protocol):
    def transcribe(self, audio: Audio) -> str: ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str) -> Audio: ...


class ScriptedSTT:
    """Returns a fixed transcript. For tests and the offline demo."""

    def __init__(self, transcript: str):
        self.transcript = transcript
        self.received: list[Audio] = []

    def transcribe(self, audio: Audio) -> str:
        if not audio.data:
            raise ValueError("empty audio")
        self.received.append(audio)
        return self.transcript


class ToneTTS:
    """Offline stand-in for a TTS engine: a short, quiet WAV whose length grows with the text.

    It proves the bytes-in, bytes-out contract with no network and no extra package.
    """

    def __init__(self, sample_rate: int = 8000, seconds_per_word: float = 0.05, max_seconds: float = 3.0):
        self.sample_rate, self.seconds_per_word, self.max_seconds = sample_rate, seconds_per_word, max_seconds

    def synthesize(self, text: str) -> Audio:
        words = max(1, len(text.split()))
        n = int(self.sample_rate * min(self.max_seconds, words * self.seconds_per_word))
        frames = b"".join(
            struct.pack("<h", int(800 * math.sin(2 * math.pi * 440 * i / self.sample_rate))) for i in range(n)
        )
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(frames)
        return Audio(buf.getvalue(), "audio/wav")


class GTTSTextToSpeech:  # pragma: no cover - optional extra, needs network
    """Google Translate TTS through gTTS (extra ``voice``). Returns MP3 bytes."""

    def __init__(self, lang: str = "en"):
        try:
            from gtts import gTTS
        except ImportError as exc:
            raise RuntimeError('gTTS needs the extra: pip install -e ".[voice]"') from exc
        self._gtts, self.lang = gTTS, lang

    def synthesize(self, text: str) -> Audio:
        buf = io.BytesIO()
        self._gtts(text=text, lang=self.lang).write_to_fp(buf)
        return Audio(buf.getvalue(), "audio/mpeg")


class WhisperSTT:  # pragma: no cover - optional extra, downloads a model
    """faster-whisper speech-to-text (extra ``voice``). Runs locally after the model download."""

    def __init__(self, model_size: str = "base", device: str = "cpu"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError('Whisper needs the extra: pip install -e ".[voice]"') from exc
        self._model = WhisperModel(model_size, device=device, compute_type="int8")

    def transcribe(self, audio: Audio) -> str:
        suffix = ".wav" if "wav" in audio.mime else ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(audio.data)
            tmp.flush()
            segments, _ = self._model.transcribe(tmp.name)
            return " ".join(s.text.strip() for s in segments).strip()


def voice_turn(companion, stt: SpeechToText, tts: TextToSpeech, audio: Audio):
    """One spoken turn: transcribe the browser audio, answer, and return (transcript, reply, audio)."""
    transcript = stt.transcribe(audio)
    reply = companion.respond(transcript)
    return transcript, reply, tts.synthesize(reply.text)

import io
import sys
import wave

import pytest

from calmvoice.companion import Companion, Reply, build_companion
from calmvoice.config import Settings
from calmvoice.generation import SYSTEM_POLICY, ExtractiveGenerator, LLMGenerator
from calmvoice.llm import LLMError, ScriptedLLM
from calmvoice.memory import SessionMemory
from calmvoice.safety import load_resources
from calmvoice.voice import Audio, ScriptedSTT, ToneTTS, voice_turn


def make(retriever, corpus, generator, memory=None, fallback=None):
    return Companion(retriever, generator, corpus, load_resources("US"), memory=memory, fallback=fallback)


def test_crisis_never_reaches_the_llm(retriever, corpus):
    # Problem 1: the safety gate runs first and the LLM is not called.
    llm = ScriptedLLM(["some answer [1]"])
    reply = make(retriever, corpus, LLMGenerator(llm)).respond("I want to kill myself")
    assert reply.route == "crisis"
    assert "988" in reply.text
    assert llm.calls == []
    assert reply.sources == []


def test_out_of_scope_never_reaches_the_llm(retriever, corpus):
    llm = ScriptedLLM(["x"])
    reply = make(retriever, corpus, LLMGenerator(llm)).respond("Should I stop my antidepressants?")
    assert reply.route == "out_of_scope" and llm.calls == []


def test_answer_has_citations_and_licensed_sources(retriever, corpus):
    reply = make(retriever, corpus, ExtractiveGenerator()).respond("I worry all day and cannot stop")
    assert reply.route == "answer"
    assert "[1]" in reply.text
    assert reply.sources and all(s.license == "CC0-1.0" for s in reply.sources)
    assert reply.generator == "offline-extractive"


def test_llm_prompt_contains_policy_and_numbered_sources(retriever, corpus):
    llm = ScriptedLLM(["Slow breathing can help [1]."])
    reply = make(retriever, corpus, LLMGenerator(llm)).respond("how do I calm down with breathing")
    call = llm.calls[0]
    assert call["system"] == SYSTEM_POLICY
    assert "[1]" in call["prompt"] and "User message:" in call["prompt"]
    assert reply.text == "Slow breathing can help [1]."


def test_llm_diagnosis_is_blocked(retriever, corpus):
    llm = ScriptedLLM(["You have clinical depression [1]."])
    reply = make(retriever, corpus, LLMGenerator(llm)).respond("I feel sad every day")
    assert reply.route == "blocked" and reply.sources == [] and "diagnosis" in reply.guard_reasons


def test_llm_failure_falls_back_to_offline(retriever, corpus):
    class Down:
        def complete(self, system, prompt, history=()):
            raise LLMError("offline")

    reply = make(retriever, corpus, LLMGenerator(Down()), fallback=ExtractiveGenerator()).respond("tips for sleep")
    assert reply.route == "answer" and reply.generator == "offline-extractive"
    with pytest.raises(LLMError):
        make(retriever, corpus, LLMGenerator(Down())).respond("tips for sleep")


def test_concern_adds_check_in_and_resources(retriever, corpus):
    reply = make(retriever, corpus, ExtractiveGenerator()).respond("I feel worthless lately")
    assert reply.risk_level == "concern" and "988" in reply.text and reply.route == "answer"


def test_memory_is_opt_in_and_bounded(retriever, corpus):
    # Problem 7: memory lives in the session object, is off by default and has a limit.
    off = make(retriever, corpus, ExtractiveGenerator())
    off.respond("tips for sleep")
    assert len(off.memory) == 0

    llm = ScriptedLLM(["ok [1]"])
    mem = SessionMemory(enabled=True, max_turns=2)
    c = make(retriever, corpus, LLMGenerator(llm), memory=mem)
    for msg in ["sleep tips", "breathing tips", "worry tips"]:
        c.respond(msg)
    assert len(mem) == 2
    assert llm.calls[-1]["history"][0] == ("user", "sleep tips")
    assert mem.history()[0] == ("user", "breathing tips")
    mem.clear()
    assert mem.history() == []
    with pytest.raises(ValueError):
        SessionMemory(max_turns=0)


def test_empty_message(retriever, corpus):
    assert make(retriever, corpus, ExtractiveGenerator()).respond("   ").route == "empty"


def test_reply_has_no_debug_fields(retriever, corpus):
    # Problem 9: the user sees a clean reply object, never a raw chain dump.
    d = make(retriever, corpus, ExtractiveGenerator()).respond("sleep").to_dict()
    assert set(d) == {"text", "route", "risk_level", "risk_categories", "sources", "guard_reasons", "generator"}


def test_build_companion_offline(tmp_path):
    s = Settings(index_dir=str(tmp_path / "idx"))
    c = build_companion(s)
    assert isinstance(c.respond("help me relax"), Reply)
    assert (tmp_path / "idx" / "index_meta.json").is_file()


def test_tone_tts_returns_wav_bytes():
    audio = ToneTTS().synthesize("hello there friend")
    assert audio.mime == "audio/wav"
    with wave.open(io.BytesIO(audio.data)) as w:
        assert w.getnframes() > 0 and w.getnchannels() == 1


def test_voice_turn_uses_bytes_only(retriever, corpus):
    # Problem 3: audio arrives as bytes from the browser and leaves as bytes. No server devices.
    stt = ScriptedSTT("I cannot sleep at night")
    transcript, reply, audio = voice_turn(make(retriever, corpus, ExtractiveGenerator()), stt, ToneTTS(),
                                          Audio(b"RIFF....", "audio/wav"))
    assert transcript == "I cannot sleep at night" and reply.route == "answer"
    assert isinstance(audio.data, bytes) and len(audio.data) > 44
    with pytest.raises(ValueError):
        stt.transcribe(Audio(b"", "audio/wav"))


def test_no_server_audio_device_libraries_loaded():
    import calmvoice.voice  # noqa: F401

    for name in ("pyaudio", "pygame", "playsound", "speech_recognition", "sounddevice"):
        assert name not in sys.modules

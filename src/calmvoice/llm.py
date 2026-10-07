"""LLM adapters behind one small interface.

``OllamaLLM`` talks to a local Ollama server with the standard library only, so user text never goes
to a third-party API. ``ScriptedLLM`` returns fixed replies for tests.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol, Sequence


class LLM(Protocol):
    def complete(self, system: str, prompt: str, history: Sequence[tuple[str, str]] = ()) -> str: ...


class LLMError(RuntimeError):
    pass


class OllamaLLM:
    def __init__(self, url: str = "http://localhost:11434", model: str = "llama3.2", temperature: float = 0.2, timeout: float = 120.0):
        self.url, self.model, self.temperature, self.timeout = url.rstrip("/"), model, temperature, timeout

    def complete(self, system: str, prompt: str, history: Sequence[tuple[str, str]] = ()) -> str:
        messages = [{"role": "system", "content": system}]
        for role, content in history:
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})
        body = json.dumps(
            {"model": self.model, "messages": messages, "stream": False, "options": {"temperature": self.temperature}}
        ).encode("utf-8")
        req = urllib.request.Request(self.url + "/api/chat", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310 - local URL from config
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        content = (data.get("message") or {}).get("content", "")
        if not content.strip():
            raise LLMError("Ollama returned an empty message")
        return content.strip()


class ScriptedLLM:
    """Returns the scripted replies in order (the last one repeats). Records every call."""

    def __init__(self, replies: Sequence[str]):
        if not replies:
            raise ValueError("ScriptedLLM needs at least one reply")
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, system: str, prompt: str, history: Sequence[tuple[str, str]] = ()) -> str:
        self.calls.append({"system": system, "prompt": prompt, "history": list(history)})
        idx = min(len(self.calls) - 1, len(self.replies) - 1)
        return self.replies[idx]

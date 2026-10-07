"""Settings from environment variables (and an optional local .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ENV_PREFIX = "CALMVOICE_"


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """Read KEY=VALUE lines from a local .env file. Existing variables win."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def _get(name: str, default: str) -> str:
    value = os.environ.get(ENV_PREFIX + name, "")
    return value if value.strip() else default


@dataclass(frozen=True)
class Settings:
    llm: str = "offline"  # "offline" or "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    temperature: float = 0.2
    embedder: str = "hashing"  # "hashing" or "minilm"
    corpus_path: str = ""  # empty = bundled demo corpus
    index_dir: str = "indexes/default"
    top_k: int = 4
    region: str = "US"
    memory_enabled: bool = False
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, dotenv: bool = True) -> "Settings":
        if dotenv:
            load_dotenv()
        temperature = float(_get("TEMPERATURE", "0.2"))
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("CALMVOICE_TEMPERATURE must be between 0 and 1")
        top_k = int(_get("TOP_K", "4"))
        if top_k < 1:
            raise ValueError("CALMVOICE_TOP_K must be 1 or more")
        llm = _get("LLM", "offline").lower()
        if llm not in {"offline", "ollama"}:
            raise ValueError("CALMVOICE_LLM must be 'offline' or 'ollama'")
        embedder = _get("EMBEDDER", "hashing").lower()
        if embedder not in {"hashing", "minilm"}:
            raise ValueError("CALMVOICE_EMBEDDER must be 'hashing' or 'minilm'")
        return cls(
            llm=llm,
            ollama_url=_get("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            ollama_model=_get("OLLAMA_MODEL", "llama3.2"),
            temperature=temperature,
            embedder=embedder,
            corpus_path=_get("CORPUS", ""),
            index_dir=_get("INDEX_DIR", "indexes/default"),
            top_k=top_k,
            region=_get("REGION", "US").upper(),
            memory_enabled=_get("MEMORY", "0").lower() in {"1", "true", "yes", "on"},
        )

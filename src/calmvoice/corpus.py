"""The curated knowledge corpus: schema, licence gate and loader.

The corpus holds only psychoeducation text with a licence that allows reuse. Personal posts from
forums or social media are rejected by the schema, because the companion shows its sources to users.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Topic = Literal["stress", "anxiety", "low-mood", "sleep", "coping", "help-seeking", "relationships"]
ContentType = Literal["psychoeducation", "self-help-exercise", "service-information"]

# Licences that allow reuse and redistribution of the text.
REUSE_LICENSES = {"CC0-1.0", "CC-BY-4.0", "CC-BY-SA-4.0", "OGL-UK-3.0", "PUBLIC-DOMAIN", "MIT"}


class CorpusDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    title: str = Field(min_length=3)
    topic: Topic
    content_type: ContentType
    source: str = Field(min_length=2, description="Publisher or author of the text")
    url: str = ""
    license: str
    text: str = Field(min_length=40)

    @field_validator("license")
    @classmethod
    def _licence_allows_reuse(cls, value: str) -> str:
        if value not in REUSE_LICENSES:
            raise ValueError(
                f"licence {value!r} is not in the reuse allow-list {sorted(REUSE_LICENSES)}; "
                "do not add text that you have no right to redistribute"
            )
        return value


class Corpus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_id: str
    version: str
    documents: list[CorpusDocument] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "Corpus":
        ids = [d.doc_id for d in self.documents]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate doc_id values: {dupes}")
        return self

    def fingerprint(self) -> str:
        """A stable hash of the corpus content. The index stores it to detect a stale index."""
        payload = json.dumps(
            [d.model_dump() for d in self.documents], sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def by_id(self) -> dict[str, CorpusDocument]:
        return {d.doc_id: d for d in self.documents}


def load_corpus(path: str | Path | None = None) -> Corpus:
    """Load and validate a corpus JSON file. With no path, load the bundled demo corpus."""
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        text = resources.files("calmvoice.data").joinpath("corpus.json").read_text(encoding="utf-8")
    return Corpus.model_validate_json(text)

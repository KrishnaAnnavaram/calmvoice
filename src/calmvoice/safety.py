"""The safety layer: crisis detection, scope limits, the output guard and crisis resources.

The safety gate runs BEFORE retrieval and generation. A ``crisis`` message never reaches the LLM:
the companion answers with a fixed escalation message and region-specific crisis lines.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import IntEnum
from importlib import resources
from typing import Protocol


class RiskLevel(IntEnum):
    NONE = 0
    CONCERN = 1
    CRISIS = 2

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def parse(cls, value: str) -> "RiskLevel":
        return cls[value.strip().upper()]


@dataclass(frozen=True)
class Rule:
    category: str
    level: RiskLevel
    pattern: re.Pattern


def _r(category: str, level: RiskLevel, pattern: str) -> Rule:
    return Rule(category, level, re.compile(pattern, re.I))


_NOT = r"(?:don'?t|do not|dont)"
_CANT = r"(?:can'?t|cannot|cant)"
_FAMILY = r"(?:mom|mum|dad|mother|father|wife|husband|partner|boyfriend|girlfriend|boss|brother|sister|child|kids?|son|daughter|roommate|friend|neighbou?r|teacher)"

CRISIS_RULES: tuple[Rule, ...] = (
    _r("suicide", RiskLevel.CRISIS, r"\bsuicid(?:e|al)\b"),
    _r("suicide", RiskLevel.CRISIS, r"\b(?:kill|killing|end|ending|take|taking)\s+(?:myself|my\s+(?:own\s+)?life)\b"),
    _r("suicide", RiskLevel.CRISIS, r"\bwant(?:ed|s)?\s+to\s+die\b"),
    _r("suicide", RiskLevel.CRISIS, r"\b(?:wish|hope)\s+i\s+(?:was|were)\s+(?:dead|never\s+born)\b"),
    _r("suicide", RiskLevel.CRISIS, r"\bbetter\s+off\s+(?:dead|without\s+me)\b"),
    _r("suicide", RiskLevel.CRISIS, r"\bno\s+(?:reason|point)\s+(?:to|in)\s+(?:live|living|go(?:ing)?\s+on|be(?:ing)?\s+alive)\b"),
    _r("suicide", RiskLevel.CRISIS, _NOT + r"\s+want\s+to\s+(?:live|be\s+alive|be\s+here|wake\s+up)\b"),
    _r("suicide", RiskLevel.CRISIS, r"\b" + _CANT + r"\s+go\s+on\b"),
    _r("suicide", RiskLevel.CRISIS, r"\b(?:goodbye|final)\s+(?:letter|note|message)\b"),
    _r("self_harm", RiskLevel.CRISIS, r"\bself[-\s]?harm\w*\b"),
    _r("self_harm", RiskLevel.CRISIS, r"\b(?:cut|cutting|burn|burning|hurt|hurting|harm|harming|punish|punishing)\s+(?:myself|my\s+(?:arms?|legs?|wrists?|skin))\b"),
    _r("self_harm", RiskLevel.CRISIS, r"\boverdos(?:e|ed|ing)\b"),
    _r("self_harm", RiskLevel.CRISIS, r"\b(?:took|take|taking|swallowed?)\s+(?:all|too\s+many|a\s+lot\s+of|a\s+handful\s+of)\s+(?:of\s+)?(?:my\s+)?(?:pills|tablets|meds)\b"),
    _r("harm_to_others", RiskLevel.CRISIS, r"\b(?:kill|hurt|attack|stab|shoot|strangle)\s+(?:him|her|them|someone|somebody|people|everyone|my\s+" + _FAMILY + r")\b"),
    _r("abuse", RiskLevel.CRISIS, r"\b(?:he|she|they|my\s+" + _FAMILY + r")\s+(?:hits?|beats?|chokes?|kicks?|threatens?|hurts?)\s+me\b"),
    _r("abuse", RiskLevel.CRISIS, r"\b(?:being|was|am|got|been)\s+(?:abused|assaulted|raped|attacked)\b"),
    _r("abuse", RiskLevel.CRISIS, r"\bnot\s+safe\s+(?:at\s+home|with\s+(?:him|her|them))\b"),
)

CONCERN_RULES: tuple[Rule, ...] = (
    _r("hopelessness", RiskLevel.CONCERN, r"\bhopeless(?:ness)?\b"),
    _r("hopelessness", RiskLevel.CONCERN, r"\bworthless\b"),
    _r("hopelessness", RiskLevel.CONCERN, r"\b(?:a|such\s+a|just\s+a)\s+burden\b"),
    _r("hopelessness", RiskLevel.CONCERN, r"\bno\s*one\s+(?:would|will)\s+(?:care|miss\s+me|notice)\b"),
    _r("hopelessness", RiskLevel.CONCERN, r"\bnothing\s+(?:matters|will\s+(?:ever\s+)?get\s+better)\b"),
    _r("hopelessness", RiskLevel.CONCERN, r"\b" + _CANT + r"\s+cope\b"),
    _r("hopelessness", RiskLevel.CONCERN, r"\bgive\s+up\s+on\s+(?:everything|life|myself)\b"),
    _r("medical", RiskLevel.CONCERN, r"\bchest\s+pains?\b"),
    _r("medical", RiskLevel.CONCERN, r"\b(?:fainted|passed\s+out|blacked\s+out)\b"),
    _r("medical", RiskLevel.CONCERN, r"\bhaven'?t\s+(?:eaten|slept)\s+(?:in|for)\s+days\b"),
)

ALL_RULES: tuple[Rule, ...] = CRISIS_RULES + CONCERN_RULES


def normalise(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class RiskAssessment:
    level: RiskLevel
    categories: tuple[str, ...] = ()
    matches: tuple[str, ...] = ()
    source: str = "rules"

    @property
    def is_crisis(self) -> bool:
        return self.level == RiskLevel.CRISIS


class RiskClassifier(Protocol):
    def predict_level(self, text: str) -> tuple[RiskLevel, float]: ...


class RuleDetector:
    """Keyword and phrase rules. Tuned for recall: a false alarm costs less than a missed crisis."""

    def __init__(self, rules: tuple[Rule, ...] = ALL_RULES):
        self.rules = rules

    def assess(self, text: str) -> RiskAssessment:
        t = normalise(text)
        level, cats, hits = RiskLevel.NONE, [], []
        for rule in self.rules:
            m = rule.pattern.search(t)
            if m:
                level = max(level, rule.level)
                if rule.category not in cats:
                    cats.append(rule.category)
                hits.append(m.group(0))
        return RiskAssessment(level, tuple(cats), tuple(hits))


class SafetyGate:
    """Rules plus an optional learned classifier. The final level is the HIGHER of the two."""

    def __init__(self, detector: RuleDetector | None = None, classifier: RiskClassifier | None = None):
        self.detector = detector or RuleDetector()
        self.classifier = classifier

    def assess(self, text: str) -> RiskAssessment:
        result = self.detector.assess(text)
        if self.classifier is None:
            return result
        level, _ = self.classifier.predict_level(text)
        if level > result.level:
            return RiskAssessment(level, result.categories or ("classifier",), result.matches, "classifier")
        return result


# ---------------------------------------------------------------- scope limits

_DIAGNOSIS = re.compile(
    r"\b(?:do\s+i\s+have|diagnos\w*|am\s+i\s+(?:bipolar|depressed|autistic|psychotic|schizophrenic|crazy)"
    r"|what\s+(?:disorder|condition|illness)\s+(?:do\s+i|have\s+i))\b",
    re.I,
)
_DRUGS = (
    r"sertraline|fluoxetine|citalopram|escitalopram|paroxetine|venlafaxine|duloxetine|bupropion|mirtazapine"
    r"|xanax|alprazolam|lorazepam|diazepam|clonazepam|lithium|quetiapine|olanzapine|aripiprazole|lamotrigine"
    r"|zoloft|prozac|lexapro|wellbutrin|seroquel|adderall|ritalin"
)
_MEDICATION = re.compile(
    r"\b(?:dose|doses|dosage|\d+\s?mg|medication|medications|meds|antidepressants?|ssris?|prescri\w*|" + _DRUGS + r")\b",
    re.I,
)


@dataclass(frozen=True)
class ScopeResult:
    in_scope: bool
    reason: str = ""


def check_scope(text: str) -> ScopeResult:
    t = normalise(text)
    if _MEDICATION.search(t):
        return ScopeResult(False, "medication")
    if _DIAGNOSIS.search(t):
        return ScopeResult(False, "diagnosis")
    return ScopeResult(True)


SCOPE_REPLIES = {
    "medication": (
        "I cannot give advice about medication, doses or prescriptions. "
        "Please ask the doctor or pharmacist who knows your health history. "
        "Do not stop or change a medication without them."
    ),
    "diagnosis": (
        "I cannot diagnose a condition, and a label from a chatbot would not be reliable. "
        "A doctor or a mental health professional can do a proper assessment. "
        "I can share general information about coping with how you feel, if that helps."
    ),
}

# ---------------------------------------------------------------- output guard

_DIAGNOSIS_CLAIM = re.compile(
    r"\byou\s+(?:have|are\s+suffering\s+from|suffer\s+from|might\s+have|may\s+have|probably\s+have|clearly\s+have)\s+"
    r"(?:clinical\s+|major\s+|severe\s+)?(?:depression|bipolar|an?\s+anxiety\s+disorder|ptsd|ocd|adhd|schizophrenia|"
    r"borderline|a\s+(?:mental\s+)?(?:disorder|illness))"
    r"|\byou\s+are\s+(?:bipolar|clinically\s+depressed|schizophrenic|psychotic)\b|\bi\s+diagnose\b",
    re.I,
)
_MED_ADVICE = re.compile(
    r"\b\d+\s?mg\b|\b(?:take|increase|decrease|stop|start|double|skip)\w*\s+(?:your\s+|the\s+|a\s+)?"
    r"(?:medication|meds|pills|dose|antidepressants?)\b|\b(?:" + _DRUGS + r")\b",
    re.I,
)
_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class GuardResult:
    text: str
    blocked: bool = False
    reasons: list[str] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)


SAFE_FALLBACK = (
    "I am sorry, I cannot answer that in a safe way. "
    "A doctor or a mental health professional is the right person for this question."
)


def guard_output(text: str, n_sources: int) -> GuardResult:
    """Block diagnoses and medication advice. Remove citations that point to no source."""
    reasons = []
    if _DIAGNOSIS_CLAIM.search(text):
        reasons.append("diagnosis")
    if _MED_ADVICE.search(text):
        reasons.append("medication")
    if reasons:
        return GuardResult(SAFE_FALLBACK, True, reasons)
    invalid, cited = [], []
    for m in _CITATION.finditer(text):
        n = int(m.group(1))
        (cited if 1 <= n <= n_sources else invalid).append(n)
    cleaned = _CITATION.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", text)
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
    return GuardResult(cleaned, False, [], sorted(set(invalid)), sorted(set(cited)))


# ---------------------------------------------------------------- crisis resources


@dataclass(frozen=True)
class CrisisLine:
    name: str
    contact: str
    url: str


@dataclass(frozen=True)
class RegionResources:
    region: str
    emergency: str
    lines: tuple[CrisisLine, ...]


def load_resources(region: str) -> RegionResources:
    data = json.loads(resources.files("calmvoice.data").joinpath("resources.json").read_text(encoding="utf-8"))
    regions = data["regions"]
    key = region.upper()
    if key == "UK":
        key = "GB"
    entry = regions.get(key) or regions["DEFAULT"]
    used = key if key in regions else "DEFAULT"
    lines = tuple(CrisisLine(**ln) for ln in entry["lines"])
    if used != "DEFAULT":
        d = regions["DEFAULT"]["lines"][0]
        lines = lines + (CrisisLine(**d),)
    return RegionResources(used, entry["emergency"], lines)


def format_resources(res: RegionResources) -> str:
    items = "\n".join(f"- {ln.name}: {ln.contact} ({ln.url})" for ln in res.lines)
    return f"If you are in immediate danger, call {res.emergency}.\n{items}"


def escalation_message(assessment: RiskAssessment, res: RegionResources) -> str:
    cats = set(assessment.categories)
    if "harm_to_others" in cats:
        opening = (
            "It sounds like you have very strong feelings right now, and someone could get hurt. "
            "Please move away from the situation and contact a crisis line or emergency services now."
        )
    elif "abuse" in cats:
        opening = (
            "I am sorry that this is happening to you. You deserve to be safe. "
            "Please contact emergency services or a crisis line, and if you can, go to a safe place."
        )
    else:
        opening = (
            "It sounds like you are in a lot of pain right now, and I am glad that you told me. "
            "Your safety matters. Please contact a crisis line or emergency services now. "
            "Trained people are there to listen at any time."
        )
    return (
        f"{opening}\n\n{format_resources(res)}\n\n"
        "I am an automated program, not a counsellor, and I cannot help in an emergency. "
        "If you can, tell someone you trust who is near you."
    )

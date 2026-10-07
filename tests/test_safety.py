import pytest

from calmvoice.safety import (
    RiskLevel,
    RuleDetector,
    SafetyGate,
    check_scope,
    escalation_message,
    guard_output,
    load_resources,
)

CRISIS = [
    "I want to die",
    "I've been thinking about ending my life",
    "everyone would be better off without me",
    "I don’t want to wake up tomorrow",
    "I feel suicidal",
    "I cut myself again last night",
    "I took too many pills",
    "I want to hurt my roommate",
    "my partner hits me",
    "I am not safe at home",
    "I can't go on like this",
]
HARMLESS = [
    "this deadline is killing me",
    "I could die of embarrassment",
    "I'm dying to see my sister",
    "I hurt my back at the gym",
    "how can I sleep better",
    "breathing exercises for panic",
]


@pytest.mark.parametrize("text", CRISIS)
def test_crisis_messages_detected(text):
    assert RuleDetector().assess(text).level == RiskLevel.CRISIS


@pytest.mark.parametrize("text", HARMLESS)
def test_harmless_idioms_not_flagged(text):
    assert RuleDetector().assess(text).level == RiskLevel.NONE


def test_concern_level():
    a = RuleDetector().assess("I feel worthless and hopeless")
    assert a.level == RiskLevel.CONCERN and a.categories == ("hopelessness",)


def test_highest_level_wins():
    a = RuleDetector().assess("I feel hopeless and I want to die")
    assert a.level == RiskLevel.CRISIS
    assert set(a.categories) == {"suicide", "hopelessness"}


def test_gate_takes_higher_of_rules_and_classifier():
    class Alarm:
        def predict_level(self, text):
            return RiskLevel.CRISIS, 0.9

    a = SafetyGate(classifier=Alarm()).assess("a calm message")
    assert a.is_crisis and a.source == "classifier"

    class Quiet:
        def predict_level(self, text):
            return RiskLevel.NONE, 0.9

    assert SafetyGate(classifier=Quiet()).assess("I want to die").is_crisis


def test_scope_limits():
    assert check_scope("what dose of sertraline should I take").reason == "medication"
    assert check_scope("do I have bipolar disorder").reason == "diagnosis"
    assert check_scope("how do I relax").in_scope


def test_output_guard_blocks_diagnosis_and_medication():
    assert guard_output("It sounds like you have clinical depression [1].", 2).blocked
    assert guard_output("Try to take 50 mg each morning.", 2).blocked
    assert guard_output("You could increase your medication.", 2).blocked
    ok = guard_output("Slow breathing can help [1].", 2)
    assert not ok.blocked and ok.cited == [1]


def test_output_guard_removes_invalid_citations():
    g = guard_output("Breathing helps [1]. Walking helps [7].", 2)
    assert g.invalid_citations == [7]
    assert "[7]" not in g.text and "[1]" in g.text


def test_resources_by_region():
    us = load_resources("us")
    assert us.region == "US" and us.emergency == "911"
    assert any("988" in ln.contact for ln in us.lines)
    assert load_resources("UK").region == "GB"
    other = load_resources("ZZ")
    assert other.region == "DEFAULT" and "findahelpline" in other.lines[0].url


def test_escalation_message_variants():
    res = load_resources("GB")
    det = RuleDetector()
    msg = escalation_message(det.assess("I want to die"), res)
    assert "116 123" in msg and "999" in msg and "not a counsellor" in msg
    assert "safe place" in escalation_message(det.assess("my dad hits me"), res)
    assert "someone could get hurt" in escalation_message(det.assess("I want to hurt him"), res)

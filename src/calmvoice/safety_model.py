"""An optional learned risk classifier (scikit-learn, extra ``ml``).

It is a second opinion next to the rules: ``SafetyGate`` takes the HIGHER level of the two. It is
evaluated with GroupKFold on the template id, so no paraphrase of a test template is in training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .safety import RiskLevel
from .synthetic import RedTeamItem


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise RuntimeError('the learned classifier needs: pip install -e ".[ml]"') from exc


class LearnedRiskClassifier:
    """Character and word TF-IDF + logistic regression with balanced class weights."""

    def __init__(self, crisis_threshold: float = 0.35, seed: int = 0):
        _require_sklearn()
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, Pipeline

        self.crisis_threshold = crisis_threshold
        self.pipeline = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("word", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)),
                            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
                        ]
                    ),
                ),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
            ]
        )

    def fit(self, texts: Sequence[str], levels: Sequence[str]) -> "LearnedRiskClassifier":
        self.pipeline.fit(list(texts), list(levels))
        return self

    def predict_level(self, text: str) -> tuple[RiskLevel, float]:
        proba = self.pipeline.predict_proba([text])[0]
        classes = list(self.pipeline.classes_)
        if "crisis" in classes and proba[classes.index("crisis")] >= self.crisis_threshold:
            return RiskLevel.CRISIS, float(proba[classes.index("crisis")])
        best = int(np.argmax(proba))
        label = classes[best]
        return RiskLevel.parse(label), float(proba[best])


@dataclass
class GroupCVResult:
    folds: int
    crisis_recall: float
    crisis_precision: float
    macro_f1: float


def group_cross_validate(items: Sequence[RedTeamItem], folds: int = 5, seed: int = 0) -> GroupCVResult:
    """GroupKFold by template id. The classifier is fit on train folds only."""
    _require_sklearn()
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold

    texts = np.array([it.text for it in items])
    y = np.array([it.level for it in items])
    groups = np.array([it.template_id for it in items])
    preds = np.empty_like(y)
    for train, test in GroupKFold(n_splits=folds).split(texts, y, groups):
        model = LearnedRiskClassifier(seed=seed).fit(texts[train], y[train])
        preds[test] = [model.predict_level(t)[0].label for t in texts[test]]
    is_c, pred_c = y == "crisis", preds == "crisis"
    recall = float((is_c & pred_c).sum() / max(1, is_c.sum()))
    precision = float((is_c & pred_c).sum() / max(1, pred_c.sum()))
    return GroupCVResult(folds, recall, precision, float(f1_score(y, preds, average="macro")))

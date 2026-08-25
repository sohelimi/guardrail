"""The model bake-off: candidate tier-1 detectors, deliberately compared.

We benchmark four classical options so the model-selection story is grounded in
measured trade-offs rather than assertion:

  1. keyword_baseline  - a transparent rule list (no ML). The honest floor.
  2. word_logreg       - TF-IDF(word) + Logistic Regression. Fast, interpretable.
  3. char_logreg       - TF-IDF(char) + Logistic Regression. Obfuscation-robust.
  4. union_logreg      - word+char union + calibrated LogReg.  <-- PRIMARY (tier-1)

Heavier options discussed in the write-up but intentionally NOT the hot-path
choice for a latency/cost-sensitive gateway:
  * sentence-embeddings + classifier  (semantic, ~10-50ms, needs a model download)
  * fine-tuned DistilBERT/DeBERTa      (best accuracy, GPU, higher latency/cost)
  * LLM-as-judge                       (flexible, but slow/costly/non-deterministic)
The LLM judge is used only as the tier-2 escalation in cascade.py.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import SEED
from .features import char_vectorizer, union_vectorizer, word_vectorizer

# ---------------------------------------------------------------------------
# 1. Transparent non-ML baseline (the floor we must beat)
# ---------------------------------------------------------------------------
_KEYWORDS = [
    r"ignore .* instructions", r"disregard", r"system prompt", r"developer mode",
    r"you are now", r"no restrictions", r"jailbreak", r"api key", r"password",
    r"exfiltrate", r"reveal .* prompt", r"forget everything",
]
_KW_RE = [re.compile(k, re.I) for k in _KEYWORDS]


class KeywordBaseline(BaseEstimator, ClassifierMixin):
    """Predicts ATTACK if any known bad phrase matches. No learning."""

    classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def _score(self, text: str) -> float:
        hits = sum(1 for r in _KW_RE if r.search(text))
        return min(1.0, hits / 2.0)

    def predict_proba(self, X):
        p1 = np.array([self._score(t) for t in X])
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ---------------------------------------------------------------------------
# 2. The learned candidates
# ---------------------------------------------------------------------------
def _logreg() -> LogisticRegression:
    # class_weight balances the (near-balanced) classes and is honest about the
    # asymmetric cost; C tuned lightly for text.
    return LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                              random_state=SEED)


def build_models() -> dict[str, Pipeline | KeywordBaseline]:
    """Return every candidate as a fit-able estimator, keyed by name."""
    return {
        "keyword_baseline": KeywordBaseline(),
        "word_logreg": Pipeline([("tfidf", word_vectorizer()), ("clf", _logreg())]),
        "char_logreg": Pipeline([("tfidf", char_vectorizer()), ("clf", _logreg())]),
        "union_logreg": Pipeline([("tfidf", union_vectorizer()), ("clf", _logreg())]),
    }


def build_primary(calibrated: bool = True):
    """The tier-1 model actually deployed in the cascade.

    Calibrated so that P(attack) is a *meaningful probability* — essential
    because the cascade's escalation band is defined in probability space.
    """
    base = Pipeline([("tfidf", union_vectorizer()), ("clf", _logreg())])
    if not calibrated:
        return base
    # sigmoid (Platt) calibration via cross-val on the training fold
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


# ---------------------------------------------------------------------------
# 3. Interpretability helper: top contributing tokens for a single prediction
# ---------------------------------------------------------------------------
def top_tokens(pipeline: Pipeline, text: str, k: int = 6) -> list[tuple[str, float]]:
    """Return the tokens pushing this text toward ATTACK (for the live demo)."""
    try:
        vec = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        feats = vec.transform([text])
        names = np.asarray(vec.get_feature_names_out())
        coef = clf.coef_[0]
        contrib = feats.multiply(coef).toarray()[0]
        order = np.argsort(contrib)[::-1]
        out = [(str(names[i]), round(float(contrib[i]), 3)) for i in order[:k] if contrib[i] > 0]
        return out
    except Exception:
        return []

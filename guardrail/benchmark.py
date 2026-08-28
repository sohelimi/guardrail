"""Benchmark multiple candidate tier-1 detectors on the same data.

Trains each model on the identical train split and scores it on both the standard
test set and the adversarial holdout, plus a real single-request latency measurement
— so model selection is a measured, apples-to-apples comparison, not an assertion.

Includes a from-scratch NumPy logistic regression (hand-written training loop) so at
least one candidate is genuinely built, not just imported. Deep-learning candidates
(embeddings, DistilBERT) are added by scripts/run_benchmark.py only when available.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import data_gen
from .config import SEED
from .evaluate import recall_at_fpr
from .features import char_vectorizer, union_vectorizer, word_vectorizer
from .models import KeywordBaseline, build_primary


# ---------------------------------------------------------------------------
class NumpyLogReg(BaseEstimator, ClassifierMixin):
    """Logistic regression with a hand-written batch-gradient-descent loop.

    sklearn-compatible (fit / predict_proba) so it drops into a Pipeline, but every
    line of the optimiser is ours — forward pass, cross-entropy gradient, L2, SGD step.
    """

    def __init__(self, lr=0.5, epochs=400, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-(X @ self.w + self.b)))   # forward
            err = p - y                                        # d(cross-entropy)/dz
            self.w -= self.lr * (X.T @ err / n + self.l2 * self.w)
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-(np.asarray(X, dtype=float) @ self.w + self.b)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _lr():
    return LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=SEED)


def _svd():
    return TruncatedSVD(n_components=200, random_state=SEED)


def classical_candidates() -> dict:
    """Every candidate is a text -> P(attack) estimator (offline, deterministic)."""
    return {
        "keyword_baseline": (KeywordBaseline(), "rules"),
        "logreg_word": (Pipeline([("t", word_vectorizer()), ("c", _lr())]), "linear"),
        "logreg_char": (Pipeline([("t", char_vectorizer()), ("c", _lr())]), "linear"),
        "logreg_union (deployed)": (build_primary(calibrated=True), "linear+cal"),
        "linear_svm": (Pipeline([("t", union_vectorizer()),
                                 ("c", CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3))]), "linear+cal"),
        "complement_nb": (Pipeline([("t", union_vectorizer()), ("c", ComplementNB())]), "bayes"),
        "sgd_logloss": (Pipeline([("t", union_vectorizer()),
                                  ("c", SGDClassifier(loss="log_loss", class_weight="balanced", random_state=SEED))]), "linear"),
        "hist_gbdt": (Pipeline([("t", union_vectorizer()), ("s", _svd()),
                                ("c", HistGradientBoostingClassifier(random_state=SEED))]), "trees"),
        "numpy_logreg (from scratch)": (Pipeline([("t", union_vectorizer()), ("s", _svd()),
                                                  ("c", NumpyLogReg())]), "from-scratch"),
    }


# ---------------------------------------------------------------------------
def _proba(model, texts):
    return model.predict_proba(list(texts))[:, 1]


def _metrics(y, s, thr=0.5):
    pred = (np.asarray(s) >= thr).astype(int)
    return {
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "pr_auc": round(average_precision_score(y, s), 4),
        "recall_at_1pct_fpr": round(recall_at_fpr(np.asarray(y), np.asarray(s), 0.01), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
    }


def _latency_ms(model, sample_texts, n=40):
    """Median single-request latency (vectorise + predict one prompt), in ms."""
    times = []
    for t in sample_texts[:n]:
        t0 = time.perf_counter()
        model.predict_proba([t])
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(times)), 3)


def evaluate_model(model, train, test, adv):
    Xtr, ytr = train["text"], train["label"].to_numpy()
    t0 = time.perf_counter()
    model.fit(Xtr, ytr)
    train_s = round(time.perf_counter() - t0, 2)
    s_te = _proba(model, test["text"])
    s_adv = _proba(model, adv["text"])
    return {
        "standard_test": _metrics(test["label"].to_numpy(), s_te),
        "adversarial_holdout": _metrics(adv["label"].to_numpy(), s_adv),
        "latency_ms": _latency_ms(model, list(test["text"])),
        "train_seconds": train_s,
    }


def run(extra_candidates: dict | None = None) -> dict:
    """Train + score every candidate; return a results dict. `extra_candidates` lets
    scripts add DL models (name -> (fitted-or-fittable estimator, family))."""
    import pandas as pd
    from .config import DATASET_CSV

    df = pd.read_csv(DATASET_CSV) if DATASET_CSV.exists() else data_gen.generate()
    cfg = data_gen.GenConfig()
    train, test, adv = data_gen.split(df, cfg)
    train = data_gen.add_label_noise(train, cfg.train_label_noise, cfg.seed)

    cands = classical_candidates()
    if extra_candidates:
        cands.update(extra_candidates)

    results = {}
    for name, (model, family) in cands.items():
        try:
            results[name] = {"family": family, **evaluate_model(model, train, test, adv)}
        except Exception as e:  # a candidate failing must not sink the whole run
            results[name] = {"family": family, "error": str(e)[:200]}
    return {"dataset": {"n_train": len(train), "n_test": len(test), "n_adversarial": len(adv)},
            "models": results}

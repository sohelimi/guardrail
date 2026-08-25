"""The two-tier cascade that ties tier-1 and tier-2 together.

    prompt ──▶ Tier-1 (cheap calibrated classifier) ──▶ P(attack)
                     │
        ┌────────────┼─────────────┐
   P < LOW       LOW ≤ P ≤ HIGH     P > HIGH
   decide         escalate to        decide
   BENIGN         Tier-2 judge       ATTACK

WHY A CASCADE: at gateway QPS you cannot afford an LLM call on every request.
Tier-1 answers the easy majority in ~1ms; only the uncertain middle band pays
for the slower judge. This buys most of the judge's accuracy at a fraction of
the cost and latency — the core architectural argument of the project.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import joblib

from .config import MODEL_PATH, UNCERTAIN_HIGH, UNCERTAIN_LOW
from .llm_judge import judge
from .models import top_tokens


@dataclass
class Decision:
    label: int                 # 1 = attack, 0 = benign
    prob: float                # tier-1 P(attack)
    tier: str                  # "tier-1" or "tier-2"
    action: str                # "allow" or "block"
    rationale: str
    top_tokens: list = field(default_factory=list)
    latency_ms: float = 0.0
    judge_source: str | None = None


class Cascade:
    def __init__(self, tier1, low: float = UNCERTAIN_LOW, high: float = UNCERTAIN_HIGH):
        self.tier1 = tier1
        self.low = low
        self.high = high

    # -- persistence -------------------------------------------------------
    def save(self, path=MODEL_PATH):
        joblib.dump({"tier1": self.tier1, "low": self.low, "high": self.high}, path)
        return path

    @classmethod
    def load(cls, path=MODEL_PATH):
        d = joblib.load(path)
        return cls(d["tier1"], d["low"], d["high"])

    # -- inference ---------------------------------------------------------
    def _prob(self, text: str) -> float:
        return float(self.tier1.predict_proba([text])[0, 1])

    def predict(self, text: str) -> Decision:
        t0 = time.perf_counter()
        p = self._prob(text)

        if p < self.low:
            dec = Decision(0, p, "tier-1", "allow",
                           f"Confidently benign (P={p:.2f} < {self.low}).")
        elif p > self.high:
            dec = Decision(1, p, "tier-1", "block",
                           f"Confidently malicious (P={p:.2f} > {self.high}).")
        else:
            j = judge(text)
            dec = Decision(j.label, p, "tier-2",
                           "block" if j.label == 1 else "allow",
                           f"Uncertain at tier-1 (P={p:.2f}); tier-2 judge → "
                           f"{'ATTACK' if j.label else 'BENIGN'}. {j.rationale}",
                           judge_source=j.source)

        # interpretability: only meaningful for the linear pipeline
        try:
            inner = self.tier1
            if hasattr(inner, "calibrated_classifiers_"):
                inner = inner.calibrated_classifiers_[0].estimator
            dec.top_tokens = top_tokens(inner, text)
        except Exception:
            dec.top_tokens = []

        dec.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return dec

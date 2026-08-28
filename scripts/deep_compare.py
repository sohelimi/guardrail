"""Deep, fair comparison: is the calibrated UNION actually better than word-only?

Answers the honest challenge from the benchmark, where word-only LogReg matched/beat
the deployed union on the adversarial holdout. Two things the main benchmark never
measured, computed here for word / char / union — each in a RAW and a fairly-CALIBRATED
variant so the calibration comparison is apples-to-apples:

  1. Calibration quality  — Brier score + Expected Calibration Error (ECE) on the
     standard test set. Probability quality is what the 0.30/0.80 cascade band relies on.
  2. Char-level obfuscation robustness — a stress holdout using obfuscations NOT in
     training that destroy word tokens but preserve character subsequences
     (space-removal / inner-char swaps, no English wrapper). This is where char n-grams
     are *supposed* to earn their place. If word-only holds up here too, the union's
     edge is not real and the deck should say so.

    KMP_DUPLICATE_LIB_OK=TRUE .venv-bench/bin/python scripts/deep_compare.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, recall_score
from sklearn.pipeline import Pipeline

from guardrail import data_gen
from guardrail.config import ARTIFACT_DIR, DATASET_CSV, SEED
from guardrail.features import char_vectorizer, union_vectorizer, word_vectorizer

RNG = np.random.default_rng(SEED)
OUT = ARTIFACT_DIR / "deep_compare.json"


# --------------------------------------------------------------------------- models
def _lr():
    return LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=SEED)


def make(vec_fn, calibrated: bool):
    pipe = Pipeline([("t", vec_fn()), ("c", _lr())])
    # calibrate the whole pipeline (vectorizer refit per fold) -> fair to every candidate
    return CalibratedClassifierCV(pipe, cv=3, method="sigmoid") if calibrated else pipe


def candidates():
    return {
        "word (raw)":              make(word_vectorizer, False),
        "word (calibrated)":       make(word_vectorizer, True),
        "char (raw)":              make(char_vectorizer, False),
        "char (calibrated)":       make(char_vectorizer, True),
        "union (raw)":             make(union_vectorizer, False),
        "union (calibrated) ★":    make(union_vectorizer, True),   # the deployed model
    }


# --------------------------------------------------------------------------- metrics
def ece(y, p, n_bins=10):
    """Expected Calibration Error — |accuracy - confidence| averaged over confidence bins."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc = y[m].mean()                       # fraction actually positive in the bin
        total += (m.sum() / len(p)) * abs(acc - conf)
    return float(total)


# --------------------------------------------------------------------------- char stress
def _concat(text: str) -> str:
    """Remove ALL spaces — destroys word tokens, keeps char subsequences. Not in training."""
    return text.replace(" ", "")


def _charswap(text: str) -> str:
    """Swap two adjacent inner characters of each longish word (typo-style). Novel."""
    out = []
    for w in text.split(" "):
        if len(w) > 3:
            i = RNG.integers(1, len(w) - 2)
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)


def build_char_stress(test: pd.DataFrame):
    """Clean (unobfuscated) TEST attacks + benign, re-obfuscated with novel char transforms.

    Uses test rows only (never trained on these exact sentences), and only obfuscation=='none'
    attacks so the ONLY novelty is our char transform."""
    atk = test[(test["label"] == 1) & (test["obfuscation"] == "none")]["text"].tolist()
    ben = test[test["label"] == 0]["text"].tolist()
    ben = list(RNG.choice(ben, size=min(len(ben), max(60, len(atk))), replace=False))
    rows = []
    for t in atk:
        rows.append((_concat(t), 1)); rows.append((_charswap(t), 1))
    for t in ben:
        rows.append((_concat(t), 0)); rows.append((_charswap(t), 0))
    df = pd.DataFrame(rows, columns=["text", "label"])
    return df


# --------------------------------------------------------------------------- run
def proba(m, texts):
    return m.predict_proba(list(texts))[:, 1]


def main():
    df = pd.read_csv(DATASET_CSV) if DATASET_CSV.exists() else data_gen.generate()
    cfg = data_gen.GenConfig()
    train, test, adv = data_gen.split(df, cfg)
    train = data_gen.add_label_noise(train, cfg.train_label_noise, cfg.seed)
    stress = build_char_stress(test)

    print(f"train {len(train)} · test {len(test)} · adv {len(adv)} · char-stress {len(stress)} "
          f"({int((stress.label==1).sum())} atk / {int((stress.label==0).sum())} benign)\n")

    yte = test["label"].to_numpy()
    yad = adv["label"].to_numpy()
    yst = stress["label"].to_numpy()

    results = {}
    for name, m in candidates().items():
        m.fit(train["text"], train["label"].to_numpy())
        pte, pad, pst = proba(m, test["text"]), proba(m, adv["text"]), proba(m, stress["text"])
        results[name] = {
            "std_f1": round(f1_score(yte, pte >= 0.5, zero_division=0), 4),
            "std_recall": round(recall_score(yte, pte >= 0.5, zero_division=0), 4),
            "brier": round(brier_score_loss(yte, pte), 4),      # lower = better
            "ece": round(ece(yte, pte), 4),                     # lower = better
            "adv_recall": round(recall_score(yad, pad >= 0.5, zero_division=0), 4),
            "charstress_recall": round(recall_score(yst[yst == 1], (pst[yst == 1] >= 0.5), zero_division=0), 4),
            "charstress_fpr": round(float(((pst[yst == 0] >= 0.5)).mean()), 4),
        }

    # ---- print
    cols = ["std_f1", "brier", "ece", "adv_recall", "charstress_recall", "charstress_fpr"]
    hdr = ["std F1", "Brier↓", "ECE↓", "adv rec", "CHAR rec", "char FPR"]
    print(f"{'model':<24}" + "".join(f"{h:>11}" for h in hdr))
    print("-" * (24 + 11 * len(hdr)))
    for name, r in results.items():
        print(f"{name:<24}" + "".join(f"{r[c]:>11}" for c in cols))

    OUT.write_text(json.dumps({"dataset": {"train": len(train), "test": len(test),
                    "adv": len(adv), "char_stress": len(stress)}, "models": results}, indent=2))
    print(f"\nWrote {OUT}")

    # ---- verdict
    w = results["word (calibrated)"]; u = results["union (calibrated) ★"]
    print("\n--- verdict (calibrated word vs calibrated union) ---")
    print(f"char-stress recall:  word {w['charstress_recall']}   union {u['charstress_recall']}"
          f"   -> union {'WINS' if u['charstress_recall'] - w['charstress_recall'] > 0.03 else 'ties'}")
    print(f"ECE (prob quality):  word {w['ece']}   union {u['ece']}")


if __name__ == "__main__":
    main()

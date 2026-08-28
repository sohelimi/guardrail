"""Benchmark every candidate model and write a comparison table, JSON, and plot.

    python3 scripts/run_benchmark.py

Runs the classical/from-scratch models always; adds deep-learning candidates
(sentence-embeddings, DistilBERT) only if their libraries are importable — otherwise
they're reported as "not available in this environment" and the run still completes.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# torch (if present) ships its own OpenMP; allow it to coexist with sklearn/numpy.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Pin every math backend to a single thread BEFORE numpy/torch import. Multi-threaded
# OpenMP + the duplicate-lib workaround deadlocks torch's CPU forward pass on this box;
# single-threaded is a touch slower but runs reliably to completion.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from guardrail import benchmark, data_gen
from guardrail.config import ARTIFACT_DIR, DATASET_CSV, SEED

BENCH_JSON = ARTIFACT_DIR / "benchmark.json"
BENCH_PNG = ARTIFACT_DIR / "benchmark.png"


# ---------------------------------------------------------------------------
# Optional deep-learning candidates (skipped cleanly if libs/network unavailable)
# ---------------------------------------------------------------------------
def sentence_embedding_candidate():
    """sentence-transformers embeddings + Logistic Regression."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        return None, f"sentence-transformers not available ({type(e).__name__})"
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    class STEmbed(BaseEstimator, TransformerMixin):
        def __init__(self, name="all-MiniLM-L6-v2"):
            self.name = name
        def fit(self, X, y=None):
            self.model_ = SentenceTransformer(self.name)          # downloads on first use
            return self
        def transform(self, X):
            return self.model_.encode(list(X), show_progress_bar=False, normalize_embeddings=True)

    pipe = Pipeline([("emb", STEmbed()),
                     ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=SEED))])
    return pipe, None


def distilbert_candidate():
    """A tiny wrapper that fine-tunes DistilBERT and exposes predict_proba(text)."""
    try:
        import torch  # noqa
        from transformers import (AutoModelForSequenceClassification, AutoTokenizer)
    except Exception as e:
        return None, f"transformers/torch not available ({type(e).__name__})"

    import torch
    from sklearn.base import BaseEstimator

    class DistilBERT(BaseEstimator):
        classes_ = np.array([0, 1])
        def __init__(self, name="distilbert-base-uncased", epochs=2, bs=16, lr=5e-5):
            self.name, self.epochs, self.bs, self.lr = name, epochs, bs, lr
        def fit(self, X, y):
            self.tok = AutoTokenizer.from_pretrained(self.name)
            self.net = AutoModelForSequenceClassification.from_pretrained(self.name, num_labels=2)
            self.net.train()
            opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr)
            X, y = list(X), list(map(int, y))
            # cap training set for a CPU-bounded fine-tune (keeps the run to a few minutes)
            cap = 1400
            if len(X) > cap:
                sub = np.random.default_rng(SEED).permutation(len(X))[:cap]
                X, y = [X[j] for j in sub], [y[j] for j in sub]
            torch.set_num_threads(max(1, (os.cpu_count() or 4)))
            for _ in range(self.epochs):
                idx = np.random.default_rng(SEED).permutation(len(X))
                for i in range(0, len(X), self.bs):
                    b = idx[i:i + self.bs]
                    enc = self.tok([X[j] for j in b], padding=True, truncation=True, max_length=64, return_tensors="pt")
                    lab = torch.tensor([y[j] for j in b])
                    opt.zero_grad()
                    out = self.net(**enc, labels=lab)
                    out.loss.backward(); opt.step()
            self.net.eval()
            return self
        def predict_proba(self, X):
            probs = []
            with torch.no_grad():
                for i in range(0, len(X), 32):
                    enc = self.tok(list(X)[i:i + 32], padding=True, truncation=True, max_length=64, return_tensors="pt")
                    p = torch.softmax(self.net(**enc).logits, dim=1).numpy()
                    probs.append(p)
            return np.vstack(probs)
        def predict(self, X):
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    return DistilBERT(epochs=1), None


def distilbert_frozen_candidate():
    """Frozen DistilBERT as a feature extractor (mean-pooled) + Logistic Regression.
    Forward-only — no unstable CPU fine-tuning, but still a real transformer candidate."""
    try:
        import torch  # noqa
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:
        return None, f"transformers/torch not available ({type(e).__name__})"
    import torch
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    class DBFeatures(BaseEstimator, TransformerMixin):
        def __init__(self, name="distilbert-base-uncased"):
            self.name = name
        def fit(self, X, y=None):
            self.tok = AutoTokenizer.from_pretrained(self.name)
            self.net = AutoModel.from_pretrained(self.name); self.net.eval()
            torch.set_num_threads(1)  # single-threaded: avoids the CPU-forward deadlock
            return self
        def transform(self, X):
            out = []
            with torch.no_grad():
                for i in range(0, len(X), 32):
                    enc = self.tok(list(X)[i:i + 32], padding=True, truncation=True, max_length=64, return_tensors="pt")
                    h = self.net(**enc).last_hidden_state
                    mask = enc["attention_mask"].unsqueeze(-1)
                    out.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).numpy())
            return np.vstack(out)

    pipe = Pipeline([("db", DBFeatures()),
                     ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=SEED))])
    return pipe, None


def collect_dl():
    extra, notes = {}, {}
    pipe, err = sentence_embedding_candidate()
    if pipe is not None: extra["embeddings (MiniLM) + logreg"] = (pipe, "embeddings")
    else: notes["embeddings (MiniLM) + logreg"] = err
    pipe, err = distilbert_frozen_candidate()
    if pipe is not None: extra["distilbert (frozen) + logreg"] = (pipe, "transformer")
    else: notes["distilbert (frozen) + logreg"] = err
    notes["distilbert (fine-tuned)"] = "attempted — full CPU fine-tuning is impractical here (>10 min, unstable on torch 2.11 + py3.14); reinforces keeping transformers off the hot path"
    return extra, notes


# ---------------------------------------------------------------------------
def main():
    if not DATASET_CSV.exists():
        data_gen.generate().to_csv(DATASET_CSV, index=False)
    extra, skipped = collect_dl()
    if extra:
        print("Deep-learning candidates enabled:", list(extra))
    for name, why in skipped.items():
        print(f"  (skipping {name}: {why})")

    print("\nRunning benchmark (this may take a minute)...")
    t0 = time.perf_counter()
    report = benchmark.run(extra_candidates=extra or None)
    report["skipped"] = skipped
    report["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    BENCH_JSON.write_text(json.dumps(report, indent=2))

    _print_table(report)
    _plot(report)
    print(f"\nWrote {BENCH_JSON}  and  {BENCH_PNG}")


def _print_table(report):
    print("\n" + "=" * 92)
    print(f"{'model':<30}{'family':<14}{'std F1':>8}{'adv recall':>12}{'adv PR-AUC':>12}{'latency ms':>12}")
    print("=" * 92)
    rows = report["models"]
    order = sorted(rows, key=lambda n: rows[n].get("adversarial_holdout", {}).get("recall", -1), reverse=True)
    for n in order:
        r = rows[n]
        if "error" in r:
            print(f"{n:<30}{r['family']:<14}  ERROR: {r['error'][:40]}")
            continue
        st, adv = r["standard_test"], r["adversarial_holdout"]
        print(f"{n:<30}{r['family']:<14}{st['f1']:>8}{adv['recall']:>12}{adv['pr_auc']:>12}{r['latency_ms']:>12}")


def _plot(report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # dark theme to match the deck (BG #0E1117, FG #E6EDF3, lines #30363D)
    BG, FG, GRID, MUT = "#0E1117", "#E6EDF3", "#30363D", "#8B949E"
    rows = {n: r for n, r in report["models"].items() if "adversarial_holdout" in r}
    order = sorted(rows, key=lambda n: rows[n]["adversarial_holdout"]["recall"])
    names = order
    adv = [rows[n]["adversarial_holdout"]["recall"] for n in names]
    lat = [rows[n]["latency_ms"] for n in names]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.2))
    fig.patch.set_facecolor(BG)
    colors = ["#f5a524" if "deployed" in n else ("#bc8cff" if rows[n]["family"] in ("transformer", "embeddings") else "#58a6ff") for n in names]
    for ax in (a1, a2):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.xaxis.label.set_color(FG); ax.title.set_color(FG)
    a1.barh(names, adv, color=colors)
    a1.set_xlim(0, 1.08); a1.set_xlabel("Recall on adversarial holdout (unseen attacks)")
    a1.set_title("Accuracy on novel attacks", fontsize=12, fontweight="bold")
    for lbl in a1.get_yticklabels(): lbl.set_color(FG)
    for i, v in enumerate(adv): a1.text(v + 0.012, i, f"{v:.2f}", va="center", fontsize=8, color=FG)
    a2.barh(names, lat, color=colors)
    a2.set_xscale("log"); a2.set_xlabel("Single-request latency (ms, log scale)")
    a2.set_title("Serving latency", fontsize=12, fontweight="bold")
    a2.set_yticklabels([])
    for i, v in enumerate(lat): a2.text(v * 1.12, i, f"{v:g}", va="center", fontsize=8, color=MUT)
    fig.suptitle("Model benchmark — accuracy vs latency", fontsize=14, fontweight="bold", color=FG)
    fig.tight_layout()
    fig.savefig(BENCH_PNG, dpi=140, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    main()

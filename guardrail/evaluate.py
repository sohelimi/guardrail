"""Formal evaluation framework.

Answers, with numbers:
  * How do the candidate models compare?  (metrics table)
  * How well do we generalise to attack families never seen in training?
    (the ADVERSARIAL holdout — the headline robustness result)
  * What does the synthetic augmentation actually buy us?  (ablation)
  * Where do we set the operating threshold given asymmetric costs?
    (cost-weighted threshold selection + PR curve)

Metrics are security-aware: we care most about RECALL @ a fixed low false-positive
rate, because blocking a legitimate employee is expensive.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import (
    average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import ARTIFACT_DIR, FN_COST, FP_COST, METRICS_JSON, SEED
from . import data_gen
from .models import build_models, build_primary


# ---------------------------------------------------------------------------
def _proba(model, X):
    return model.predict_proba(list(X))[:, 1]


def recall_at_fpr(y_true, scores, max_fpr=0.01) -> float:
    """Highest recall achievable while keeping FPR <= max_fpr."""
    neg = scores[y_true == 0]
    if len(neg) == 0:
        return float("nan")
    thr = np.quantile(neg, 1 - max_fpr)  # threshold that lets through max_fpr of negatives
    pred = scores >= thr
    return float(recall_score(y_true, pred, zero_division=0))


def cost_optimal_threshold(y_true, scores):
    """Pick the threshold minimising FP_COST*FP + FN_COST*FN."""
    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.05, 0.95, 91):
        pred = scores >= t
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        cost = FP_COST * fp + FN_COST * fn
        if cost < best_cost:
            best_cost, best_t = cost, float(t)
    return best_t, best_cost


def _metrics_block(y_true, scores, threshold=0.5) -> dict:
    pred = (scores >= threshold).astype(int)
    return {
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, scores), 4) if len(set(y_true)) > 1 else None,
        "pr_auc": round(average_precision_score(y_true, scores), 4),
        "recall@1%fpr": round(recall_at_fpr(np.asarray(y_true), np.asarray(scores), 0.01), 4),
    }


# ---------------------------------------------------------------------------
def run(train, test, adversarial) -> dict:
    import pandas as pd

    cfg = data_gen.GenConfig()
    train = data_gen.add_label_noise(train, cfg.train_label_noise, cfg.seed)
    Xtr, ytr = train["text"], train["label"].to_numpy()
    Xte, yte = test["text"], test["label"].to_numpy()

    # The adversarial holdout is all-attack (novel families/obfuscations). To make
    # precision/FPR well-defined we add the benign test prompts as controls: this
    # asks "do we catch UNSEEN attacks without raising new false positives?"
    benign_controls = test[test["label"] == 0]
    adv_eval = pd.concat([adversarial, benign_controls], ignore_index=True)
    Xadv, yadv = adv_eval["text"], adv_eval["label"].to_numpy()

    report: dict = {"bakeoff": {}, "primary": {}, "ablation": {}, "dataset": {}}

    # --- 1. Model bake-off on the standard test set ---
    for name, model in build_models().items():
        model.fit(Xtr, ytr)
        s = _proba(model, Xte)
        report["bakeoff"][name] = _metrics_block(yte, s)

    # --- 2. Primary calibrated model + the deployed CASCADE end-to-end ---
    from .cascade import Cascade

    primary = build_primary(calibrated=True)
    primary.fit(Xtr, ytr)
    s_te = _proba(primary, Xte)
    s_adv = _proba(primary, Xadv)
    thr, cost = cost_optimal_threshold(yte, s_te)

    # Threshold-independent separability (honest even where an operating point saturates)
    cascade = Cascade(primary)
    pred_te = np.array([cascade.predict(t).label for t in Xte])
    pred_adv = np.array([cascade.predict(t).label for t in Xadv])
    report["primary"] = {
        "tier1_scores": {  # probability-based, threshold-independent
            "standard_test": _metrics_block(yte, s_te, thr),
            "adversarial_holdout_pr_auc": round(average_precision_score(yadv, s_adv), 4),
        },
        "cascade_end_to_end": {  # what actually ships (tier-1 + judge)
            "standard_test": {
                "precision": round(precision_score(yte, pred_te, zero_division=0), 4),
                "recall": round(recall_score(yte, pred_te, zero_division=0), 4),
                "f1": round(f1_score(yte, pred_te, zero_division=0), 4),
            },
            "adversarial_holdout": {
                "precision": round(precision_score(yadv, pred_adv, zero_division=0), 4),
                "recall": round(recall_score(yadv, pred_adv, zero_division=0), 4),
                "f1": round(f1_score(yadv, pred_adv, zero_division=0), 4),
            },
        },
        "cost_optimal_threshold": round(thr, 3),
        "note": "adversarial_holdout = attack families/obfuscations NEVER seen in training",
    }

    # per-family recall on the adversarial (all-attack) holdout -> error analysis
    per_family = {}
    for fam in sorted(adversarial["family"].unique()):
        mask = (adversarial["family"] == fam).to_numpy()
        preds = np.array([cascade.predict(t).label for t in adversarial["text"][mask]])
        per_family[fam] = {"n": int(mask.sum()), "recall": round(float(preds.mean()), 3)}
    report["per_family_adversarial_recall"] = per_family
    _plot_family(per_family, ARTIFACT_DIR / "per_family_recall.png")

    # --- 3. Ablation: does synthetic augmentation help generalisation? ---
    cfg_noaug = data_gen.GenConfig(augment=False)
    df_noaug = data_gen.generate(cfg_noaug)
    tr2, te2, adv2 = data_gen.split(df_noaug, cfg_noaug)
    m2 = build_primary(calibrated=True).fit(tr2["text"], tr2["label"])
    # evaluate both cascades on the SAME adversarial holdout for a fair comparison
    cas2 = Cascade(m2)
    pred_adv2 = np.array([cas2.predict(t).label for t in Xadv])
    report["ablation"] = {
        "with_augmentation_adv_recall":
            report["primary"]["cascade_end_to_end"]["adversarial_holdout"]["recall"],
        "without_augmentation_adv_recall": round(
            recall_score(yadv, pred_adv2, zero_division=0), 4),
        "interpretation": "augmentation should raise recall on unseen obfuscated attacks",
    }

    # --- 4. Plots ---
    _plot_pr(yte, s_te, ARTIFACT_DIR / "pr_curve.png")
    _plot_bakeoff(report["bakeoff"], ARTIFACT_DIR / "bakeoff.png")

    report["dataset"] = {
        "n_train": int(len(train)), "n_test": int(len(test)),
        "n_adversarial": int(len(adversarial)),
    }
    METRICS_JSON.write_text(json.dumps(report, indent=2))
    return report


def _plot_pr(y_true, scores, path):
    p, r, _ = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)
    plt.figure(figsize=(5, 4))
    plt.plot(r, p, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Primary model — PR curve (AP={ap:.3f})")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def _plot_family(per_family: dict, path):
    fams = list(per_family)
    rec = [per_family[f]["recall"] for f in fams]
    plt.figure(figsize=(7, 4))
    colors = ["#c0392b" if r < 0.5 else "#e67e22" if r < 0.8 else "#27ae60" for r in rec]
    plt.barh(fams, rec, color=colors)
    plt.xlim(0, 1.0); plt.xlabel("Recall (catch rate)")
    plt.title("Recall on UNSEEN attack families (adversarial holdout)")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def _plot_bakeoff(bakeoff: dict, path):
    names = list(bakeoff)
    f1s = [bakeoff[n]["f1"] for n in names]
    praucs = [bakeoff[n]["pr_auc"] for n in names]
    x = np.arange(len(names))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, f1s, 0.4, label="F1")
    plt.bar(x + 0.2, praucs, 0.4, label="PR-AUC")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylim(0, 1.05); plt.legend(); plt.title("Model bake-off")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()

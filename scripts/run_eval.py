"""Run the full evaluation framework and print the results table."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from guardrail import data_gen, evaluate
from guardrail.config import ARTIFACT_DIR, DATASET_CSV


def _fmt_row(name, m):
    return (f"  {name:<18} P={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}  PR-AUC={m['pr_auc']:.3f}  R@1%FPR={m['recall@1%fpr']:.3f}")


def main():
    df = pd.read_csv(DATASET_CSV) if DATASET_CSV.exists() else data_gen.generate()
    train, test, adv = data_gen.split(df)
    report = evaluate.run(train, test, adv)

    print("=" * 78)
    print("MODEL BAKE-OFF  (standard test set)")
    print("=" * 78)
    for name, m in report["bakeoff"].items():
        print(_fmt_row(name, m))

    print("\n" + "=" * 78)
    print("DEPLOYED CASCADE — generalisation to UNSEEN attacks")
    print("=" * 78)
    p = report["primary"]
    print("  Standard test (cascade) :", p["cascade_end_to_end"]["standard_test"])
    print("  Adversarial (cascade)   :", p["cascade_end_to_end"]["adversarial_holdout"])
    print("  Tier-1 adversarial PR-AUC:", p["tier1_scores"]["adversarial_holdout_pr_auc"])
    print("  Cost-optimal threshold  :", p["cost_optimal_threshold"])
    print("\n  Per-family recall on unseen attacks (error analysis):")
    for fam, v in report["per_family_adversarial_recall"].items():
        print(f"    {fam:<22} n={v['n']:3d}  recall={v['recall']:.3f}")

    print("\n" + "=" * 78)
    print("ABLATION — value of synthetic augmentation")
    print("=" * 78)
    for k, v in report["ablation"].items():
        print(f"  {k}: {v}")

    print(f"\nPlots + metrics.json written to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()

"""Generate the synthetic dataset and print the fidelity report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardrail import data_gen
from guardrail.config import DATASET_CSV


def main():
    cfg = data_gen.GenConfig()
    df = data_gen.generate(cfg)
    df.to_csv(DATASET_CSV, index=False)

    report = data_gen.validate(df)
    train, test, adv = data_gen.split(df, cfg)

    print("=" * 60)
    print("SYNTHETIC DATA — FIDELITY REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))
    print("-" * 60)
    print(f"Saved {len(df)} rows -> {DATASET_CSV}")
    print(f"Split: train={len(train)}  test={len(test)}  adversarial-holdout={len(adv)}")
    print("Adversarial holdout families:", sorted(adv['family'].unique().tolist()))
    print("\nSample benign :", df[df.label == 0]['text'].iloc[0])
    print("Sample attack :", df[df.label == 1]['text'].iloc[0])


if __name__ == "__main__":
    main()

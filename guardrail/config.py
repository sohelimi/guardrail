"""Central configuration: paths, seeds, and the decision thresholds for the cascade."""
from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
DATA_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

DATASET_CSV = DATA_DIR / "guardrail_dataset.csv"
MODEL_PATH = ARTIFACT_DIR / "cascade_model.joblib"
METRICS_JSON = ARTIFACT_DIR / "metrics.json"

# --- Reproducibility ---------------------------------------------------------
SEED = 42

# --- Cascade decision policy -------------------------------------------------
# Tier-1 (cheap classifier) returns P(attack). If that probability lands in the
# uncertain band we escalate to the Tier-2 judge; otherwise we decide directly.
# The band is deliberately asymmetric: blocking a real employee (false positive)
# is expensive, so we keep the "confidently benign" region wide.
UNCERTAIN_LOW = 0.30   # below -> decide BENIGN on tier-1 alone
UNCERTAIN_HIGH = 0.80  # above -> decide ATTACK on tier-1 alone
# In between -> escalate to tier-2 judge.

# Operating threshold used when we report a single hard label for tier-1 alone.
# Chosen on the validation set to hit a target false-positive rate (see evaluate.py).
DEFAULT_THRESHOLD = 0.50

# Cost model for threshold selection: a false positive (blocking a legitimate
# employee) is treated as N times more costly than letting one attack through
# a *first* line of defense (there are downstream controls too).
FP_COST = 1.0
FN_COST = 3.0

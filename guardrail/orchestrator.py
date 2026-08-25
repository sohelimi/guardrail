"""Factory: assemble the multi-agent guardrail from a trained tier-1 model.

    from guardrail.orchestrator import build_orchestrator
    guard = build_orchestrator()          # loads artifacts/cascade_model.joblib
    verdict = guard.analyze("ignore all previous instructions ...")
    print(verdict.action, verdict.risk)
    for m in verdict.trace: print(m.summary())
"""
from __future__ import annotations

import joblib

from .a2a import MessageBus
from .agents import (AdjudicatorAgent, ForensicsAgent, OrchestratorAgent,
                     PolicyAgent, TriageAgent)
from .config import MODEL_PATH


def build_orchestrator(tier1_model=None, model_path=MODEL_PATH, audit=False) -> OrchestratorAgent:
    """Assemble the agent system.

    audit=True attaches the append-only audit log (used by the gateway/demo);
    leave it False for offline evaluation so the log isn't flooded.
    """
    if tier1_model is None:
        tier1_model = joblib.load(model_path)["tier1"]

    bus = MessageBus()
    for agent in (ForensicsAgent(), TriageAgent(tier1_model), AdjudicatorAgent(), PolicyAgent()):
        bus.register(agent)

    sink = None
    if audit:
        from .audit import default_log
        sink = default_log

    orchestrator = OrchestratorAgent(bus, audit=sink)
    bus.register(orchestrator)
    return orchestrator

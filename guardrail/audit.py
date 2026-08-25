"""Append-only audit log — persists every decision for after-the-fact review.

For a security control, "auditable by design" has to mean the decisions are
actually recorded, not just shown in a UI. This writes one JSON record per
decision to a JSONL file (one object per line — the standard shape for audit /
SIEM ingestion), capturing the verdict AND the full A2A deliberation trace, so
any block can be replayed and explained months later.

    from guardrail.audit import AuditLog
    log = AuditLog()                 # -> logs/audit.jsonl
    log.record(verdict, prompt, role)

Set `redact=True` to store only a SHA-256 of the prompt (retention / PII policy).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import threading
import uuid
from pathlib import Path

from .config import ROOT

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
DEFAULT_PATH = LOG_DIR / "audit.jsonl"

_lock = threading.Lock()  # serialize appends across the threaded demo server


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


class AuditLog:
    def __init__(self, path: Path | str = DEFAULT_PATH, redact: bool = False):
        self.path = Path(path)
        self.redact = redact

    def record(self, verdict, prompt: str, role: str) -> dict:
        """Append one decision record; returns it (with its request_id)."""
        rec = {
            "ts": _now_iso(),
            "request_id": uuid.uuid4().hex[:12],
            "role": role,
            "action": verdict.action,
            "label": verdict.label,
            "risk": round(float(verdict.risk), 4),
            "decided_by": verdict.decided_by,
            "transforms": list(verdict.transforms),
            "judge_source": verdict.judge_source,
            "latency_ms": verdict.latency_ms,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt": "<redacted>" if self.redact else prompt,
            "trace": [
                {"sender": m.sender, "receiver": m.receiver,
                 "performative": m.performative.value, "content": m.content}
                for m in verdict.trace
            ],
        }
        line = json.dumps(rec, default=str, ensure_ascii=False)
        with _lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return rec

    def tail(self, n: int = 20) -> list[dict]:
        """Return the last n records (for the demo's audit view)."""
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(x) for x in lines[-n:] if x.strip()]


# process-wide default used by the serving layer
default_log = AuditLog()

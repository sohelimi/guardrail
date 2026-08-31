"""PII/PHI detection for the PrivacyAgent — a data-loss-prevention (DLP) concern
distinct from prompt-injection defense.

DESIGN NOTE: this is regex + keyword-context detection, not a trained NER model.
That's a deliberate scope choice, not an oversight — the same honest trade-off
this project makes elsewhere (see llm_judge.py's heuristic judge): structured,
high-precision patterns (SSN format, Luhn-valid card numbers) catch the clearest
violations cheaply and explainably; a real production DLP layer would add a
trained PII-NER model (e.g. Presidio, spaCy) on top for free-text names/addresses,
which this intentionally does not attempt.

Two sensitivity tiers:
    LOW  — general PII (email, phone). Redacted in place; the request continues.
    HIGH — regulated data (SSN, credit card, medical record number, diagnosis
           disclosure, passport/driver's-license numbers). Never redacted-and-
           forwarded: the PrivacyAgent's caller routes these straight to a
           block/review decision, bypassing injection scoring entirely.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?:\+?1[-.\s]?)?\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_MRN_RE = re.compile(r"\bMRN[:\s#]*\d{5,10}\b", re.I)
_DIAGNOSIS_RE = re.compile(
    r"\b(?:diagnos(?:is|ed)|icd-?10)\b.{0,60}\b(?:patient|dob|name)\b"
    r"|\b(?:patient|dob|name)\b.{0,60}\b(?:diagnos(?:is|ed)|icd-?10)\b", re.I,
)
_PASSPORT_RE = re.compile(r"\bpassport\s*(?:no\.?|number|#)?\s*[:#]?\s*[A-Z0-9]{6,9}\b", re.I)
_DL_RE = re.compile(r"\bdriver'?s?\s*licen[cs]e\s*(?:no\.?|number|#)?\s*[:#]?\s*[A-Z0-9]{5,12}\b", re.I)


def _luhn_valid(digits: str) -> bool:
    """Standard Luhn checksum — filters out ordinary 13-19 digit numbers
    (phone extensions, tracking IDs) that aren't actually card numbers."""
    d = [int(c) for c in digits]
    checksum = 0
    for i, val in enumerate(reversed(d)):
        if i % 2 == 1:
            val *= 2
            if val > 9:
                val -= 9
        checksum += val
    return checksum % 10 == 0


@dataclass
class PIIResult:
    sensitivity: str = "none"          # "none" | "low" | "high"
    types: list = field(default_factory=list)
    redacted: str = ""                 # low-sensitivity hits masked; unchanged if high/none


def scan_pii(text: str) -> PIIResult:
    high_types: list[str] = []
    low_types: list[str] = []
    out = text

    if _SSN_RE.search(text):
        high_types.append("ssn")
    for m in _CARD_RE.finditer(text):
        digits = re.sub(r"[ -]", "", m.group())
        if len(digits) in range(13, 20) and _luhn_valid(digits):
            high_types.append("credit_card")
            break
    if _MRN_RE.search(text):
        high_types.append("medical_record_number")
    if _DIAGNOSIS_RE.search(text):
        high_types.append("diagnosis_disclosure")
    if _PASSPORT_RE.search(text):
        high_types.append("passport_number")
    if _DL_RE.search(text):
        high_types.append("drivers_license")

    if high_types:
        # regulated data present -> caller bypasses redaction/continuation entirely,
        # so no need to mask here; the raw text still travels with the finding for
        # audit purposes (the Orchestrator, not this function, decides what happens).
        return PIIResult(sensitivity="high", types=sorted(set(high_types)), redacted=text)

    if _EMAIL_RE.search(text):
        low_types.append("email")
        out = _EMAIL_RE.sub("[REDACTED-EMAIL]", out)
    if _PHONE_RE.search(text):
        low_types.append("phone")
        out = _PHONE_RE.sub("[REDACTED-PHONE]", out)

    if low_types:
        return PIIResult(sensitivity="low", types=sorted(set(low_types)), redacted=out)

    return PIIResult(sensitivity="none", types=[], redacted=text)

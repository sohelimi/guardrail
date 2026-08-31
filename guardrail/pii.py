"""PII/PHI detection for the PrivacyAgent — a data-loss-prevention (DLP) concern
distinct from prompt-injection defense.

Two-tier detection, mirroring the same "real engine, honest fallback" pattern
used by the tier-2 Adjudicator (llm_judge.py's `_llm_judge() or _heuristic_judge()`):

    * PRIMARY: a real NER (Named Entity Recognition) pass via Microsoft Presidio,
      backed by spaCy's small English model. This catches unstructured PII that
      no regex can — a bare name ("Priya Sharma"), a street address — by
      recognizing the *shape* of the entity from context, not a fixed pattern.
      Presidio's own built-in recognizers additionally give SSN/credit-card
      detection real validation (a canonical-placeholder blocklist for SSNs,
      a genuine Luhn checksum for cards) rather than the bare regex this
      project shipped with initially. Two custom recognizers are registered
      alongside spaCy's NER for entity types Presidio doesn't ship: medical
      record numbers and diagnosis-disclosure phrasing.
    * FALLBACK: if `presidio-analyzer` or the spaCy model isn't installed (or
      fails to load for any reason), we fall straight back to the original
      regex + keyword-context detection — no NER, no crash, same interface.
      This is what ran before the NER upgrade and is still fully sufficient
      for structured formats (SSN, card, MRN, diagnosis phrasing, email, phone).

Two sensitivity tiers either way:
    LOW  — general PII (email, phone, a bare person name/location). Redacted
           in place; the request continues.
    HIGH — regulated data (SSN, credit card, medical record number, diagnosis
           disclosure, passport/driver's-license numbers). Never redacted-and-
           forwarded: the PrivacyAgent's caller routes these straight to a
           block/review decision, bypassing injection scoring entirely.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Must be set before spaCy/thinc/blis are imported anywhere in the process —
# their native OpenMP runtime aborts the whole interpreter (SIGABRT, not a
# catchable exception) if a second OpenMP copy is already linked in, which
# happens routinely alongside numpy/scikit-learn on macOS. Setting this here,
# at module import time and before the lazy NER engine is ever built, is what
# makes the try/except around engine creation actually able to catch failures
# instead of the process just dying.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?:\+?1[-.\s]?)?\(?\b\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b")

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_MRN_RE = re.compile(r"\bMRN[:\s#]*\d{5,10}\b", re.I)
_DIAGNOSIS_PATTERN = (
    r"\b(?:diagnos(?:is|ed)|icd-?10)\b.{0,60}\b(?:patient|dob|name)\b"
    r"|\b(?:patient|dob|name)\b.{0,60}\b(?:diagnos(?:is|ed)|icd-?10)\b"
)
_DIAGNOSIS_RE = re.compile(_DIAGNOSIS_PATTERN, re.I)
_PASSPORT_RE = re.compile(r"\bpassport\s*(?:no\.?|number|#)?\s*[:#]?\s*[A-Z0-9]{6,9}\b", re.I)
_DL_RE = re.compile(r"\bdriver'?s?\s*licen[cs]e\s*(?:no\.?|number|#)?\s*[:#]?\s*[A-Z0-9]{5,12}\b", re.I)

# Entity types Presidio's NER pass can return, bucketed by sensitivity.
_NER_HIGH = {
    "US_SSN", "CREDIT_CARD", "US_BANK_NUMBER", "US_ITIN", "IBAN_CODE",
    "MEDICAL_LICENSE", "MRN", "DIAGNOSIS_DISCLOSURE",
    "PASSPORT_DISCLOSURE", "DRIVERS_LICENSE_DISCLOSURE",
}
_NER_LOW = {"EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON", "LOCATION", "NRP"}
# NRP ("nationality/religious/political group") is included here because spaCy's
# small model has a documented bias: it misclassifies some non-Western names as
# NRP instead of PERSON (verified directly — "Priya Sharma" -> NRP 0.85, "John
# Smith" -> PERSON 0.85, same sentence shape). Treating NRP as low-sensitivity
# PII too is a practical mitigation, not a fix for the underlying model bias —
# a real production deployment would want a name-detection benchmark across
# name origins before trusting this model's PERSON recall unevenly.
_NER_SCORE_THRESHOLD = 0.4  # excludes Presidio's own "very weak" (<=0.3) built-ins


def _luhn_valid(digits: str) -> bool:
    """Standard Luhn checksum — filters out ordinary 13-19 digit numbers
    (phone extensions, tracking IDs) that aren't actually card numbers.
    Used only in the regex-fallback path; the NER path delegates to
    Presidio's own (identical) Luhn validation on CREDIT_CARD."""
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
    source: str = "regex"              # "ner" | "regex" — which path produced this result


def _regex_scan(text: str) -> PIIResult:
    """Fallback path: no NER engine available. Structured formats only."""
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
        return PIIResult(sensitivity="high", types=sorted(set(high_types)),
                         redacted=text, source="regex")

    if _EMAIL_RE.search(text):
        low_types.append("email")
        out = _EMAIL_RE.sub("[REDACTED-EMAIL]", out)
    if _PHONE_RE.search(text):
        low_types.append("phone")
        out = _PHONE_RE.sub("[REDACTED-PHONE]", out)

    if low_types:
        return PIIResult(sensitivity="low", types=sorted(set(low_types)),
                         redacted=out, source="regex")

    return PIIResult(sensitivity="none", types=[], redacted=text, source="regex")


_ner_engine = "unset"  # lazy singleton: "unset" -> not yet attempted; None -> unavailable; else engine


def _build_ner_engine():
    """Build a Presidio AnalyzerEngine pinned to the small (non-transformer,
    no torch dependency) spaCy model, with two custom recognizers registered
    for entity types Presidio doesn't ship out of the box. Returns None on
    any failure so the caller falls back to regex — never raises."""
    try:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=config).create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="MRN", name="MrnRecognizer",
            patterns=[Pattern("MRN", _MRN_RE.pattern, 0.85)],
        ))
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="DIAGNOSIS_DISCLOSURE", name="DiagnosisDisclosureRecognizer",
            patterns=[Pattern("DIAGNOSIS_DISCLOSURE", _DIAGNOSIS_PATTERN, 0.85)],
        ))
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="PASSPORT_DISCLOSURE", name="PassportDisclosureRecognizer",
            patterns=[Pattern("PASSPORT_DISCLOSURE", _PASSPORT_RE.pattern, 0.85)],
        ))
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity="DRIVERS_LICENSE_DISCLOSURE", name="DriversLicenseDisclosureRecognizer",
            patterns=[Pattern("DRIVERS_LICENSE_DISCLOSURE", _DL_RE.pattern, 0.85)],
        ))
        return analyzer
    except Exception:
        return None


def _get_ner_engine():
    global _ner_engine
    if _ner_engine == "unset":
        _ner_engine = _build_ner_engine()
    return _ner_engine


def _ner_scan(text: str) -> PIIResult | None:
    """Primary path. Returns None (not a PIIResult) if the engine is
    unavailable, so the caller knows to fall back rather than trust an
    'all clear' that never actually ran."""
    engine = _get_ner_engine()
    if engine is None:
        return None
    try:
        results = engine.analyze(text=text, language="en", score_threshold=_NER_SCORE_THRESHOLD)
    except Exception:
        return None

    high_types: list[str] = []
    low_hits: list[tuple[int, int, str]] = []  # (start, end, entity_type) for redaction
    for r in results:
        if r.entity_type in _NER_HIGH:
            high_types.append(r.entity_type.lower())
        elif r.entity_type in _NER_LOW:
            low_hits.append((r.start, r.end, r.entity_type))

    if high_types:
        return PIIResult(sensitivity="high", types=sorted(set(high_types)),
                         redacted=text, source="ner")

    if low_hits:
        out = text
        for start, end, etype in sorted(low_hits, key=lambda h: h[0], reverse=True):
            out = out[:start] + f"[REDACTED-{etype}]" + out[end:]
        low_types = sorted({etype.lower() for _, _, etype in low_hits})
        return PIIResult(sensitivity="low", types=low_types, redacted=out, source="ner")

    return PIIResult(sensitivity="none", types=[], redacted=text, source="ner")


def scan_pii(text: str) -> PIIResult:
    """Public entry point used by PrivacyAgent. NER path if the engine loaded
    successfully; otherwise the original regex/keyword detection."""
    return _ner_scan(text) or _regex_scan(text)

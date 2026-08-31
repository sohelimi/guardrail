"""The specialized agents that collaborate to adjudicate a prompt.

Each agent owns ONE responsibility and speaks only via A2A messages:

    ForensicsAgent   - normalises/de-obfuscates the input, reports transforms
    PrivacyAgent     - PII/PHI (data-loss-prevention) check, distinct from injection intent
    TriageAgent      - fast ML tier-1 detector -> calibrated risk score
    AdjudicatorAgent - tier-2 deep judge, only for the uncertain band
    PolicyAgent      - maps (verdict|pii_sensitivity, user role) -> concrete action
    OrchestratorAgent- coordinates the above and issues the final ruling

Separation of concerns matters here beyond tidiness: each agent is independently
testable, replaceable (swap the heuristic Adjudicator for an LLM guard model
without touching anyone else), and its messages are logged for audit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import re
import unicodedata

from .a2a import AgentMessage, MessageBus, Performative
from .config import UNCERTAIN_HIGH, UNCERTAIN_LOW
from .llm_judge import _deobfuscate, judge
from .models import top_tokens
from .pii import scan_pii

_URL_ENC_RE = re.compile(r"(?:%[0-9A-Fa-f]{2})")
_HEX_ENC_RE = re.compile(r"(?:\\x[0-9A-Fa-f]{2}){3,}")
_HTML_ENT_RE = re.compile(r"(?:&#\d+;){3,}")
_FULLWIDTH_RE = re.compile(r"[！-～]{3,}")
_ZERO_WIDTH_RE = re.compile("[​‌‍﻿]")
_COMMON_WORDS = {"the", "you", "and", "your", "please", "ignore", "instructions",
                 "system", "prompt", "reveal", "all", "this", "with", "what"}


def _looks_rot13(text: str) -> bool:
    """ROT13 has no structural signature — the only tell is that decoding it
    makes the text look MORE like English than it already did."""
    import codecs
    rot = codecs.decode(text, "rot_13")
    before = sum(1 for w in re.findall(r"[a-z]+", text.lower()) if w in _COMMON_WORDS)
    after = sum(1 for w in re.findall(r"[a-z]+", rot.lower()) if w in _COMMON_WORDS)
    return after >= 2 and after > before


class Agent:
    name: str = "agent"

    def handle(self, msg: AgentMessage, bus: MessageBus) -> list[AgentMessage]:
        return []

    def _msg(self, to, perf, **content) -> AgentMessage:
        return AgentMessage(self.name, to, perf, content)


# ---------------------------------------------------------------------------
class ForensicsAgent(Agent):
    """De-obfuscates input so downstream agents see the true intent."""
    name = "Forensics"

    def handle(self, msg, bus):
        text = msg.content["text"]
        transforms = []
        low = text.lower()
        letters = [c for c in text if c.isalpha()]
        if letters and sum(c.isupper() for c in letters) > len(letters) * 0.7:
            transforms.append("uppercase")
        if "base64" in low or _looks_b64(text):
            transforms.append("base64")
        if text.count(" ") > max(6, len(text) * 0.3):
            transforms.append("spaced")
        # leetspeak only when a digit is sandwiched inside a word (l3ak, pr3v0us),
        # so ordinary tokens like "Q3" or "SOC 2" don't trip it.
        if len(re.findall(r"[a-z][0-9][a-z]", low)) >= 1:
            transforms.append("leetspeak")
        if _URL_ENC_RE.search(text):
            transforms.append("url_encode")
        if _HEX_ENC_RE.search(text):
            transforms.append("hex_encode")
        if _HTML_ENT_RE.search(text):
            transforms.append("html_entities")
        if _FULLWIDTH_RE.search(text):
            transforms.append("fullwidth")
        if _ZERO_WIDTH_RE.search(text):
            transforms.append("zero_width")
        if _looks_rot13(text):
            transforms.append("rot13")
        # Unicode homoglyphs (Cyrillic/Greek look-alikes, math-bold, etc.) that
        # NFKC/unidecode would collapse away — a cheap giveaway is that the text
        # normalizes to something meaningfully different from itself.
        if unicodedata.normalize("NFKC", text) != text and not transforms:
            transforms.append("unicode_homoglyph")
        # only de-obfuscate when a transform was actually detected; otherwise the
        # normalized view is just the original text (clean, readable trace).
        normalized = _deobfuscate(text) if transforms else text
        return [self._msg(msg.sender, Performative.INFORM,
                          normalized=normalized, transforms=transforms,
                          obfuscated=bool(transforms))]


def _looks_b64(text: str) -> bool:
    import re
    return bool(re.search(r"[A-Za-z0-9+/]{16,}={0,2}", text))


# ---------------------------------------------------------------------------
class PrivacyAgent(Agent):
    """Data-loss-prevention (PII/PHI) check — a concern distinct from injection
    intent, and the reason this agent exists as its own pipeline stage rather
    than being folded into Forensics or Triage.

    WHY IT'S SEPARATE FROM INJECTION DEFENSE
        A prompt can be entirely benign — zero injection intent — and still be
        a compliance incident. An employee pasting a patient's SSN and
        diagnosis into a prompt to ask for help drafting a referral letter is
        not an attack; it's a data-loss event a real enterprise gateway has to
        catch regardless. Folding that into the injection classifier would
        conflate two independent questions ("is this trying to manipulate the
        model?" vs. "is regulated data about to leave the boundary?") behind a
        single score, which is neither auditable nor correct — a request can
        be a "1" on either axis independently of the other.

    WHERE IT SITS IN THE PIPELINE
        Runs immediately after Forensics and before Triage, on the
        de-obfuscated ("normalized") view of the input — so an attacker can't
        hide a leaked SSN behind leetspeak or base64 any more than they could
        hide an injection payload that way. This ordering also means neither
        the ML classifier nor the audit log ever needs to see raw regulated
        data unnecessarily: by the time Triage runs, low-sensitivity PII has
        already been masked, and high-sensitivity PII/PHI has already exited
        the pipeline entirely (see below) without ever reaching the
        injection-scoring stage.

    DETECTION STRATEGY (see guardrail/pii.py for the full implementation)
        NER-primary, regex-fallback — the exact same cascade shape the tier-2
        Adjudicator already uses (`_llm_judge() or _heuristic_judge()`):
          * Primary: a real named-entity-recognition pass (Microsoft Presidio,
            backed by spaCy's small English model — no torch dependency,
            consistent with keeping heavyweight models off the hot path).
            This catches unstructured PII no fixed pattern can, e.g. a bare
            name ("Priya Sharma") or a street address, by recognizing the
            *shape* of the entity from surrounding context. It also gives
            SSN/credit-card detection real validation instead of a bare
            pattern match — a canonical-placeholder blocklist for SSNs
            (rejects the textbook fake "123-45-6789"), a genuine Luhn
            checksum for cards. Two custom recognizers are registered
            alongside spaCy's NER for entity types Presidio doesn't ship out
            of the box: medical record numbers and diagnosis-disclosure
            phrasing.
          * Fallback: if the NER engine can't load for any reason (package
            not installed, model missing, initialization failure), detection
            falls straight back to the original regex + keyword-context
            patterns — same interface, no crash, and still fully sufficient
            for every *structured* format (SSN, card, MRN, diagnosis
            phrasing, email, phone). Only the free-text/bare-entity cases
            (a name with no other PII marker nearby) are missed in fallback
            mode — an honest, bounded degradation, not a silent one.
        `result.source` records which path actually ran ("ner" or "regex"),
        threaded all the way through to the Verdict and the demo UI as
        `pii_source`, so the audit trail never has to guess.

    SENSITIVITY TIERS AND WHAT HAPPENS NEXT (decided by the Orchestrator,
    not this agent — PrivacyAgent only reports findings, it doesn't rule)
        none — nothing found. The (unmodified) view continues to Triage as
               normal; this agent is invisible in the outcome.
        low  — general PII: email, phone, or a bare person/location entity
               (including spaCy's occasional NRP mistag for some names — see
               the honest-limitations note in pii.py). Redacted in place
               (e.g. "[REDACTED-EMAIL]"); the *scrubbed* text continues on to
               Triage/Adjudicator, so injection scoring still runs, just
               never on the raw PII.
        high — regulated data: SSN, a Luhn-valid credit card, a medical
               record number, diagnosis-disclosure phrasing, or a
               passport/driver's-license number. The Orchestrator bypasses
               injection scoring *entirely* for these — Triage and the
               Adjudicator are never consulted — and routes straight to
               Policy for a block/review decision. This is deliberate: a
               regulated-data leak is a violation on its own terms,
               independent of whether the prompt also happens to be an
               attack, so there's no reason to pay for (or wait on) an
               injection verdict that wouldn't change the outcome.

    MESSAGE INTERFACE (A2A)
        REQUEST  {"text": <de-obfuscated prompt>}
        INFORM   {"sensitivity": "none"|"low"|"high", "pii_types": [...],
                  "redacted": <text with low-sensitivity spans masked>,
                  "pii_source": "ner"|"regex"}
    """
    name = "Privacy"

    def handle(self, msg, bus):
        text = msg.content["text"]
        result = scan_pii(text)
        return [self._msg(msg.sender, Performative.INFORM,
                          sensitivity=result.sensitivity, pii_types=result.types,
                          redacted=result.redacted, pii_source=result.source)]


# ---------------------------------------------------------------------------
class TriageAgent(Agent):
    """Fast ML tier-1 detector. Emits a calibrated P(attack) and top tokens."""
    name = "Triage"

    def __init__(self, tier1_model):
        self.model = tier1_model

    def handle(self, msg, bus):
        text = msg.content["text"]
        prob = float(self.model.predict_proba([text])[0, 1])
        tokens = self._explain(text)
        return [self._msg(msg.sender, Performative.INFORM, risk=round(prob, 3),
                          top_tokens=tokens)]

    def _explain(self, text):
        inner = self.model
        if hasattr(inner, "calibrated_classifiers_"):
            inner = inner.calibrated_classifiers_[0].estimator
        try:
            return top_tokens(inner, text)
        except Exception:
            return []


# ---------------------------------------------------------------------------
class AdjudicatorAgent(Agent):
    """Tier-2 deep judge. Consulted only when Triage is uncertain."""
    name = "Adjudicator"

    def handle(self, msg, bus):
        text = msg.content["text"]
        j = judge(text)  # LLM if enabled, else heuristic
        return [self._msg(msg.sender, Performative.INFORM, label=j.label,
                          confidence=round(j.score, 3), source=j.source,
                          rationale=j.rationale)]


# ---------------------------------------------------------------------------
class PolicyAgent(Agent):
    """Turns a verdict into an action, applying enterprise policy + user role."""
    name = "Policy"

    # higher-trust roles get a 'human review' step instead of a hard block on
    # borderline calls, so we don't wall off legitimate power users.
    REVIEW_ROLES = {"admin", "security_engineer"}

    def handle(self, msg, bus):
        pii_sensitivity = msg.content.get("pii_sensitivity", "none")
        pii_types = msg.content.get("pii_types", [])
        role = msg.content.get("role", "employee")
        if pii_sensitivity == "high":
            kinds = ", ".join(pii_types) or "regulated data"
            if role in self.REVIEW_ROLES:
                action = "review"
                reason = f"Regulated PII/PHI ({kinds}) detected -> human review for trusted role '{role}'."
            else:
                action = "block"
                reason = f"Regulated PII/PHI ({kinds}) detected -> blocked per data-loss-prevention policy."
            return [self._msg(msg.sender, Performative.DECIDE, action=action, reason=reason)]

        verdict = msg.content["verdict"]        # 1 = attack, 0 = benign
        confidence = msg.content.get("confidence", 1.0)
        if verdict == 0:
            action, reason = "allow", "No injection intent detected."
        elif confidence < 0.80 and role in self.REVIEW_ROLES:
            action, reason = "review", f"Borderline for trusted role '{role}' -> human review."
        else:
            action, reason = "block", "Injection/jailbreak intent -> blocked at gateway."
        return [self._msg(msg.sender, Performative.DECIDE, action=action, reason=reason)]


# ---------------------------------------------------------------------------
@dataclass
class Verdict:
    label: int
    action: str                       # allow | block | review
    risk: float
    decided_by: str                   # "Triage" | "Adjudicator" | "Privacy"
    reason: str
    transforms: list = field(default_factory=list)
    top_tokens: list = field(default_factory=list)
    trace: list = field(default_factory=list)   # list[AgentMessage]
    latency_ms: float = 0.0
    judge_source: str | None = None
    pii_sensitivity: str = "none"     # "none" | "low" | "high"
    pii_types: list = field(default_factory=list)
    pii_source: str = "regex"         # "ner" | "regex" — which Privacy detection path ran


class OrchestratorAgent(Agent):
    """Coordinates the multi-agent workflow via A2A messages and rules on the result.

    Flow:
        Orchestrator -> Forensics : REQUEST  (normalise)
        Forensics    -> Orchestrator: INFORM (transforms)
        Orchestrator -> Privacy   : REQUEST  (PII/PHI scan)
        Privacy      -> Orchestrator: INFORM (sensitivity, pii_types)
        [if pii sensitivity == "high"]
        Orchestrator -> Policy     : REQUEST (decide)      # bypasses injection scoring
        Policy       -> Orchestrator: DECIDE (action)
        [else]
        Orchestrator -> Triage    : REQUEST  (classify; low-sensitivity view is redacted)
        Triage       -> Orchestrator: INFORM (risk)
        [if uncertain]
        Orchestrator -> Adjudicator: ESCALATE
        Adjudicator  -> Orchestrator: INFORM (label)
        Orchestrator -> Policy     : REQUEST (decide)
        Policy       -> Orchestrator: DECIDE (action)
    """
    name = "Orchestrator"

    def __init__(self, bus: MessageBus, low=UNCERTAIN_LOW, high=UNCERTAIN_HIGH, audit=None):
        self.bus = bus
        self.low, self.high = low, high
        self.audit = audit  # optional AuditLog; set at the gateway, off during eval

    def analyze(self, text: str, role: str = "employee") -> Verdict:
        import time
        t0 = time.perf_counter()
        self.bus.reset()

        # 1. Forensics
        fx = self.bus.send(self._msg("Forensics", Performative.REQUEST, text=text))[0]
        transforms = fx.content["transforms"]
        view = fx.content["normalized"] if transforms else text

        # 2. Privacy — PII/PHI scan on the de-obfuscated view, before injection scoring
        priv = self.bus.send(self._msg("Privacy", Performative.REQUEST, text=view))[0]
        pii_sensitivity = priv.content["sensitivity"]
        pii_types = priv.content["pii_types"]
        pii_source = priv.content["pii_source"]

        if pii_sensitivity == "high":
            # regulated data present -> bypass injection scoring entirely; this is a
            # DLP violation regardless of whether the prompt is also an attack
            pol = self.bus.send(self._msg("Policy", Performative.REQUEST,
                                          pii_sensitivity=pii_sensitivity,
                                          pii_types=pii_types, role=role))[0]
            verdict = Verdict(
                label=1,
                action=pol.content["action"],
                risk=1.0,
                decided_by="Privacy",
                reason=pol.content["reason"],
                transforms=transforms,
                trace=list(self.bus.trace),
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                pii_sensitivity=pii_sensitivity,
                pii_types=pii_types,
                pii_source=pii_source,
            )
            if self.audit is not None:
                self.audit.record(verdict, text, role)
            return verdict

        # low-sensitivity PII is redacted in place; the (now-scrubbed) view continues
        if pii_sensitivity == "low":
            view = priv.content["redacted"]

        # 3. Triage (classify the de-obfuscated, PII-scrubbed view)
        tri = self.bus.send(self._msg("Triage", Performative.REQUEST, text=view))[0]
        risk = tri.content["risk"]
        tokens = tri.content["top_tokens"]

        # 4. Decide directly, or escalate the uncertain middle band
        if risk < self.low:
            label, decided_by, jsrc, confidence = 0, "Triage", None, 1 - risk
            escalate_reason = f"Confidently benign (risk={risk:.2f} < {self.low})."
        elif risk > self.high:
            label, decided_by, jsrc, confidence = 1, "Triage", None, risk
            escalate_reason = f"Confidently malicious (risk={risk:.2f} > {self.high})."
        else:
            adj = self.bus.send(self._msg("Adjudicator", Performative.ESCALATE, text=text,
                                          risk=risk))[0]
            label = adj.content["label"]
            confidence = adj.content["confidence"]
            jsrc = adj.content["source"]
            decided_by = "Adjudicator"
            escalate_reason = (f"Uncertain at tier-1 (risk={risk:.2f}); "
                               f"Adjudicator[{jsrc}] -> {'ATTACK' if label else 'BENIGN'}.")

        # 5. Policy
        pol = self.bus.send(self._msg("Policy", Performative.REQUEST, verdict=label,
                                      confidence=confidence, role=role,
                                      pii_sensitivity=pii_sensitivity))[0]

        reason = f"{escalate_reason} {pol.content['reason']}"
        if pii_sensitivity == "low":
            reason += f" (PII redacted before scoring: {', '.join(pii_types)}.)"

        verdict = Verdict(
            label=label,
            action=pol.content["action"],
            risk=risk,
            decided_by=decided_by,
            reason=reason,
            transforms=transforms,
            top_tokens=tokens,
            trace=list(self.bus.trace),
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            judge_source=jsrc,
            pii_sensitivity=pii_sensitivity,
            pii_types=pii_types,
            pii_source=pii_source,
        )
        # persist the decision + full A2A trace for auditability
        if self.audit is not None:
            self.audit.record(verdict, text, role)
        return verdict

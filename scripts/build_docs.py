"""Generate a single self-contained HTML documentation file for the codebase.

Reads every source module directly and embeds it with build-time syntax
highlighting (via Python's own `tokenize`, so the highlighting is always correct
and the snippets always match the real code). Prose is authored here.

    python3 scripts/build_docs.py    ->    docs/documentation.html
"""
from __future__ import annotations

import html
import io
import json
import keyword
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "documentation.html"

# ---------------------------------------------------------------------------
# Build-time Python syntax highlighter (correct by construction: uses tokenize)
# ---------------------------------------------------------------------------
_BUILTINS = {"self", "cls", "True", "False", "None"}


def _seg(lines, start, end):
    (sr, sc), (er, ec) = start, end
    if sr == er:
        return lines[sr - 1][sc:ec]
    parts = [lines[sr - 1][sc:]]
    for r in range(sr + 1, er):
        parts.append(lines[r - 1])
    parts.append(lines[er - 1][:ec])
    return "\n".join(parts)


def highlight(src: str) -> str:
    lines = src.split("\n")
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError):
        return html.escape(src)
    out, prev = [], (1, 0)
    for tok in toks:
        ttype, tstr, start, end, _ = tok
        out.append(html.escape(_seg(lines, prev, start)))
        prev = end
        if not tstr:
            continue
        cls = None
        if ttype == tokenize.COMMENT:
            cls = "c-com"
        elif ttype == tokenize.STRING or ttype == getattr(tokenize, "FSTRING_START", -99):
            cls = "c-str"
        elif ttype == tokenize.NUMBER:
            cls = "c-num"
        elif ttype == tokenize.NAME:
            if keyword.iskeyword(tstr):
                cls = "c-kw"
            elif tstr in _BUILTINS:
                cls = "c-self"
        esc = html.escape(tstr)
        out.append(f'<span class="{cls}">{esc}</span>' if cls else esc)
    return "".join(out)


def code_block(path: Path) -> str:
    src = path.read_text()
    n = src.count("\n") + 1
    nums = "\n".join(str(i) for i in range(1, n + 1))
    return (f'<div class="code"><div class="code-head"><span class="fn">{path.name}</span>'
            f'<span class="ln-count">{n} lines</span></div>'
            f'<div class="code-body"><pre class="gutter">{nums}</pre>'
            f'<pre class="src">{highlight(src)}</pre></div></div>')


def p(*paras: str) -> str:
    return "".join(f"<p>{t}</p>" for t in paras)


def ARCH_SVG() -> str:
    # Hub-and-spoke, matching orchestrator.analyze(): every agent replies ONLY to the
    # Orchestrator (double-headed edges); agents never call each other. Policy's DECIDE
    # reply is what the Orchestrator turns into the final allow/review/block.
    return """
<div class="diagram"><svg viewBox="0 0 820 250" role="img" aria-label="A2A architecture — hub and spoke">
  <defs>
    <marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 Z" fill="var(--mut)"/></marker>
    <marker id="a2" markerWidth="9" markerHeight="9" refX="2" refY="3" orient="auto">
      <path d="M7,0 L0,3 L7,6 Z" fill="var(--mut)"/></marker>
  </defs>
  <rect x="20" y="100" width="140" height="46" rx="8" class="box b-orch"/>
  <text x="90" y="128" class="bx">Orchestrator</text>
  <rect x="320" y="20"  width="120" height="46" rx="8" class="box"/><text x="380" y="48" class="bx">Forensics</text>
  <rect x="320" y="100" width="120" height="46" rx="8" class="box b-ml"/><text x="380" y="128" class="bx">Triage (ML)</text>
  <rect x="320" y="180" width="120" height="46" rx="8" class="box"/><text x="380" y="208" class="bx">Adjudicator</text>
  <rect x="560" y="100" width="120" height="46" rx="8" class="box b-pol"/><text x="620" y="128" class="bx">Policy</text>
  <rect x="700" y="100" width="100" height="46" rx="8" class="box b-out"/><text x="750" y="128" class="bx">allow / review / block</text>
  <!-- every edge is double-headed: dispatch out from Orchestrator, reply back in -->
  <line x1="160" y1="108" x2="320" y2="44"  class="arw" marker-end="url(#a)" marker-start="url(#a2)"/>
  <line x1="160" y1="123" x2="320" y2="123" class="arw" marker-end="url(#a)" marker-start="url(#a2)"/>
  <line x1="160" y1="138" x2="320" y2="200" class="arw" marker-end="url(#a)" marker-start="url(#a2)"/>
  <line x1="160" y1="128" x2="560" y2="123" class="arw" marker-end="url(#a)" marker-start="url(#a2)" stroke-dasharray="4 3"/>
  <line x1="680" y1="123" x2="700" y2="123" class="arw" marker-end="url(#a)"/>
  <text x="230" y="16"  class="lbl">REQUEST</text>
  <text x="230" y="207" class="lbl">ESCALATE (uncertain band)</text>
  <text x="290" y="118" class="lbl">REQUEST → Policy (dashed: routed by Orchestrator, not Triage)</text>
</svg></div>
<p class=note>Every agent connects <b>only</b> to the Orchestrator — REQUEST/ESCALATE out,
INFORM/DECIDE back. Agents never call each other directly, and the Orchestrator itself builds
the final <code>Verdict</code> after Policy's reply. See <code>OrchestratorAgent.analyze()</code>
in <code>agents.py</code> below.</p>"""


# ---------------------------------------------------------------------------
# Documentation content: (anchor, title, prose_html, [source files])
# ---------------------------------------------------------------------------
G = ROOT / "guardrail"
S = ROOT / "scripts"

SECTIONS = [
    ("overview", "1 · Overview", p(
        "<b>Guardrail</b> is a multi-agent system that screens prompts arriving at an enterprise "
        "LLM gateway and blocks <b>prompt-injection</b> and <b>jailbreak</b> attempts, while emitting "
        "an auditable <b>agent-to-agent (A2A)</b> deliberation trace for every decision.",
        "It is built as a full ML system in four movements: "
        "<b>synthetic data → model bake-off → multi-agent cascade → formal evaluation</b>. "
        "The entire project depends only on scikit-learn and the Python standard library, and the "
        "live demo runs fully offline — no web framework, no API key, no network.",
        "<b>Problem, formally.</b> Binary classification: given a prompt (optionally with pasted "
        "context), predict <span class=k>attack</span> vs <span class=k>benign</span> at gateway "
        "latency. <b>Success metric:</b> recall (catch-rate) on <i>unseen</i> attacks at a fixed low "
        "false-positive rate — blocking a legitimate employee is expensive, so a false negative is "
        "weighted 3× a false positive.",
        "<b>Presentation materials</b> (not covered further in this document): "
        "<code>Guardrail_PANW_Deck_Condensed.pptx</code> — an 11-slide condensed deck, each "
        "technical slide carrying a measured Design-Choice callout, full depth preserved in "
        "Appendix A–Z; <code>SPEAKER_SCRIPT.md</code> and <code>PRESENTER_HANDBOOK.html</code> — "
        "word-for-word narration and a Q&amp;A bank for that deck; "
        "<code>DEMO_CHOREOGRAPHY.md</code> — the exact live-demo prompt sequence."), []),

    ("architecture", "2 · Architecture", p(
        "A request flows through five single-responsibility agents coordinated by an "
        "<b>Orchestrator</b>. Each hop is a typed, logged A2A message. The core is a <b>cascade</b>: "
        "the cheap ML <b>Triage</b> agent answers the easy majority in ~1&nbsp;ms; only the uncertain "
        "middle band escalates to the slower <b>Adjudicator</b>. This buys most of the judge's "
        "accuracy at a fraction of its latency and cost.") + ARCH_SVG(), []),

    ("config", "3 · Configuration", p(
        "Central paths, the random seed (everything is reproducible), the cascade's asymmetric "
        "escalation band, and the cost model used for threshold selection. The band is deliberately "
        "asymmetric — the &ldquo;confidently benign&rdquo; region is wide because false positives are "
        "costly."), [G / "config.py"]),

    ("data", "4 · Synthetic Data Generation", p(
        "Real labelled injection attacks are scarce, sensitive, and evolve daily, so we manufacture a "
        "labelled corpus from two grammars: <b>benign</b> enterprise prompts across eight personas, "
        "and <b>attack</b> prompts from seven injection/jailbreak families seeded from public "
        "taxonomies, mutated by 10 obfuscation transforms (leetspeak, spacing, base64, uppercase, "
        "URL/percent-encoding, hex escapes, HTML entities, ROT13, Unicode fullwidth forms, and "
        "zero-width character insertion).",
        "The credibility of the whole project rests on two deliberately hard sets: <b>hard negatives</b> "
        "— benign prompts that legitimately use sensitive vocabulary (&ldquo;summarise our "
        "<i>password</i> rotation policy&rdquo;) — and <b>look-alikes</b> — innocent uses of attack "
        "trigger-words (&ldquo;<i>ignore</i> my last message, I meant Q2&rdquo;). Without these a model "
        "just keyword-spots and scores a meaningless 1.0; with them it must learn <i>intent</i>.",
        "Fidelity is validated: exact-duplicate removal, class-balance and length reports, and a hard "
        "train/test <b>leakage assertion</b>. The split reserves an <b>adversarial holdout</b> of whole "
        "attack families and obfuscations that the model never sees in training — the only honest test "
        "of generalisation to novel attacks. A small amount of label noise is injected into training "
        "only, to reflect real annotator disagreement."), [G / "data_gen.py"]),

    ("features", "5 · Features", p(
        "Two complementary text views. <b>Word n-grams</b> capture intent "
        "(&ldquo;ignore previous instructions&rdquo;); <b>character n-grams</b> (3–5) survive "
        "obfuscation, because they still overlap after leetspeak/spacing/casing. The primary model "
        "unions both, so it reads intent <i>and</i> resists evasion."), [G / "features.py"]),

    ("models", "6 · Model Bake-off", p(
        "Four candidates are benchmarked so model selection is grounded in measured trade-offs: a "
        "transparent <b>keyword baseline</b> (the honest floor), word-only and char-only Logistic "
        "Regression, and the deployed <b>union + calibrated Logistic Regression</b>. Calibration "
        "matters because the cascade's escalation band is defined in probability space. Heavier options "
        "(sentence-embeddings, fine-tuned DistilBERT, an LLM judge) are discussed but kept off the hot "
        "path for latency/cost/opacity reasons. A token-level interpretability helper explains any "
        "single prediction for the live demo."), [G / "models.py"]),

    ("a2a", "7 · A2A Protocol", p(
        "A lightweight agent-to-agent messaging layer modelled on FIPA ACL performatives and the shape "
        "of Google A2A / Anthropic MCP tool-messages: every interaction is a typed, addressed, "
        "<b>logged</b> speech-act rather than a hidden function call. For a security control, "
        "auditability is the entire point — and the recorded trace drives the demo's deliberation view."),
     [G / "a2a.py"]),

    ("agents", "8 · The Agents", p(
        "Six agents, each owning exactly one responsibility and communicating only via A2A messages. "
        "<b>Forensics</b> de-obfuscates the input and reports the transforms it found; <b>Privacy</b> "
        "is a data-loss-prevention (PII/PHI) check run right after it and before injection scoring "
        "&mdash; a concern distinct from attack intent (see below); <b>Triage</b> is "
        "the fast ML tier-1 detector; <b>Adjudicator</b> is the tier-2 deep judge consulted only for "
        "the uncertain band; <b>Policy</b> maps a verdict (or PII sensitivity) plus the user's role to "
        "a concrete action (allow / block / human-review); the <b>Orchestrator</b> runs the workflow "
        "and issues the ruling. Because each agent is isolated, the heuristic Adjudicator can be "
        "swapped for a real LLM guard model without touching anyone else."), [G / "agents.py"]),

    ("privacy", "8b · PII/PHI Data-Loss-Prevention", p(
        "A prompt can be entirely benign &mdash; no injection intent at all &mdash; and still be a "
        "compliance incident: an employee pasting a patient's SSN and diagnosis into a prompt is not "
        "an attack, but it is a data-loss event a real enterprise gateway must catch. The "
        "<b>Privacy</b> agent (<code>guardrail/pii.py</code>) is <b>NER-primary</b>: a real named-"
        "entity-recognition pass (Microsoft Presidio, backed by spaCy's small English model, no "
        "torch dependency) that recognizes the <i>shape</i> of an entity from context &mdash; a bare "
        "name (&ldquo;Priya Sharma&rdquo;), a street address &mdash; that no fixed regex can catch. "
        "Two custom recognizers are registered alongside spaCy's NER for entity types Presidio "
        "doesn't ship (medical record numbers, diagnosis-disclosure phrasing). If the NER engine "
        "isn't available for any reason, detection falls straight back to the original regex + "
        "keyword-context patterns &mdash; the exact same <code>_llm_judge() or _heuristic_judge()</code> "
        "cascade shape the tier-2 Adjudicator already uses, just applied to PII detection.",
        "<b>Two sensitivity tiers.</b> <i>Low</i> (email, phone, a bare person/location entity) is "
        "redacted in place and the request continues &mdash; Triage and the audit log only ever see "
        "the scrubbed view. <i>High</i> (SSN with a real placeholder-blocklist check, Luhn-validated "
        "credit card, medical record number, diagnosis-disclosure phrasing, passport/driver's-license "
        "numbers) bypasses injection scoring entirely: the Orchestrator routes straight to Policy for "
        "a block/review decision, because regulated-data exposure is a violation on its own terms, "
        "independent of whether the prompt also happens to be an attack.",
        "<b>Honest limitations.</b> This adds real but modest latency (~4&nbsp;ms/request measured "
        "over the regex-only baseline &mdash; ~13&nbsp;ms vs ~9&nbsp;ms). And the small spaCy model "
        "has a documented name-classification bias: verified directly, it tags &ldquo;Priya "
        "Sharma&rdquo; as <code>NRP</code> (nationality/religious/political group) rather than "
        "<code>PERSON</code>, while &ldquo;John Smith&rdquo; and &ldquo;Wei Zhang&rdquo; classify "
        "correctly. Both are still redacted here (<code>NRP</code> is bucketed as low-sensitivity "
        "PII too), which mitigates the practical impact without fixing the underlying model bias "
        "&mdash; a real deployment would want a name-detection benchmark across name origins before "
        "trusting this model's <code>PERSON</code> recall unevenly."),
     [G / "pii.py"]),

    ("orchestrator", "9 · Orchestrator Factory", p(
        "A small factory that loads the trained tier-1 model, registers the agents on a message bus, "
        "and returns a ready orchestrator. This is the single entry point the demo and tests use."),
     [G / "orchestrator.py"]),

    ("judge", "10 · Tier-2 Judge", p(
        "The Adjudicator's brain. In production this would call a guard LLM; for an offline demo it "
        "falls back to a deterministic heuristic that de-obfuscates the prompt across 10 disguise "
        "families (base64, spacing, leetspeak, uppercase, URL/percent-encoding, hex escapes, HTML "
        "entities, ROT13, Unicode fullwidth forms, zero-width insertion) plus a general mojibake/"
        "homoglyph safety net (<code>ftfy</code> + <code>unidecode</code>), then matches high-signal "
        "attack structure. Same interface either way — set <code>GUARDRAIL_LLM=1</code> (with an "
        "<code>ANTHROPIC_API_KEY</code>) to enable the real LLM path."), [G / "llm_judge.py"]),

    ("cascade", "11 · Cascade (evaluation core)", p(
        "The two-tier decision logic expressed as a single object. The multi-agent Orchestrator is this "
        "same logic decomposed into collaborating agents; the plain <code>Cascade</code> is retained "
        "because the evaluation harness uses it to score models quickly."), [G / "cascade.py"]),

    ("audit", "12 · Audit Log", p(
        "An append-only audit trail. For a security control, &ldquo;auditable by design&rdquo; has to "
        "mean decisions are actually recorded, not just shown in a UI. Every decision — verdict plus the "
        "full A2A deliberation trace — is written as one JSON line to <code>logs/audit.jsonl</code> (the "
        "standard shape for SIEM ingestion), with a timestamp, request id, and a SHA-256 of the prompt "
        "(or a redaction), so any block can be replayed and explained months later. The gateway enables "
        "it; evaluation runs with it off so the trail isn&rsquo;t flooded."), [G / "audit.py"]),

    ("evaluate", "13 · Evaluation Framework", p(
        "Answers, with numbers: how the candidates compare; how well the system generalises to attack "
        "families never seen in training (the headline robustness result); what synthetic augmentation "
        "actually buys (an ablation); and where to set the operating threshold given asymmetric costs. "
        "Metrics are security-aware — recall at a fixed low false-positive rate, PR-AUC, and per-family "
        "recall for error analysis."), [G / "evaluate.py"]),

    ("benchmark", "14 · Benchmark &amp; Model-Selection Evidence", p(
        "Model selection is grounded in measurement, not assertion. <b>Eleven candidates</b> — the "
        "keyword floor; word / char / union Logistic Regression (the deployed model is the calibrated "
        "union); Linear SVM; ComplementNB; SGD; HistGBDT; a <b>from-scratch NumPy</b> logistic "
        "regression; plus two deep-learning candidates (<b>MiniLM sentence-embeddings</b> and a "
        "<b>frozen DistilBERT</b> feature-extractor) — are trained on the identical split and scored on "
        "the same adversarial holdout, with a real median single-request latency.",
        "<b>The result that matters:</b> the heavyweight models earn no measurable accuracy on this task "
        "while costing 5–150× the latency — the empirical basis for keeping them off the hot path, "
        "behind the escalation band. (Full fine-tuning of DistilBERT was attempted but is impractical on "
        "CPU here, which only reinforces the point; it is recorded as such.)",
        "<b>Is the union really better than word-only?</b> On the benchmark it isn&rsquo;t, quite — "
        "word-only <i>ties</i> it on adversarial recall. But only because the two held-out "
        "obfuscations (base64, zero-width) still leave enough intact word tokens (or an English "
        "wrapper, for base64: &ldquo;Decode this base64&hellip;&rdquo;) that word features catch for free. "
        "So a deeper, fair test (<code>deep_compare.py</code>) calibrates "
        "every candidate and adds a <b>char-stress holdout</b>: obfuscations that <i>destroy word "
        "tokens</i> — space-removal and inner-char swaps, never seen in training, no wrapper.",
        "<b>Two honest findings.</b> (1) <b>Calibration is not the differentiator</b> — a fairly "
        "calibrated word-only model reaches the same ECE, so that argument is dropped. (2) "
        "<b>Robustness is</b> — when obfuscation destroys word tokens, word-only&rsquo;s "
        "<b>false-positive rate explodes to 46%</b> (it blocks nearly half of benign de-spaced text) "
        "while the union holds <b>&lt;1%</b> and still recalls 0.97. That gap is the precision-first thesis, "
        "measured — and it is why the calibrated union is deployed."),
     [S / "run_benchmark.py", S / "deep_compare.py"]),

    ("serve", "15 · Live Demo Server", p(
        "A dependency-free web app built on <code>http.server</code>. It serves a single page that "
        "calls the orchestrator and renders the A2A trace live. No Flask, no Streamlit, no network — "
        "safe to run in an interview with the Wi-Fi off. Port is configurable to avoid collisions."),
     [ROOT / "serve.py"]),

    ("scripts", "16 · Scripts", p(
        "Three thin entry points that wire the pieces together: generate the data (with a fidelity "
        "report), train and save the cascade, and run the full evaluation with plots."),
     [S / "make_data.py", S / "train.py", S / "run_eval.py"]),
]


def results_table() -> str:
    mp = ROOT / "artifacts" / "metrics.json"
    if not mp.exists():
        return "<p class=note>Run <code>python3 scripts/run_eval.py</code> to populate results.</p>"
    m = json.loads(mp.read_text())
    c = m["primary"]["cascade_end_to_end"]
    st, adv = c["standard_test"], c["adversarial_holdout"]
    abl = m["ablation"]
    rows = "".join(
        f"<tr><td>{fam}</td><td>{v['n']}</td><td>{v['recall']:.3f}</td></tr>"
        for fam, v in m["per_family_adversarial_recall"].items())
    return f"""
<table class=tbl>
<tr><th>Slice</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
<tr><td>Standard test (in-distribution)</td><td>{st['precision']:.2f}</td><td>{st['recall']:.2f}</td><td>{st['f1']:.2f}</td></tr>
<tr><td><b>Adversarial holdout</b> (families never seen in training)</td><td>{adv['precision']:.2f}</td><td>{adv['recall']:.2f}</td><td>{adv['f1']:.2f}</td></tr>
</table>
<p><b>Ablation — value of synthetic augmentation:</b> novel-attack recall
{abl['without_augmentation_adv_recall']:.2f} → {abl['with_augmentation_adv_recall']:.2f}.</p>
<p><b>Per-family recall on unseen attacks</b> (error analysis):</p>
<table class=tbl><tr><th>Attack family</th><th>n</th><th>Recall</th></tr>{rows}</table>"""


def _b64_png(path: Path) -> str:
    import base64
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def benchmark_tables() -> str:
    """Render the 11-model benchmark and the union-vs-word deep comparison from JSON."""
    bp = ROOT / "artifacts" / "benchmark.json"
    dp = ROOT / "artifacts" / "deep_compare.json"
    out = []
    if bp.exists():
        b = json.loads(bp.read_text())
        models = b["models"]
        order = sorted(models, key=lambda n: models[n].get("adversarial_holdout", {}).get("recall", -1), reverse=True)
        rows = ""
        for n in order:
            r = models[n]
            if "adversarial_holdout" not in r:
                continue
            rows += (f"<tr><td>{html.escape(n)}</td><td>{r['family']}</td>"
                     f"<td>{r['standard_test']['f1']:.3f}</td>"
                     f"<td>{r['adversarial_holdout']['recall']:.3f}</td>"
                     f"<td>{r['latency_ms']:g}</td></tr>")
        ds = b.get("dataset", {})
        out.append(f"<p><b>Benchmark — 11 candidates</b> (identical split: {ds.get('n_train')} train / "
                   f"{ds.get('n_test')} test / {ds.get('n_adversarial')} adversarial holdout; ranked by "
                   f"recall on unseen attacks):</p>")
        out.append("<table class=tbl><tr><th>Model</th><th>Family</th><th>std F1</th>"
                   f"<th>adv recall</th><th>latency ms</th></tr>{rows}</table>")
        png = ROOT / "artifacts" / "benchmark.png"
        if png.exists():
            out.append(f'<p><img alt="accuracy vs latency across 11 models" '
                       f'style="max-width:100%;border:1px solid var(--line);border-radius:10px;margin-top:8px" '
                       f'src="{_b64_png(png)}"></p>')
    if dp.exists():
        d = json.loads(dp.read_text())
        m = d["models"]
        rows = ""
        for n in ("word (calibrated)", "char (calibrated)", "union (calibrated) ★"):
            if n not in m:
                continue
            r = m[n]
            fpr_cls = ' style="color:var(--kw);font-weight:700"' if r["charstress_fpr"] >= 0.2 else ' style="color:var(--acc);font-weight:700"'
            rows += (f"<tr><td>{html.escape(n)}</td><td>{r['std_f1']:.3f}</td><td>{r['ece']:.3f}</td>"
                     f"<td>{r['charstress_recall']:.2f}</td><td{fpr_cls}>{r['charstress_fpr']:.2f}</td></tr>")
        cs = d.get("dataset", {}).get("char_stress")
        out.append(f"<p style='margin-top:18px'><b>Deep comparison — is the union really better than "
                   f"word-only?</b> Every candidate calibrated fairly, then scored on a <b>char-stress "
                   f"holdout</b> (n={cs}) of obfuscations that destroy word tokens (space-removal + "
                   f"inner-char swaps, not in training):</p>")
        out.append("<table class=tbl><tr><th>Model</th><th>std F1</th><th>ECE&darr;</th>"
                   f"<th>char-stress recall</th><th>char-stress FPR</th></tr>{rows}</table>")
        wc = m.get("word (calibrated)", {})
        uc = m.get("union (calibrated) ★", {})
        out.append("<p class=note><b>Two findings.</b> Calibration is <i>not</i> the differentiator "
                   "(word-only reaches the same ECE). The real edge is robustness: when obfuscation "
                   f"destroys word tokens, word-only&rsquo;s false-positive rate hits "
                   f"<b>{wc.get('charstress_fpr', 0):.0%}</b> while the "
                   f"union holds <b>{uc.get('charstress_fpr', 0):.0%}</b> and still recalls "
                   f"{uc.get('charstress_recall', 0):.2f}. Reproduce with "
                   "<code>python3 scripts/deep_compare.py</code>.</p>")
    return "".join(out) or "<p class=note>Run <code>scripts/run_benchmark.py</code> and <code>scripts/deep_compare.py</code> to populate.</p>"


def build() -> str:
    toc = "".join(f'<a href="#{a}">{t}</a>' for a, t, _, _ in SECTIONS)
    body = []
    for anchor, title, prose, files in SECTIONS:
        blocks = "".join(code_block(f) for f in files)
        if anchor == "evaluate":
            extra = results_table()
        elif anchor == "benchmark":
            extra = benchmark_tables()
        else:
            extra = ""
        body.append(f'<section id="{anchor}"><h2>{title}</h2>{prose}{extra}{blocks}</section>')
    return TEMPLATE.replace("__TOC__", toc).replace("__BODY__", "".join(body))


TEMPLATE = r"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Guardrail — Codebase Documentation</title>
<style>
:root{--bg:#ffffff;--panel:#f6f8fa;--card:#ffffff;--line:#d0d7de;--fg:#1f2328;--mut:#656d76;
--acc:#0969da;--kw:#cf222e;--str:#0a3069;--com:#6e7781;--num:#0550ae;--self:#953800;
--box:#eaeef2;--boxln:#afb8c1;}
:root:not([data-theme=light]) @media (prefers-color-scheme:dark){}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){
--bg:#0d1117;--panel:#161b22;--card:#161b22;--line:#30363d;--fg:#e6edf3;--mut:#8b949e;
--acc:#58a6ff;--kw:#ff7b72;--str:#a5d6ff;--com:#8b949e;--num:#79c0ff;--self:#ffa657;
--box:#1c2333;--boxln:#3d444d;}}
:root[data-theme=dark]{--bg:#0d1117;--panel:#161b22;--card:#161b22;--line:#30363d;--fg:#e6edf3;
--mut:#8b949e;--acc:#58a6ff;--kw:#ff7b72;--str:#a5d6ff;--com:#8b949e;--num:#79c0ff;--self:#ffa657;
--box:#1c2333;--boxln:#3d444d;}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.65 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.layout{display:grid;grid-template-columns:260px 1fr;max-width:1240px;margin:0 auto}
nav{position:sticky;top:0;align-self:start;height:100vh;overflow:auto;padding:26px 16px;
border-right:1px solid var(--line)}
nav .brand{font-weight:700;font-size:17px;margin:0 0 4px}
nav .tag{color:var(--mut);font-size:12px;margin:0 0 18px}
nav a{display:block;color:var(--mut);text-decoration:none;font-size:13.5px;padding:5px 8px;border-radius:6px}
nav a:hover{background:var(--panel);color:var(--fg)}
main{padding:34px 40px 90px;min-width:0}
header.hero{border-bottom:1px solid var(--line);padding-bottom:22px;margin-bottom:10px}
header.hero h1{font-size:28px;margin:0 0 6px}
header.hero p{color:var(--mut);margin:0;max-width:70ch}
.pill{display:inline-block;font-size:11.5px;color:var(--acc);border:1px solid var(--line);
border-radius:999px;padding:2px 10px;margin:10px 6px 0 0}
section{padding:26px 0;border-bottom:1px solid var(--line)}
h2{font-size:20px;margin:0 0 12px}
p{margin:0 0 12px}p .k,span.k,.note code,p code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
font-size:.9em;background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:1px 5px}
i{color:var(--fg)}
.diagram{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;
margin:6px 0 4px;overflow-x:auto}
.box{fill:var(--box);stroke:var(--boxln)}.b-orch{fill:rgba(88,166,255,.18);stroke:var(--acc)}
.b-ml{fill:rgba(63,185,80,.16);stroke:#3fb950}.b-pol{fill:rgba(210,153,34,.16);stroke:#d29922}
.b-out{fill:rgba(248,81,73,.14);stroke:#f85149}
.bx{fill:var(--fg);font-size:13px;font-weight:600;text-anchor:middle}
.arw{stroke:var(--mut);stroke-width:1.5}.lbl{fill:var(--mut);font-size:11px;text-anchor:middle}
.tbl{width:100%;border-collapse:collapse;margin:8px 0 16px;font-size:14px}
.tbl th,.tbl td{border:1px solid var(--line);padding:7px 10px;text-align:left}
.tbl th{background:var(--panel)}
.code{border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:14px 0}
.code-head{display:flex;justify-content:space-between;background:var(--panel);
border-bottom:1px solid var(--line);padding:7px 12px;font-size:12.5px}
.code-head .fn{font-family:ui-monospace,Menlo,monospace;font-weight:600;color:var(--acc)}
.code-head .ln-count{color:var(--mut)}
.code-body{display:flex;overflow-x:auto;background:var(--card)}
pre{margin:0;font:12.5px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
pre.gutter{padding:12px 10px;color:var(--mut);text-align:right;user-select:none;
border-right:1px solid var(--line);background:var(--panel)}
pre.src{padding:12px 14px;white-space:pre;flex:1}
.c-kw{color:var(--kw)}.c-str{color:var(--str)}.c-com{color:var(--com);font-style:italic}
.c-num{color:var(--num)}.c-self{color:var(--self)}
.note{color:var(--mut)}
@media(max-width:820px){.layout{grid-template-columns:1fr}nav{position:static;height:auto;
border-right:none;border-bottom:1px solid var(--line)}main{padding:24px 18px 70px}}
</style></head><body>
<div class=layout>
<nav>
  <p class=brand>🛡️ Guardrail</p>
  <p class=tag>Codebase Documentation</p>
  __TOC__
</nav>
<main>
<header class=hero>
  <h1>Guardrail — Codebase Documentation</h1>
  <p>A multi-agent prompt-injection &amp; jailbreak defense for an enterprise LLM gateway.
     Synthetic data → model bake-off → multi-agent A2A cascade → formal evaluation.</p>
  <div>
    <span class=pill>Python · scikit-learn · stdlib only</span>
    <span class=pill>5 agents over an A2A protocol</span>
    <span class=pill>100% offline live demo</span>
  </div>
</header>
__BODY__
</main>
</div></body></html>"""


if __name__ == "__main__":
    OUT.write_text(build())
    kb = OUT.stat().st_size / 1024
    print(f"Wrote {OUT}  ({kb:.0f} KB)")

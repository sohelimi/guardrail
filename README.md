# 🛡️ Guardrail — Multi-Agent Prompt-Injection & Jailbreak Defense

A compact, **fully-offline, live-demoable** reference project: a multi-agent system
that screens prompts hitting an enterprise LLM gateway and blocks prompt-injection
/ jailbreak attempts — with an auditable **agent-to-agent (A2A)** deliberation trace.

Built as an end-to-end ML system: **synthetic data → model bake-off → multi-agent
cascade → formal evaluation.**

---

## The problem (formally)

**Task:** binary classification — given a user prompt (optionally with pasted
context), predict `attack` (prompt-injection / jailbreak) vs `benign`, at gateway
latency.

**Success metric:** *recall (catch-rate) on unseen attacks at a fixed, very low
false-positive rate.* Blocking a legitimate employee is expensive, so precision is
protected first; the cost model weights a false negative 3× a false positive.

## Why synthetic data

Real labelled injection attacks are scarce, sensitive, and evolve daily. We
manufacture a labelled corpus from two grammars (`guardrail/data_gen.py`):

- **Benign** — enterprise prompts across 11 personas (HR, eng, finance, legal, security, product, …).
- **Attack** — 9 injection/jailbreak families seeded from public taxonomies
  (OWASP LLM Top-10 style), mutated with **obfuscation** transforms
  (leetspeak, spacing, base64, uppercase).

The credibility hinges on two deliberately hard sets:
- **Hard negatives** — benign prompts that *use* sensitive vocabulary legitimately
  (“summarise our **password** rotation policy”, “write a **phishing**-awareness email”).
- **Look-alikes** — innocent uses of attack trigger-words (“**ignore** my last message,
  I meant Q2”). These force the model to learn *intent*, not keyword-spotting.

**Fidelity validation:** exact/near-dup removal, class-balance + length reports, and a
hard train/test leakage check (`make_data.py`).

## Architecture — 5 agents over an A2A protocol

```
prompt ─▶ Orchestrator ─REQUEST─▶ Forensics   (de-obfuscate; report transforms)
                        ─REQUEST─▶ Triage      (fast ML tier-1 → calibrated risk)
              risk<0.30 → allow │ risk>0.80 → block │ else ↓
                        ─ESCALATE▶ Adjudicator (tier-2 deep judge; LLM-optional)
                        ─REQUEST─▶ Policy       (role-based action: allow/block/review)
```

- **Cascade rationale:** at gateway QPS you can’t afford an LLM call per request.
  Tier-1 (calibrated TF-IDF word+char Logistic Regression, ~1 ms) answers the easy
  majority; only the uncertain middle band pays for the slower Adjudicator.
- **A2A protocol** (`guardrail/a2a.py`): every step is a typed, addressed, **logged**
  message (FIPA-style `REQUEST/INFORM/ESCALATE/DECIDE`). For a security control,
  auditability is the point — and it drives the live demo trace.
- Each agent owns one job and is independently swappable (e.g. replace the heuristic
  Adjudicator with a real LLM guard model — set `GUARDRAIL_LLM=1` — touching no one else).
- **Audit log** (`guardrail/audit.py`): the gateway appends every decision — verdict +
  full A2A trace + timestamp + request id + prompt SHA-256 — to `logs/audit.jsonl`
  (JSONL, SIEM-ready; optional prompt redaction). View it live at `/audit`; evaluation
  runs with logging off so the trail isn't flooded.

## Model selection (bake-off, measured)

| Candidate | Role | Trade-off |
|---|---|---|
| Keyword rules | floor | transparent, but F1 ≈ 0.32 — fooled by look-alikes/obfuscation |
| TF-IDF(word)+LogReg | intent | fast, interpretable |
| TF-IDF(char)+LogReg | obfuscation-robust | char 3–5 grams survive leet/spacing |
| **Union + calibrated LogReg** | **tier-1 (deployed)** | reads intent *and* survives obfuscation; calibrated so the cascade band is meaningful |
| Embeddings / DistilBERT / LLM-judge | discussed | higher accuracy, but latency/cost/opacity → not the hot path |

## Results (see `artifacts/metrics.json`, plots)

| Slice | Precision | Recall | F1 |
|---|---|---|---|
| Standard test (in-distribution) | 1.00 | 0.98 | 0.99 |
| **Adversarial holdout** (attack families/obfuscations **never seen in training**) | 1.00 | 0.92 | 0.96 |

- **Zero false positives** across benign prompts including all hard-negatives/look-alikes.
- **Ablation:** synthetic augmentation lifts novel-attack recall **0.76 → 0.92**.
- **Per-family error analysis:** the low-signal families remain the gap
  (`indirect_injection` 0.83, `stealth_injection` 0.89) — honest, and the driver of “what I’d do differently”.

---

## Run it

```bash
cd Guardrail
python3 scripts/make_data.py     # generate synthetic corpus + fidelity report
python3 scripts/train.py         # train tier-1, save the cascade
python3 scripts/run_eval.py      # full evaluation → metrics.json + plots
python3 serve.py 8010            # live demo → http://localhost:8010
```

No third-party web framework; the demo is Python standard library only. Optional real
LLM Adjudicator: `GUARDRAIL_LLM=1` + `pip install anthropic` + `ANTHROPIC_API_KEY`.

## Layout

```
guardrail/
  data_gen.py     synthetic data grammar, obfuscation, hard-negatives, splits, fidelity
  features.py     word + char TF-IDF vectorizers
  models.py       bake-off candidates, calibrated primary, token-level interpretability
  a2a.py          A2A message protocol + bus (typed, logged messages)
  agents.py       Forensics / Triage / Adjudicator / Policy / Orchestrator
  orchestrator.py factory that wires the trained model into the agent system
  llm_judge.py    tier-2 judge (LLM-optional, heuristic fallback)
  evaluate.py     bake-off, adversarial holdout, per-family, ablation, cost-threshold
scripts/          make_data · train · run_eval
serve.py          dependency-free live demo (A2A trace UI)
```

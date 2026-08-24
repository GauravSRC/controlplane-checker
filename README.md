# ControlPlane Checker

> A **risk-adaptive verification mesh** that sits between any LLM and your application and
> spends verification compute **proportional to consequence**.
> _Accenture Innovation Challenge 2026 — Round 2 (Track: ControlPlane.ai)._

Not every AI response deserves the same scrutiny. Like airport security — everyone walks
through the metal detector (fast, cheap), only some get a pat-down, very few get their bag
opened — ControlPlane Checker runs a **three-tier cascade** and routes each response on
**confidence × blast radius**, never a single score.

> **Status:** scaffold — interfaces are stubbed pending implementation. Stated
> design assumptions are tracked in [`ASSUMPTIONS.md`](ASSUMPTIONS.md).

## Headline: zero-code-change deployability

The service exposes an **OpenAI-compatible `/v1/chat/completions`** endpoint. Any existing
app just points its `base_url` at ControlPlane Checker and is instantly governed.

## Architecture at a glance

- **Layer 0 REFLEX** (~10 ms) — free signals: logprobs, citation-ID validity, schema,
  PII/secret regex, budget counters.
- **Layer 1 INSPECT** (~80 ms) — guard model (safety/injection), NLI groundedness,
  self-consistency hallucination detection.
- **Layer 2 ADJUDICATE** (~1 s) — LLM-as-judge / human, with a pre-assembled evidence packet.
- **Router** — multi-label risk vector (hallucination/privacy/bias/safety/cost) ×
  blast radius (informational → advisory → side-effect → irreversible).
- **Evidence Ledger** — OpenTelemetry GenAI-semconv spans → Arize Phoenix, mapped to
  NIST AI RMF & OWASP LLM Top 10.
- **Policy-as-code** — hot-reloadable YAML packs (EU AI Act / India DPDP+RBI / US-healthcare).
- **Feedback loop** — human overrides recalibrate thresholds under a flag budget.

Success metric: **Cost per Verified Correct Outcome (CVCO)**, not cost per token.

## Quick start

```bash
cp .env.example .env          # add your GROQ_API_KEY
make setup                    # install deps + spaCy model (CPU-only)
make demo                     # single-command end-to-end demo for judges
```

Other targets: `make up` (proxy), `make console` (Streamlit), `make eval`, `make test`.
On Windows without `make`, run `python scripts/demo.py`.

## Tech stack

Python 3.11 · FastAPI · Pydantic · LiteLLM · Groq (Llama 3.1 8B + Llama Guard + Prompt
Guard) · sentence-transformers (NLI) · Presidio + spaCy · OpenTelemetry → Arize Phoenix ·
SQLite · Streamlit · scikit-learn · pytest. **CPU-only, no GPU, no training.**

## Data & safety

All data is **synthetic / illustrative** — no real enterprise or personal data. See
[`ASSUMPTIONS.md`](ASSUMPTIONS.md).

## License

[MIT](LICENSE).

# ASSUMPTIONS.md

> The Accenture Round-2 brief explicitly rewards **clearly stated assumptions**. Log every
> non-obvious design decision, scope boundary, and simplification here as we build. Keep
> entries short, dated, and honest about what is real vs. illustrative.

**Format:** one row per assumption. Status ∈ {ASSUMED, VALIDATED, REVISED, DROPPED}.

| # | Date | Area | Assumption / Decision | Rationale | Status |
|---|------|------|------------------------|-----------|--------|
| 1 | 2026-08-24 | Data | All datasets are **synthetic / illustrative**; no real enterprise or personal data is used. | Brief mandate + privacy. | ASSUMED |
| 2 | 2026-08-24 | Compute | Everything runs **CPU-only**; no GPU, no model training/fine-tuning. | Brief mandate + judge reproducibility. | ASSUMED |
| 3 | 2026-08-24 | Deploy | Single-command local run via `make demo` → `scripts/demo.py`; no Docker required. | Judges run on stock machines. | ASSUMED |
| 4 | 2026-08-24 | Providers | Groq (Anthropic/OpenAI-style API) is treated as a **Tier C** (text-only) provider by default; logprobs/hidden-states paths are exercised via capability probing, not assumed. | Matches "consume via API" constraint. | ASSUMED |

---

## Open questions / to confirm with the user

- [ ] Exact Groq model IDs for Prompt Guard (injection) — confirm the current
      Groq-hosted identifier.
- [ ] CVCO cost model inputs (₹ or $ per token, per human-review minute) for the demo.
- [ ] Which single use-case profile headlines the demo video (e.g. BFSI vs healthcare).

## Scope boundaries (explicitly OUT for the prototype)

- _TBD — record what we deliberately stub, mock, or defer here as we go._

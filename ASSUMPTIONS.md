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

## Provider / model availability (recorded 30 Aug 2026)

- **`llama-3.1-8b-instant` is retired on Groq.** The key in `.env` returns
  `model_not_found`. The governed model is now **`openai/gpt-oss-20b`**, chosen as the
  closest small open-weight substitute. It is config-driven
  (`CONTROLPLANE_MODEL` in `.env`, `controlplane/config.py`) — a one-line swap.
- **`meta-llama/llama-guard-4-12b` is also retired.** Layer 1 safety is therefore
  unresolved; `openai/gpt-oss-safeguard-20b` and `meta-llama/llama-prompt-guard-2-86m`
  are available on the key as candidates. Decision deferred.
- **`gpt-oss-20b` rejects the `logprobs` parameter**, so the live provider is
  **capability Tier C (black-box)**, not Tier B. This is the graceful-degradation path
  from CLAUDE.md §3.1 and it is now exercised by default:
  `sequence_confidence` returns `None` (not a fake 1.0), the detector sets
  `degraded=True`, and the router sees the signal as *unmeasured* rather than
  *confident*. The proxy also auto-degrades B→C and retries once if a provider rejects
  `logprobs`.
- **`gpt-oss-20b` is a reasoning model**: it spends `reasoning_tokens` before emitting
  content, and returned empty content at `max_tokens=20`. The proxy enforces a
  `max_tokens` floor of 512 (`Settings.max_tokens_floor`).
- **Environment is Python 3.12.3, not 3.11**, so the pins in `requirements.txt` do not
  all resolve as written. `litellm` was installed unpinned to unblock tonight.

## Detector calibration (illustrative, un-tuned)

- REFLEX signal scores (e.g. dangling citation = 0.55 + 0.15/id, Aadhaar = 0.90,
  secrets = 0.95) are **hand-set placeholders**, not fitted. The cost-weighted threshold
  optimizer (CLAUDE.md §3.4) is what should set them; until then every published FP/FN
  number must be labelled un-tuned.
- The router fuses the vector with **max-over-labels** at routing time only; the full
  vector is preserved on the `Decision`. The worst label governs the route.
- `PERSON_HINT_RE` is a two-capitalised-words heuristic for "a named person" and will
  over-fire on ordinary title-case text. It only ever *adds* a privacy signal alongside
  an already-dangling citation, so its blast radius is limited, but it is a known
  false-positive source pending Presidio NER in Layer 1.

## Guard models (verified live 30 Aug 2026)

Both IDs were confirmed with a live call before any code was written against them —
**no substitution was needed**:

- **`meta-llama/llama-prompt-guard-2-86m`** (injection/jailbreak). Returns a *bare
  probability as text*, not a label — e.g. `"0.999559"` for an injection vs
  `"0.000352"` for a benign question. Separation is excellent and it answers in
  ~130 ms warm, so it carries the latency story. Parsed by `parse_injection_score`.
- **`openai/gpt-oss-safeguard-20b`** (safety categories). **It is a chat model, not a
  classifier head** — asked a harmful question directly it *refuses* rather than
  classifying. It only emits structured labels when given an explicit policy prompt
  (`SAFEGUARD_POLICY`) and asked to classify an ASSISTANT RESPONSE. Verified emitting
  `safe/none`, `unsafe/S7` (privacy), `unsafe/S6` (specialized advice).

Design notes:
- Injection co-fires **safety + privacy** when the prompt targets a system prompt or
  credentials (`EXFIL_HINT_RE`) — an injection that extracts instructions is an
  exfiltration incident, not only a safety one.
- Safety category **S10 hate** additionally fires **bias**, and **S5 defamation** fires
  **privacy**, keeping the multi-label overlap story consistent across layers.
- Verdicts are cached on a SHA-256 of the model + text so repeated demo runs do not
  burn the 30 RPM free-tier quota. Cache hit measured at ~0 ms.
- Rate limits/5xx retry with exponential backoff + jitter, bounded by the caller's
  deadline. On exhaustion or timeout the detector returns a partial result with
  `degraded=True` and **never raises** (verified with a 1 ms timeout).

## Layer 1 orchestration and fail modes

- `detectors/inspect.py` was **renamed to `detectors/tier1.py`**. The old name shadows
  the stdlib `inspect` module that pydantic and pytest both import. `tests/test_scaffold.py`
  was updated to match.
- Layer 1 runs its detectors under **one shared budget** from the policy profile via
  `asyncio.wait`, so wall-clock is bounded by the slowest detector, not their sum.
- **Fail-open vs fail-closed is per route, and blast radius overrides profile
  preference:** a `SIDE_EFFECT`/`IRREVERSIBLE` route fails **closed** (risk floor 0.65,
  so the router cannot PASS an unverified high-consequence response) even when the
  profile sets `fail_open: true`. Verified.
- Cascade gate: Layer 0 always runs; Layer 1 is entered when composite risk >= 0.25
  (`escalate_at`, per profile) **OR** blast radius >= SIDE_EFFECT.

## Policy packs

- Shipped `eu_ai_act.yaml` and `india_dpdp_rbi.yaml`, each with the three required
  profiles (`customer_support_chat` 150 ms, `internal_copilot` 800 ms,
  `decision_support` 5000 ms) carrying `risk_appetite`, `latency_budget_ms`,
  `flag_budget_pct`, `fail_open`, and per-category thresholds.
- The earlier stub packs `eu-ai-act.yaml` and `india-dpdp.yaml` were deleted (superseded).
  `us-healthcare.yaml` remains a stub — out of scope for this pass.
- **Divergence is driven entirely by YAML, not code.** Verified on identical input:
  | input | EU | India |
  |---|---|---|
  | PAN + email leak @ ADVISORY | `REPAIR` (EU-GDPR-PII, repairable) | `BLOCK` (DPDP-SENSITIVE-ID, non-repairable) |
  | email only @ ADVISORY | `ANNOTATE` (no rule) | `REPAIR` (DPDP-S8-PII) |
  | financial action @ IRREVERSIBLE, risk 0.00 | `HOLD` | `BLOCK` (RBI-FREEAI-HUMAN-LOOP) |
- `RBI-FREEAI-HUMAN-LOOP` uses `min_composite: 0.0` deliberately: RBI FREE-AI expects a
  human decision-maker for customer-affecting financial actions *regardless of model
  confidence*, so it fires on any irreversible route.
- **Hot reload is mtime-based**, checked on each `resolve()` — a stat costs microseconds,
  so no watcher thread is needed. Verified live: editing a rule ID and latency budget
  changed proxy behaviour **without restarting the server**. A pack that fails Pydantic
  validation is rejected and the **last good version stays live** — a YAML typo must
  never take governance offline.

## Dependency pins

- `requirements.txt` was repinned for **Python 3.12** (the environment is 3.12.3; the
  previous pins targeted 3.11 and did not resolve). Versions come from an actual
  `pip-compile` resolution of the full graph and were then confirmed with a
  `pip install --dry-run` of the finished file: **105 packages, no conflicts**.
- Notable jumps: pydantic 2.8.2 -> 2.13.5, torch 2.3.1 -> 2.13.0, numpy 1.26.4 -> 2.4.6,
  transformers 4.43.3 -> 5.16.1, arize-phoenix 4.21.0 -> 20.3.0, pandas 2.2.2 -> 3.0.5.
  `openai` is now an explicit dependency (the smoke test uses it to prove the
  base_url-only integration claim).
- **Not yet re-verified against these new pins:** the CPU NLI/groundedness path
  (sentence-transformers 6.x + transformers 5.x) and Presidio 2.2.364, since those
  detectors are not implemented yet. Both are major-version jumps and may need API
  changes when Layer 1 groundedness is built.

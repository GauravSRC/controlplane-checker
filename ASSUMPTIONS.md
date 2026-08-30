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

## Designed but NOT implemented

Each of these is specified in the architecture and referenced by the code, but has no
working implementation. They are listed explicitly rather than left implied by the repo
structure, because empty modules that look like features are exactly the kind of
overstatement a governance tool must not make.

**NLI groundedness** (`detectors/groundedness.py`). Design: run a
`cross-encoder/nli-deberta-v3-small` entailment check of each response claim against the
retrieved evidence, giving a real GROUNDED verification mode instead of the citation-ID
proxy Layer 0 uses today. Scoped out because it needs `sentence-transformers` + `torch`
(~500 MB) plus per-claim segmentation, and the CPU latency at Layer 1 could not be
measured in the time available. Today groundedness is approximated by citation-ID
validity, which catches fabricated sources but NOT claims that contradict a real source.

**SelfCheckGPT semantic entropy** (`detectors/selfcheck.py`). Design: resample the
response N times at temperature, cluster samples by NLI-based semantic equivalence, and
treat dispersion as a hallucination signal - the intended answer to the "no ground truth
available" problem on Tier C providers. Scoped out because N resamples per checked
response multiply both cost and latency, and the Groq free-tier limit (30 RPM) makes it
impossible to demonstrate at eval scale. This is the most significant missing detector:
it is what UNGROUNDED mode is supposed to use.

**Presidio PII / NER** (`detectors/pii.py`). Design: Microsoft Presidio with a spaCy
small model for entity-level PII detection, backing the deterministic regex pre-pass
already running in Layer 0. Scoped out as a dependency-weight decision (Presidio + spaCy
+ model download). The consequence is real: `PERSON_HINT_RE` is a two-capitalised-words
heuristic that over-fires on ordinary title-case text, and Presidio is what would
replace it.

**Session risk ledger** (`session/risk_ledger.py`). Design: accumulate risk across turns
in a conversation and decay it over clean turns, so a user probing repeatedly escalates
even when no single turn crosses a threshold; for agents, gate the PLANNED ACTION before
tool execution rather than only the text. The router already accepts a `session_risk`
argument and folds it into the effective score - the plumbing exists, the accumulation
and decay do not. Scoped out for time; this is the top gap for agentic workloads.

**Feedback recalibration** (`feedback/calibration.py`). Design: human overrides in the
review queue become labeled examples, thresholds are re-optimized against the growing
set, and the false-positive rate is shown falling after the first review cycle. The
optimizer it would call is fully built and runs on any labeled set; what is missing is
the override capture path and the store. Scoped out because it depends on the console and
the ledger store, neither of which is built.

**Phoenix / OTLP collector** (`ledger/otel.py`, `ledger/store.py`). Design: emit
OpenTelemetry GenAI-semconv spans over OTLP to a locally-run Arize Phoenix, with a SQLite
evidence store behind it. Spans are constructed in the correct shape today (demo scene 6:
`gen_ai.*` attributes plus NIST AI RMF and OWASP LLM Top 10 mappings) but are PRINTED,
NOT EXPORTED - there is no collector, no exporter wiring, and no persistence. Scoped out
to keep the dependency footprint small; the span schema is the part that matters and it
is real.

**Streamlit operator console** (`console/app.py`). Design: a review queue showing flagged
traffic with its evidence packet, where a reviewer overrides a verdict and that override
feeds calibration. Scoped out for time. It is the human-facing half of the feedback loop,
and without it the loop cannot close.

**Layer 2 ADJUDICATE** (`detectors/adjudicate.py`). Design: LLM-as-judge or human review
with the evidence packet pre-assembled, for the small residual that Layers 0 and 1 cannot
resolve. The router already computes the escalation decision and emits
`escalate_to_layer: 2`, so the routing is live - but nothing consumes it. In the measured
run 0% of traffic reached Layer 2, because Layer 2 does not exist.

---

## Bias: we do NOT claim coverage

**Bias recall is 0.143** (2 of 14 labeled cases caught) at precision 1.000. Plainly:

- **Layer 0 has no fairness detector at all.** Nothing in REFLEX examines protected
  attributes or disparate reasoning. The bias label can only be set at Layer 1.
- **The guard model catches only blatant cases.** `gpt-oss-safeguard-20b` returns S10
  (hate) for overt discriminatory language, but the labeled cases that actually matter -
  declining a loan by region, preferring one gender for a technical role, inferring
  income from a surname - are phrased as ordinary business reasoning and are not
  returned as unsafe.
- The precision of 1.000 is not reassuring here. It reflects that the detector fires
  almost never, not that it fires accurately.

Closing this needs a dedicated fairness detector - counterfactual token substitution
across protected attributes, or a classifier trained for disparate-impact reasoning.
Neither is built. **No bias claim should appear in the deck, the video, or the proposal.**

---

## Eval-set enrichment: which numbers are quotable

The 300-case eval set is **deliberately enriched to 40% risky** (180 clean / 120 risky).
Real traffic looks nothing like this; the enrichment exists so every label has enough
positive support for a stable PR curve.

**Consequence: the raw 34% flag rate from the eval set must never be quoted as a
production number.** At realistic prevalence the review queue is dominated by the
false-positive rate on clean traffic (1.1%), not by true risk:

| real-world risk rate | projected flag rate | within a 3% budget |
|---|---|---|
| 10% | 8.5% | no |
| 5% | 4.8% | no |
| 3% | 3.3% | marginal |
| 1% | 1.9% | yes |

All deck, README, and demo figures use these projected values.

---

## The 3% flag budget is infeasible on the enriched set, and the tool reports it

A 3% review ceiling cannot be met against 40% risk prevalence: meeting it would require
missing real risk. The optimizer **reports this as a reviewer-capacity finding** and keeps
the cost-optimal operating point, instead of raising thresholds until the flag rate fits.

This matters because an earlier version did the opposite - it "satisfied" the budget by
driving false-negative rates to 1.000, i.e. detecting nothing at all. A budget solver that
silently stops detecting is a worse failure than a budget honestly reported as
unachievable. Two guards now exist: the budget is enforced **jointly at case level** (one
flagged case consumes one review slot regardless of how many labels fired, so a per-label
ceiling cannot over-constrain by ~5x), and any solution that fits the budget only by
collapsing recall below 25% is rejected as infeasible.

---

## Degenerate thresholds excluded above a 60% flag rate

With `cost_fn = 25 x cost_fp` in the `decision_support` profile, "flag everything"
(threshold 0.0) is the genuine minimum of the cost objective - at t=0 there are no misses,
so the FN term vanishes. It is also operationally worthless: a control plane that flags
100% of traffic carries no signal and guarantees the alert fatigue this project exists to
prevent.

Any candidate flagging more than **60%** of traffic (`MAX_USABLE_FLAG_RATE`) is therefore
excluded from the optimizer search. Before this guard, `decision_support` hallucination
selected t=0.00 and flagged 100% of traffic; after it, t=0.60 and 16.8%.

Related: thresholds are reported as the **midpoint of the optimal plateau**, not its lower
edge. Detector scores are discrete, so a range of thresholds shares one confusion matrix;
the lower edge produces brittle values like 0.005 sitting against a score boundary, while
the midpoint leaves equal headroom on both sides.

## Provider / model availability (recorded 30 Aug 2026)

- **`llama-3.1-8b-instant` is retired on Groq.** The key in `.env` returns
  `model_not_found`. The governed model is now **`openai/gpt-oss-20b`**, chosen as the
  closest small open-weight substitute. It is config-driven
  (`CONTROLPLANE_MODEL` in `.env`, `controlplane/config.py`) — a one-line swap.
- **`meta-llama/llama-guard-4-12b` is also retired.** RESOLVED: Layer 1 now uses
  `meta-llama/llama-prompt-guard-2-86m` for injection and `openai/gpt-oss-safeguard-20b`
  for safety categories. Both IDs were verified with a live call before any code was
  written against them.
- **Prompt Guard returns a bare probability as TEXT**, not a label - e.g. `"0.999559"`
  for an injection versus `"0.000352"` for a benign question. Separation is excellent and
  it answers in ~130 ms warm, which is why it carries the latency story.
- **`gpt-oss-safeguard-20b` REFUSES rather than classifies unless given an explicit
  policy prompt.** Asked a harmful question directly, it replies "I'm sorry, but I can't
  help with that" - which is useless as a classifier signal, and would have silently
  produced zero safety detections had we not probed it first. It emits structured labels
  only when handed an explicit category policy (`SAFEGUARD_POLICY` in
  `detectors/guard_model.py`) and asked to classify an ASSISTANT RESPONSE. Verified
  emitting `safe/none`, `unsafe/S7` (privacy), `unsafe/S6` (specialized advice).
- **Injection is scored on the RESPONSE, not the prompt.** A hostile prompt that the
  model correctly refused is a SUCCESS, not a risk; scoring the prompt punishes exactly
  the behaviour we want. Fixing this moved safety F1 from 0.256 to 0.850.
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
- **Environment is Python 3.12.3, not 3.11.** `requirements.txt` was repinned for 3.12
  from an actual `pip-compile` resolution and verified with `pip install --dry-run`
  (83 packages, no conflicts). The deferred ML stack lives in `requirements-future.txt`.
- **litellm is imported lazily** inside the guard call path. Importing it costs ~8.7 s,
  which a cached or offline run never needs to pay; this is what keeps `make demo`
  under 0.05 s.
- **Windows long-path caveat:** installing into a deeply-nested directory fails with
  `OSError [Errno 2]` on litellm's nested guardrail YAML files when total path length
  exceeds ~260 chars and `LongPathsEnabled` is not set. Clone to a short path such as
  `C:\controlplane-checker`.

## Detector calibration (TUNED - supersedes the earlier placeholder note)

The hand-set placeholder scores flagged here previously have been **replaced with
values chosen by the cost-weighted threshold optimizer** over the 300-case labeled set
(`controlplane/eval/optimizer.py`, `make tune`). Re-run `make tune` after touching any
detector severity.

**Tuned operating points** (written into `policy/packs/india_dpdp_rbi.yaml`, per profile):

| label | threshold | precision | recall | F1 | FP rate |
|---|---|---|---|---|---|
| hallucination | 0.60 | 1.000 | 0.755 | 0.860 | 0.000 |
| privacy | 0.36 | 0.962 | 0.943 | 0.952 | 0.008 |
| safety | 0.40 | 0.964 | 0.761 | 0.850 | 0.009 |
| cost | 0.50 | 1.000 | 1.000 | 1.000 | 0.000 |
| bias | 0.46 | 1.000 | 0.143 | 0.250 | 0.000 |

**Hard-negative false-positive rate: 2.5% (2/81), down from 50% before tuning.**
False-positive rate on clean traffic overall: **1.1%**.

Two real bugs were found by the eval and fixed - this is what the exercise was for:

1. **Citation normalisation.** `[doc-7]` cited against a retrieved `doc-7` was reported
   as DANGLING, because the regex stripped the `doc-` prefix into the capture group and
   then compared bare `7` to `doc-7`. Every correctly-cited response was accruing a
   false hallucination signal. Fixed with `_norm_citation`; accounted for 17 of the FPs.
2. **Injection scored on the wrong text.** Prompt Guard was run against the PROMPT, so a
   hostile prompt that the model correctly REFUSED was flagged as unsafe - punishing
   exactly the behaviour we want. It now scores the RESPONSE (falling back to the prompt
   only for pre-flight input screening). Safety F1 went 0.256 -> 0.850.

A third fix was to ground truth, not code: a PII leak is also an S7 safety event, and the
guard models were right to say so. `safety` positives went 28 -> 71 once labelled properly.

**Known weakness, stated rather than hidden: bias recall is 0.143.** Layer 0 has no bias
detector at all, and `gpt-oss-safeguard-20b` rarely returns S10 for the subtler
discriminatory-reasoning cases. Bias is effectively detected only when it is blatant.
Fixing this needs a dedicated fairness detector, which is not built.

**Optimizer design decisions worth knowing:**

- The flag budget is a **case-level** constraint, not per-label. Applying a 3% ceiling to
  each of 5 labels independently over-constrains by ~5x and drives recall to zero; an
  earlier version did exactly that and reported FN rates of 1.000.
- **Degenerate thresholds are rejected.** With `cost_fn = 25 x cost_fp` in
  decision_support, "flag everything" (t=0) is the genuine cost minimum. It is also
  operationally useless, so any point flagging >60% of traffic is excluded.
- Thresholds are the **midpoint of the optimal plateau**, not its lower edge. Detector
  scores are discrete, so a range of thresholds share one confusion matrix; the lower
  edge gives brittle values like 0.005 sitting against a score boundary.
- **The 3% flag budget is INFEASIBLE on the eval set and the tool says so** rather than
  silently suppressing detection. The set is deliberately enriched to 40% risky so every
  label has support; at realistic 1-3% prevalence the projected flag rate is 1.9-3.3%,
  i.e. the budget is met at 1% and marginal at 3%. Both numbers are reported.

## Eval harness and the Groq rate limit

Groq free tier is 30 RPM, so the harness is built around not spending it:

- Guard models run **only on cases that escalate to Layer 1** (52% of the set here).
- Every guard verdict is **cached to disk** (`data/guard_cache.json`, 261 entries) keyed
  on a hash of model + text.
- **`--offline`** runs entirely from cache and never touches the network: 300 cases in
  **~0.1-0.3 s**, deterministic across runs, so the demo can be re-recorded instantly.
- **`--limit N`** subsamples for fast iteration.
- Warming the cache from cold is ~145 unique calls after dedupe (~5 min at 28 RPM),
  not the 600 a naive 300 x 2 pass would need.

**Latency reporting**: the harness scores pre-recorded responses, so its numbers are
pure ADDED governance latency with no model call in the path - p50 **0.33 ms**, p95
**1.10 ms** against a ~10 ms Layer 0 budget. Live model latency (386-967 ms on Groq) is
measured separately by the proxy and is reported separately on purpose.


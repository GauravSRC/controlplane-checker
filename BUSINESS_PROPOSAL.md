# ControlPlane Checker — Business Proposal

_Accenture Innovation Challenge 2026 · Round 2 · Track: ControlPlane.ai_

---

## 1. Problem

Enterprises have shipped LLM features faster than they have built any way to know whether
those features are working. Three failure modes recur, and they are usually discovered
only after someone has already acted on the output:

- **Confidently wrong.** The model fabricates a policy clause, a figure, or a citation,
  and states it in the same tone it uses when correct.
- **Quietly expensive.** Retry loops, runaway agent tool-calling, and prompt bloat burn
  budget invisibly. The bill arrives monthly; the cause is per-request.
- **Subtly unsafe.** Personal data leaks into an output, an injection surfaces in a
  reply, or a decision leans on a protected attribute.

Four properties make this hard to solve with a naive checker:

1. **Risk tolerance is not uniform.** A summarizer and a payment agent cannot share one
   threshold. A customer-facing chat has a 150 ms budget; a credit decision can afford
   five seconds. One-size-fits-all checking fails at both ends.
2. **The categories overlap.** A fabricated detail about a named person is *simultaneously*
   a hallucination and a privacy incident. Systems that force a single label lose
   information at the moment it matters.
3. **There is often no ground truth available in real time.** Much of what a model says
   cannot be checked against a source within the latency budget — sometimes not at all.
4. **Over-flagging is its own failure.** A checker that flags 30% of traffic trains users
   to click through warnings, which is worse than no checker, because it manufactures the
   appearance of oversight. Under-flagging creates liability. Real systems *tune* this
   trade-off; they do not solve it away.

And the binding constraint: enterprises consume foundation models **via API**. There is no
access to model internals. Any workable control layer operates at the input/output
boundary.

## 2. Solution

A verification mesh that sits between the model and the application and **spends
verification compute in proportion to consequence**. Everyone goes through the metal
detector; only some get a pat-down.

**Three-tier cascade.** Layer 0 (~10 ms, all traffic) uses only signals that are already
free: citation-ID validity against the retrieved set, JSON/schema conformance,
deterministic PII and secret patterns, token/retry/tool-loop budget counters, and sequence
logprobs where the provider exposes them. Layer 1 (~80 ms, elevated risk only) adds guard
models for injection and safety categories. Layer 2 (~1 s) is LLM-as-judge or a human,
with the evidence packet pre-assembled. In measurement, **52% of traffic escalated past
Layer 0** — the rest was resolved by free signals.

**Route on consequence, not on a score.** Five risk labels are tracked independently and
allowed to co-fire. The routing decision crosses that vector with *blast radius* — what
acting on the response actually does: `INFORMATIONAL` (a user reads it) → `ADVISORY` (a
user decides on it) → `SIDE_EFFECT` (it triggers a tool call) → `IRREVERSIBLE` (payment,
deletion, external communication, clinical or legal advice). The same risk score produces
`PASS` on a summary and `BLOCK` on a payment.

**Policy as versioned code.** Regulatory expectations differ by geography and keep
changing, so rules live in hot-reloadable YAML packs, not in application code. Two ship
today: EU AI Act, and India DPDP Act + RBI FREE-AI. Identical input reaches different
verdicts under different packs, with no redeploy.

**Prefer abstention to silent blocking.** A hard block teaches users to route around the
control layer. Irreversible actions get a rollback window: the text streams immediately
while the side effect waits for clearance, so users rarely feel the check.

**Deployability is the wedge.** The service exposes an OpenAI-compatible
`/v1/chat/completions`. An existing application is governed by changing its `base_url` —
no SDK migration, no code review, no re-architecture. This is what makes a pilot a
one-afternoon exercise instead of a quarter.

## 3. Target users and buyers

| Buyer | What they own | What they need from us |
|---|---|---|
| **Head of AI Governance / Chief AI Officer** | Answering the board's question "how do we know our AI is behaving?" | Evidence, not assurances. Per-control audit trail mapped to NIST AI RMF and OWASP LLM Top 10, produced continuously rather than assembled before an audit. Jurisdiction-specific policy they can point at. |
| **CISO** | Data exfiltration, prompt injection, credential leakage | Injection and PII detection at the output boundary, on traffic they do not control; incidents surfaced with the evidence attached; fail-closed behaviour on high-consequence routes. |
| **Platform / ML engineering lead** | Latency budgets, reliability, on-call | Something that does not blow the p95, does not require re-instrumenting every service, and degrades predictably when a provider misbehaves. Added latency is measured and reported as a first-class metric (p95 **1.12 ms** against a 10 ms budget). |
| **Business owner of the use case** | Whether the feature is worth running | The trade-off made explicit: how much review capacity buys how much recall, in their units, for their use case. |

The economic buyer is usually AI Governance or the CISO. The **technical veto** sits with
the platform lead — which is why zero-code-change integration and a measured latency
budget are product decisions, not engineering vanity.

## 4. Business case and impact

**The metric is Cost per Verified Correct Outcome (CVCO), not cost per token.** Cheap
tokens that produce an answer someone has to redo, or that trigger an incident, are not
cheap. CVCO makes the comparison honest: total spend (inference + verification + human
review + rework) divided by outcomes that were correct *and* checked.

Three sources of value:

1. **Incidents prevented.** On the labeled set, privacy detection runs at 0.962 precision
   / 0.943 recall, safety at 0.964 / 0.761. Each caught PII disclosure on a customer-facing
   route is a regulatory notification avoided — under DPDP, a reportable personal-data
   breach; under the EU AI Act, a transparency and oversight failure on a high-risk system.
2. **Review capacity spent where it matters.** Tuning moved the hard-negative false
   positive rate from **50% to 2.5%**, and the false-positive rate on clean traffic to
   **1.1%**. At a realistic 1–3% risk prevalence, the projected review load is **1.9–3.3%
   of traffic** — inside or near a 3% ceiling. The same reviewer headcount covers roughly
   an order of magnitude more traffic than an untuned checker would allow.
3. **Audit evidence as a by-product.** Every verdict emits an OpenTelemetry GenAI-semconv
   span carrying the risk vector, the policy rule that fired, the blast radius, and the
   NIST/OWASP mappings. The audit artifact is produced by running the system, rather than
   reconstructed by hand at audit time. This is the difference between oversight as a tax
   and oversight as exhaust.

**Cost of the control layer itself.** Layer 0 is pure CPU, no model call — effectively
free. Guard-model calls are incurred only on escalated traffic (52% here, and lower at
production prevalence), and verdicts are cached on a content hash. The verification budget
scales with risk, not with volume.

## 5. Roadmap

**Phase 1 — Pilot, one use case (4–6 weeks).** Pick a single high-blast-radius workflow
where a wrong answer has a name — a BFSI advisory chat or a claims summarizer. Point it at
the proxy in shadow mode: govern, log, and block nothing. Build the labeled set from that
team's real traffic patterns (not our synthetic set) and tune thresholds against their
actual cost of a false alarm versus a miss. Exit criterion: the team agrees the flags are
worth reading.

**Phase 2 — One business unit (3–4 months).** Enable enforcement on high-blast-radius
routes; keep informational routes on annotate-only. Add the human review queue and the
feedback loop, so overrides become labeled data and thresholds recalibrate — the
false-positive rate should fall measurably after the first review cycle. Stand up the
Phoenix/OTLP collector so governance has a live dashboard rather than a monthly export.
Add the jurisdiction packs that unit actually operates under.

**Phase 3 — Enterprise-wide (6–12 months).** Multi-tenant deployment with per-team
policy packs and per-team flag budgets. Session-level risk ledger for multi-turn and
agentic workloads, gating the *action* rather than only the text. Full Layer 2 adjudication
with an SLA. Integrate CVCO into the AI portfolio review, so use cases are compared on
verified outcomes rather than on model spend.

Sequencing rationale: enforcement authority is earned. Shadow mode first, high-consequence
routes next, broad enforcement only once the false-positive rate is demonstrably low
enough that people trust the flags.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| **The checker is itself wrong.** A verification layer that hallucinates is worse than none — it manufactures false confidence. | Layer 0 is deterministic: regex, set membership, schema, counters. No model judgment, no failure mode of its own. Model-based judgment is confined to Layer 1/2, always with the deterministic evidence attached, and every verdict carries the signals that produced it so a human can check the checker. Our own false-positive and false-negative rates are published per label, including the one that is bad (bias recall 0.143). A control layer that hides its error rates has the same credibility problem as the model it governs. |
| **Latency budget breach.** Governance that blows the p95 gets removed. | Latency is a first-class, measured metric (p50 0.34 ms / p95 1.12 ms added, against a 10 ms Layer 0 budget). Every Layer 1 detector runs in parallel under one shared budget from the use-case profile; when the budget is exhausted, partial results are kept and the response is marked degraded. Fail-open versus fail-closed is set per route: low-blast-radius traffic degrades to the Layer 0 verdict and logs it; `SIDE_EFFECT` and `IRREVERSIBLE` routes fail **closed**, overriding the profile's preference. |
| **Regulatory change.** Rules age faster than release cycles. | Policy lives in versioned YAML packs, hot-reloaded on file change with no redeploy. A pack that fails validation is rejected and the last good version stays live, so a bad edit cannot take governance offline. New jurisdictions are new files, not new code. |
| **Alert fatigue.** Over-flagging trains users to bypass the control. | An explicit flag budget with a cost-weighted optimizer, tuned against hard negatives — responses engineered to *look* risky but be correct. Where the budget is not achievable, the tool reports it as a capacity finding rather than suppressing detection to appear compliant. |
| **Provider capability changes underneath us.** Models are retired and features disappear. | Capability negotiation probes what each provider exposes and degrades deliberately across Tier A/B/C. Unavailable logprobs return "unmeasured", never a fabricated confidence. This happened during development: two of the three originally planned Groq models were retired, and the system degraded to Tier C rather than breaking. |
| **Synthetic-only calibration.** Thresholds tuned on invented data may not transfer. | Stated plainly rather than papered over. Phase 1 exists precisely to re-tune on real traffic; the current numbers demonstrate the *method*, and the tuning pipeline is one command against a new labeled set. |

## 7. What is built today

Working and measured: the OpenAI-compatible proxy; capability negotiation; Layer 0
REFLEX; Layer 1 guard models with caching, backoff and per-route timeouts; the
multi-label router with the full blast-radius matrix; the policy engine with two packs
and hot reload; a 300-case labeled eval set with hard negatives; and the cost-weighted
threshold optimizer with a flag-budget solver.

Designed and specified but **not implemented**: NLI groundedness, SelfCheckGPT semantic
entropy, Presidio NER, the session risk ledger, feedback recalibration, the Phoenix/OTLP
collector, the operator console, and the Layer 2 human queue. Each is documented in
[`ASSUMPTIONS.md`](ASSUMPTIONS.md) with its design and the reason it was scoped out.

The prototype's claim is not that the problem is solved. It is that the *architecture* —
cascade by consequence, multi-label routing, policy as code, and tuning as a
first-class activity — is sound and demonstrable end to end.

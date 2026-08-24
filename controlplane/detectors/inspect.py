"""Layer 1 — INSPECT (~80 ms, elevated-risk traffic only).

Orchestrates the mid-tier detectors and fuses their partial risk vectors:
  - ``guard_model``  — injection/jailbreak/safety (Llama Guard + Prompt Guard).
  - ``groundedness`` — NLI cross-encoder entailment vs retrieved evidence.
  - ``selfcheck``    — consistency-based hallucination detection over resampled outputs.

Invoked only when Layer 0 or the session ledger elevates risk, keeping the fast path
fast. Runs its sub-detectors in parallel with per-detector timeouts.

NOTE: module name shadows the stdlib ``inspect``; always import as
``controlplane.detectors.inspect`` and avoid bare ``import inspect`` within this package.

Scaffold only — no logic yet.
"""

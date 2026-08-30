"""Layer 0 — REFLEX (~10 ms, majority of traffic, concurrent with token streaming).

Uses only signals that are already free (no extra model calls):
  - sequence logprobs (when the provider exposes them),
  - citation-ID validity (does each cited doc ID exist in the retrieved set?),
  - schema / JSON conformance,
  - deterministic PII and secret regex (fast pre-pass; deep PII lives in ``pii``),
  - budget counters for tokens / retries / tool-loop depth (feeds the COST label).

TODO: implement as one or more ``Detector`` subclasses.
Scaffold only — no logic yet.
"""

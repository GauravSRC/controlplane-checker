"""Detector contract and shared risk types.

Defines the abstractions every detector implements so the mesh can run them uniformly
in parallel with per-detector timeouts and fuse their outputs.

Planned surface (to implement):
  - ``RiskLabel`` enum: HALLUCINATION, PRIVACY, BIAS, SAFETY, COST.
  - ``VerificationMode`` enum: GROUNDED, UNGROUNDED, UNVERIFIABLE.
  - ``RiskVector``: per-label scores in [0, 1] that may co-fire, plus provenance.
  - ``DetectorResult``: risk vector + latency_ms + verification mode + evidence refs
    + degraded flag (set when a timeout forced fail-open).
  - ``Detector`` (ABC): ``layer``, ``name``, ``timeout_ms``, and an async
    ``run(context) -> DetectorResult``.

Design rules: multi-label (never collapse to one scalar here), latency-measured, and
fail-open/closed decided by the caller per route.

Scaffold only — no logic yet.
"""

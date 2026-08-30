"""Feedback-driven recalibration.

Human overrides recorded in the ledger (``ledger.store``) become labeled data. Periodically
re-run the cost-weighted threshold optimizer (``eval.optimizer``) on the growing labeled
set so per-detector thresholds recalibrate over time.

Demo goal: show the false-positive rate dropping after feedback is incorporated, while
staying within the flag budget.

Planned surface (to implement):
  - ``ingest_overrides()`` -> labeled examples,
  - ``recalibrate(profile)`` -> new thresholds (persisted + hot-applied),
  - ``fp_rate_trend()`` for the console/demo.

Scaffold only — no logic yet.
"""

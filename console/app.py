"""Streamlit operator console.

Human-facing surface over the evidence ledger (``controlplane.ledger.store``):
  - live verdict stream with the multi-label risk vector and blast radius,
  - evidence packets for HOLD / ADJUDICATE items + human override controls (feedback loop),
  - PR curves, thresholds, flag-budget utilization, and the FP-rate-after-feedback trend,
  - p50/p95 latency and CVCO dashboards,
  - policy-pack selector (hot reload).

Run with ``make console`` / ``streamlit run console/app.py``.

Scaffold only — no logic yet.
"""

"""Eval harness — run detectors over the labeled set and report quality + latency.

Drives the synthetic dataset (``generate_dataset``) through the detector mesh and reports:
  - per-detector precision / recall and PR curves (via ``optimizer``),
  - end-to-end verdict accuracy against labels,
  - p50 / p95 added latency (latency is a first-class metric),
  - CVCO (Cost per Verified Correct Outcome).

Runnable via ``make eval`` / ``python -m controlplane.eval.harness`` and reused by pytest.

Scaffold only — no logic yet.
"""

"""Single-command demo entry point for judges (``make demo`` / ``python scripts/demo.py``).

Intended end-to-end walkthrough (to implement):
  1. Ensure the synthetic dataset exists (``controlplane.eval.generate_dataset``).
  2. Start the governance proxy + local Arize Phoenix, or run in-process.
  3. Fire a scripted set of requests that exercise every headline feature:
       - capability tiers A/B/C,
       - the multi-label OVERLAP case (fabricated personal detail => hallucination + privacy),
       - the three verification modes (GROUNDED / UNGROUNDED / UNVERIFIABLE),
       - blast-radius routing incl. the irreversible rollback window,
       - session risk compounding across turns + agent action gating,
       - a policy hot-reload,
       - a human override followed by recalibration (FP rate drops).
  4. Print CVCO and p50/p95 added latency, and point to the Phoenix + Streamlit URLs.

Scaffold only — no logic yet.
"""

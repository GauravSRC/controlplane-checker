"""Detector mesh — the three-tier verification cascade (Reflex / Inspect / Adjudicate).

All detectors conform to the ``base.Detector`` contract and run in parallel via
``asyncio.gather`` with per-detector timeouts. Each returns a partial multi-label risk
vector (hallucination / privacy / bias / safety / cost) that is fused by the router.
"""

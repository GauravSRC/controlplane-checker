"""Self-consistency detector — UNGROUNDED verification mode (HALLUCINATION label).

When no retrievable source exists, estimate hallucination risk without ground truth:
resample the response N times at temperature and measure semantic consistency
(SelfCheckGPT-style NLI/clustering, reusing the NLI cross-encoder). High divergence /
semantic entropy across samples => higher hallucination score.

Claims that are neither groundable nor consistently answerable are surfaced as
UNVERIFIABLE rather than pretending to have checked them.

TODO: implement resampling + NLI clustering as a ``Detector``.
Scaffold only — no logic yet.
"""

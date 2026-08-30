"""Groundedness detector — GROUNDED verification mode (HALLUCINATION label).

When retrieved source evidence exists, run an NLI cross-encoder
(``cross-encoder/nli-deberta-v3-small``, CPU) to check whether each response claim is
entailed by the evidence. Contradiction / neutral at the sentence level raises the
hallucination score and supplies span-level evidence refs for the ledger.

TODO: implement as a ``Detector`` subclass; sentence-split the
response, score entailment vs each evidence chunk, aggregate. Scaffold only — no logic yet.
"""

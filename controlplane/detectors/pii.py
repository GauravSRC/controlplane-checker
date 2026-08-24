"""PII / secrets detector — PRIVACY label (and the overlap case).

Deep PII/secret detection with Microsoft Presidio + spaCy (``en_core_web_sm``, CPU),
layered on the deterministic regex pre-pass in ``reflex``.

Overlap requirement: a fabricated detail *about a person* must co-fire PRIVACY here and
HALLUCINATION in ``groundedness``/``selfcheck`` — the two labels are independent and both
raised, demonstrating that bias/hallucination/privacy risks overlap rather than being
cleanly categorized.

Maps to NIST AI RMF privacy subcategories for the evidence ledger.

TODO: implement as a ``Detector`` subclass using a shared
Presidio analyzer engine. Scaffold only — no logic yet.
"""

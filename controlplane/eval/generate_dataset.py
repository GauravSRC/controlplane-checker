"""Generate the synthetic labeled eval set (~400 cases).

Builds clearly-synthetic, illustrative cases spanning the risk labels
(hallucination / privacy / bias / safety / cost), the three verification modes
(GROUNDED / UNGROUNDED / UNVERIFIABLE), and the four blast radii. Includes the required
OVERLAP case: a fabricated detail about a person labeled BOTH hallucination and privacy.

Output: JSONL under ``data/`` (gitignored), regenerated deterministically from a seed.

Planned surface (to implement):
  - case templates per label/mode/blast-radius,
  - ``generate(n=400, seed=...) -> writes data/eval.jsonl``.

Scaffold only — no logic yet.
"""

"""Decision router — confidence x blast radius, NEVER a single score.

Blast radius (what acting on the response actually does):
  INFORMATIONAL < ADVISORY < SIDE_EFFECT < IRREVERSIBLE

Decision matrix (risk score x blast radius):
  risk        INFORMATIONAL  ADVISORY   SIDE_EFFECT  IRREVERSIBLE
  <0.30       PASS           PASS       PASS         HOLD
  0.30-0.60   ANNOTATE       ANNOTATE   HOLD         HOLD
  >0.60       ANNOTATE       ANNOTATE   HOLD         BLOCK+handoff
  policy hit  REPAIR         REPAIR     BLOCK        BLOCK

Design principle: ABSTAIN rather than silently block. Irreversible actions get a
ROLLBACK WINDOW — text streams to the user instantly while side effects wait for
clearance, so users never feel the check.

Inputs fuse the detector risk vector, the policy outcome (``policy.engine``), and the
accumulated session risk (``session.risk_ledger``). It also decides which layer to
escalate to (Reflex -> Inspect -> Adjudicate).

Planned surface (to implement):
  - ``BlastRadius`` and ``Verdict`` enums,
  - ``decide(risk_vector, blast_radius, policy_outcome, session_risk) -> Decision``.

Scaffold only — no logic yet.
"""

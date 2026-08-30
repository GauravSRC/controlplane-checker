"""Decision router - confidence x blast radius, NEVER a single score.

Blast radius (what acting on the response actually does):
  INFORMATIONAL < ADVISORY < SIDE_EFFECT < IRREVERSIBLE

Decision matrix (CLAUDE.md 2.2), reproduced exactly:

  risk        INFORMATIONAL  ADVISORY   SIDE_EFFECT  IRREVERSIBLE
  <0.30       PASS           PASS       PASS         HOLD
  0.30-0.60   ANNOTATE       ANNOTATE   HOLD         HOLD
  >0.60       ANNOTATE       ANNOTATE   HOLD         BLOCK+handoff
  policy hit  REPAIR         REPAIR     BLOCK        BLOCK

A policy hit SHORT-CIRCUITS: it bypasses the score bands entirely.

The vector is never summed. The routing scalar is the MAX over labels (the worst thing
found governs the route) and the label that produced it is carried through, so an
ANNOTATE always says *what* fired. Irreversible actions get a rollback window: text
streams to the user instantly while side effects wait for clearance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from controlplane.detectors.base import RiskLabel, RiskVector


class BlastRadius(str, Enum):
    INFORMATIONAL = "INFORMATIONAL"  # user reads it
    ADVISORY = "ADVISORY"            # user decides on it
    SIDE_EFFECT = "SIDE_EFFECT"      # triggers a tool call, writes state
    IRREVERSIBLE = "IRREVERSIBLE"    # payment, deletion, external comms, clinical/legal

    @classmethod
    def parse(cls, value: str | None, default: "BlastRadius | None" = None) -> "BlastRadius":
        default = default or cls.INFORMATIONAL
        if not value:
            return default
        try:
            return cls(str(value).strip().upper())
        except ValueError:
            return default


class Verdict(str, Enum):
    PASS = "PASS"
    ANNOTATE = "ANNOTATE"
    HOLD = "HOLD"
    REPAIR = "REPAIR"
    BLOCK = "BLOCK"


# Score band edges. <0.30 low, 0.30-0.60 mid (inclusive), >0.60 high.
LOW_BAND = 0.30
HIGH_BAND = 0.60


@dataclass
class PolicyOutcome:
    """Result of policy evaluation (controlplane.policy.engine)."""

    hit: bool = False
    rule_id: str = ""
    reason: str = ""
    repairable: bool = True


@dataclass
class Decision:
    verdict: Verdict
    blast_radius: BlastRadius
    risk_score: float
    dominant_label: RiskLabel | None
    band: str
    risk_vector: dict[str, float] = field(default_factory=dict)
    fired_labels: list[str] = field(default_factory=list)
    escalate_to_layer: int = 0
    rollback_window: bool = False
    handoff: bool = False
    policy_rule: str = ""
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "action": self.verdict.value,
            "blast_radius": self.blast_radius.value,
            "risk_score": round(self.risk_score, 4),
            "dominant_label": self.dominant_label.value if self.dominant_label else None,
            "band": self.band,
            "risk_vector": self.risk_vector,
            "fired_labels": self.fired_labels,
            "escalate_to_layer": self.escalate_to_layer,
            "rollback_window": self.rollback_window,
            "handoff": self.handoff,
            "policy_rule": self.policy_rule,
            "rationale": self.rationale,
        }


# The matrix, as data. Rows are score bands; policy hits are handled before lookup.
_MATRIX: dict[str, dict[BlastRadius, Verdict]] = {
    "low": {
        BlastRadius.INFORMATIONAL: Verdict.PASS,
        BlastRadius.ADVISORY: Verdict.PASS,
        BlastRadius.SIDE_EFFECT: Verdict.PASS,
        BlastRadius.IRREVERSIBLE: Verdict.HOLD,
    },
    "mid": {
        BlastRadius.INFORMATIONAL: Verdict.ANNOTATE,
        BlastRadius.ADVISORY: Verdict.ANNOTATE,
        BlastRadius.SIDE_EFFECT: Verdict.HOLD,
        BlastRadius.IRREVERSIBLE: Verdict.HOLD,
    },
    "high": {
        BlastRadius.INFORMATIONAL: Verdict.ANNOTATE,
        BlastRadius.ADVISORY: Verdict.ANNOTATE,
        BlastRadius.SIDE_EFFECT: Verdict.HOLD,
        BlastRadius.IRREVERSIBLE: Verdict.BLOCK,  # + handoff
    },
    "policy": {
        BlastRadius.INFORMATIONAL: Verdict.REPAIR,
        BlastRadius.ADVISORY: Verdict.REPAIR,
        BlastRadius.SIDE_EFFECT: Verdict.BLOCK,
        BlastRadius.IRREVERSIBLE: Verdict.BLOCK,
    },
}


def band_for(score: float) -> str:
    if score < LOW_BAND:
        return "low"
    if score <= HIGH_BAND:
        return "mid"
    return "high"


def fuse(risk: RiskVector) -> tuple[float, RiskLabel | None]:
    """Collapse to a routing scalar ONLY at the moment of routing.

    Max-over-labels: the worst thing found governs the route. The vector itself is
    preserved on the Decision so nothing downstream sees just a number.
    """
    if not risk.scores:
        return 0.0, None
    label = max(risk.scores, key=lambda k: risk.scores[k])
    return risk.scores[label], (label if risk.scores[label] > 0.0 else None)


def decide(
    risk_vector: RiskVector,
    blast_radius: BlastRadius = BlastRadius.INFORMATIONAL,
    policy_outcome: PolicyOutcome | None = None,
    session_risk: float = 0.0,
) -> Decision:
    """Route on confidence x blast radius. Policy hits short-circuit the score bands."""
    score, dominant = fuse(risk_vector)

    # Multi-turn compounding: accumulated session risk lifts the effective score but
    # never lowers it (controlplane.session.risk_ledger owns the accumulation/decay).
    effective = min(1.0, max(score, 0.0) + max(session_risk, 0.0))

    policy_outcome = policy_outcome or PolicyOutcome()
    if policy_outcome.hit:
        band = "policy"
        verdict = _MATRIX["policy"][blast_radius]
        rationale = f"policy hit [{policy_outcome.rule_id}]: {policy_outcome.reason}"
        # A non-repairable policy hit cannot be fixed by rewriting the text.
        if verdict is Verdict.REPAIR and not policy_outcome.repairable:
            verdict = Verdict.BLOCK
            rationale += " (not repairable -> BLOCK)"
    else:
        band = band_for(effective)
        verdict = _MATRIX[band][blast_radius]
        label_txt = dominant.value if dominant else "none"
        rationale = (
            f"risk {effective:.2f} ({band} band, dominant={label_txt}) "
            f"x blast_radius={blast_radius.value}"
        )
        if session_risk > 0 and effective > score:
            rationale += f"; session risk +{session_risk:.2f}"

    # Escalation: how much more verification this response has earned.
    if verdict is Verdict.PASS:
        escalate = 0
    elif verdict in (Verdict.ANNOTATE, Verdict.REPAIR):
        escalate = 1
    else:  # HOLD / BLOCK
        escalate = 2

    # ABSTAIN over silent block: irreversible actions stream text immediately while
    # the side effect waits for clearance.
    rollback = blast_radius is BlastRadius.IRREVERSIBLE and verdict in (
        Verdict.HOLD,
        Verdict.BLOCK,
    )
    handoff = verdict is Verdict.BLOCK

    return Decision(
        verdict=verdict,
        blast_radius=blast_radius,
        risk_score=effective,
        dominant_label=dominant,
        band=band,
        risk_vector=risk_vector.to_dict(),
        fired_labels=[lbl.value for lbl in risk_vector.fired_labels],
        escalate_to_layer=escalate,
        rollback_window=rollback,
        handoff=handoff,
        policy_rule=policy_outcome.rule_id if policy_outcome.hit else "",
        rationale=rationale,
    )

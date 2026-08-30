"""Detector contract and shared risk types.

Multi-label by construction: a single response can be simultaneously a hallucination
and a privacy leak (CLAUDE.md 3.2), so scores live in a vector and are never collapsed
to one scalar here. Fusion to a routing scalar happens in ``router.decision``, per label.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RiskLabel(str, Enum):
    HALLUCINATION = "hallucination"
    PRIVACY = "privacy"
    BIAS = "bias"
    SAFETY = "safety"
    COST = "cost"


class VerificationMode(str, Enum):
    """How (or whether) a claim could actually be checked."""

    GROUNDED = "grounded"          # retrieved source exists -> entailment check
    UNGROUNDED = "ungrounded"      # no source -> self-consistency
    UNVERIFIABLE = "unverifiable"  # labelled as unverifiable, not silently "checked"


class CapabilityTier(str, Enum):
    """What the provider actually exposes (CLAUDE.md 3.1)."""

    A = "A"  # self-hosted open-weight: hidden states + logprobs
    B = "B"  # API with logprobs: token entropy + sampled consistency
    C = "C"  # API text-only: black-box consistency + retrieval verification


@dataclass
class Signal:
    """One fired observation, kept for the evidence ledger."""

    name: str
    label: RiskLabel
    score: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label.value,
            "score": round(self.score, 4),
            "detail": self.detail,
        }


@dataclass
class RiskVector:
    """Per-label risk in [0, 1]. Labels co-fire; they are never summed into one number."""

    scores: dict[RiskLabel, float] = field(
        default_factory=lambda: {label: 0.0 for label in RiskLabel}
    )
    signals: list[Signal] = field(default_factory=list)

    def fire(self, name: str, label: RiskLabel, score: float, detail: str = "") -> None:
        """Record a signal, keeping the strongest score seen for that label."""
        score = _clamp(score)
        self.scores[label] = max(self.scores.get(label, 0.0), score)
        self.signals.append(Signal(name=name, label=label, score=score, detail=detail))

    def get(self, label: RiskLabel) -> float:
        return self.scores.get(label, 0.0)

    def merge(self, other: "RiskVector") -> "RiskVector":
        for label, score in other.scores.items():
            self.scores[label] = max(self.scores.get(label, 0.0), score)
        self.signals.extend(other.signals)
        return self

    @property
    def fired_labels(self) -> list[RiskLabel]:
        return [lbl for lbl, s in self.scores.items() if s > 0.0]

    def to_dict(self) -> dict[str, float]:
        return {lbl.value: round(s, 4) for lbl, s in self.scores.items()}


@dataclass
class DetectorResult:
    detector: str
    layer: int
    risk: RiskVector
    latency_ms: float
    mode: VerificationMode = VerificationMode.UNGROUNDED
    degraded: bool = False   # set when a timeout/missing capability forced fail-open
    notes: list[str] = field(default_factory=list)


class Detector(ABC):
    """All detectors run in parallel under a per-detector timeout owned by the caller."""

    name: str = "detector"
    layer: int = 0
    timeout_ms: int = 10

    @abstractmethod
    async def run(self, context: "DetectionContext") -> DetectorResult: ...


@dataclass
class DetectionContext:
    """Everything a detector may look at. Absent fields degrade, never crash."""

    prompt: str = ""
    response: str = ""
    retrieved_ids: list[str] = field(default_factory=list)
    logprobs: list[float] | None = None
    tier: CapabilityTier = CapabilityTier.C
    expect_json: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    tool_loop_depth: int = 0
    usecase: str = "default"
    meta: dict[str, Any] = field(default_factory=dict)


def _clamp(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


class _Timer:
    """Wall-clock helper so every detector path is measurable (latency is a feature)."""

    def __enter__(self) -> "_Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.ms = (time.perf_counter() - self._t0) * 1000.0

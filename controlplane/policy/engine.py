"""Policy engine - load, validate, hot-reload, and evaluate YAML policy packs.

Regulatory expectations differ by geography/industry and keep evolving, so rules live in
versioned YAML (policy/packs) rather than in code, and reload without redeploy.

Hot reload is mtime-based: every resolve() stats the file and reparses if it changed.
A stat is ~microseconds, so this costs nothing on the request path and needs no watcher
thread. A pack that fails validation is REJECTED and the last good version stays live -
a typo in a policy file must never take governance offline.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from controlplane.detectors.base import RiskLabel, RiskVector
from controlplane.router.decision import BlastRadius, PolicyOutcome

PACKS_DIR = Path(__file__).resolve().parent / "packs"

_BLAST_ORDER = {
    BlastRadius.INFORMATIONAL: 0,
    BlastRadius.ADVISORY: 1,
    BlastRadius.SIDE_EFFECT: 2,
    BlastRadius.IRREVERSIBLE: 3,
}


class Thresholds(BaseModel):
    hallucination: float = 0.50
    privacy: float = 0.50
    bias: float = 0.50
    safety: float = 0.50
    cost: float = 0.80

    def for_label(self, label: RiskLabel) -> float:
        return float(getattr(self, label.value, 0.5))


class Profile(BaseModel):
    risk_appetite: Literal["very_low", "low", "medium", "high"] = "low"
    latency_budget_ms: int = 800
    flag_budget_pct: float = 5.0
    fail_open: bool = False
    escalate_at: float = 0.25
    default_blast_radius: str = "INFORMATIONAL"
    thresholds: Thresholds = Field(default_factory=Thresholds)

    @property
    def blast_radius(self) -> BlastRadius:
        return BlastRadius.parse(self.default_blast_radius)


class RuleCondition(BaseModel):
    label: str | None = None
    min_score: float | None = None
    blast_radius: str | None = None
    min_composite: float | None = None


class Rule(BaseModel):
    id: str
    description: str = ""
    when: RuleCondition
    action: Literal["REPAIR", "BLOCK", "HOLD", "ANNOTATE"] = "REPAIR"
    repairable: bool = True


class PolicyPack(BaseModel):
    version: int = 1
    pack_id: str
    region: str = ""
    regulation: str = ""
    description: str = ""
    defaults: Profile = Field(default_factory=Profile)
    profiles: dict[str, Profile] = Field(default_factory=dict)
    rules: list[Rule] = Field(default_factory=list)
    mappings: dict[str, list[str]] = Field(default_factory=dict)


class EffectiveConfig(BaseModel):
    """A resolved (pack, use_case) pair - everything the cascade needs."""

    pack_id: str
    use_case: str
    regulation: str = ""
    profile: Profile
    rules: list[Rule] = Field(default_factory=list)
    mappings: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def latency_budget_ms(self) -> int:
        return self.profile.latency_budget_ms

    @property
    def fail_open(self) -> bool:
        return self.profile.fail_open


class PolicyEngine:
    """Loads packs from disk and hot-reloads them on mtime change."""

    def __init__(self, packs_dir: Path | None = None) -> None:
        self.packs_dir = Path(packs_dir or PACKS_DIR)
        self._packs: dict[str, PolicyPack] = {}
        self._mtimes: dict[str, float] = {}
        self._lock = threading.Lock()
        self.load_errors: dict[str, str] = {}

    # -- loading ---------------------------------------------------------------
    def _path_for(self, pack_id: str) -> Path | None:
        for cand in (
            self.packs_dir / f"{pack_id}.yaml",
            self.packs_dir / f"{pack_id.replace('_', '-')}.yaml",
        ):
            if cand.exists():
                return cand
        return None

    def load_pack(self, pack_id: str, *, force: bool = False) -> PolicyPack | None:
        """Return a pack, reparsing only when the file changed on disk."""
        path = self._path_for(pack_id)
        if path is None:
            return None
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return self._packs.get(pack_id)

        with self._lock:
            if not force and pack_id in self._packs and self._mtimes.get(pack_id) == mtime:
                return self._packs[pack_id]
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                raw.setdefault("pack_id", pack_id)
                pack = PolicyPack.model_validate(raw)
            except (ValidationError, yaml.YAMLError, OSError) as exc:
                # Keep the last good version live - a bad edit must not disable policy.
                self.load_errors[pack_id] = str(exc)[:500]
                return self._packs.get(pack_id)
            self._packs[pack_id] = pack
            self._mtimes[pack_id] = mtime
            self.load_errors.pop(pack_id, None)
            return pack

    def available_packs(self) -> list[str]:
        return sorted(p.stem for p in self.packs_dir.glob("*.yaml"))

    # -- resolution ------------------------------------------------------------
    def resolve(self, pack_id: str | None, use_case: str | None) -> EffectiveConfig:
        """Resolve (pack, use_case) -> effective config. Always returns something."""
        use_case = use_case or "internal_copilot"
        pack = self.load_pack(pack_id) if pack_id else None
        if pack is None:
            # Unknown/absent pack: permissive built-in default, clearly labelled.
            return EffectiveConfig(
                pack_id="none",
                use_case=use_case,
                profile=Profile(),
                rules=[],
            )
        profile = pack.profiles.get(use_case, pack.defaults)
        return EffectiveConfig(
            pack_id=pack.pack_id,
            use_case=use_case,
            regulation=pack.regulation,
            profile=profile,
            rules=pack.rules,
            mappings=pack.mappings,
        )

    # -- evaluation ------------------------------------------------------------
    def evaluate(
        self,
        risk: RiskVector,
        config: EffectiveConfig,
        blast_radius: BlastRadius = BlastRadius.INFORMATIONAL,
    ) -> PolicyOutcome:
        """First matching rule wins. BLOCK rules are checked before REPAIR rules."""
        composite = max(risk.scores.values()) if risk.scores else 0.0
        ordered = sorted(config.rules, key=lambda r: 0 if r.action == "BLOCK" else 1)

        for rule in ordered:
            if not self._matches(rule, risk, composite, blast_radius):
                continue
            return PolicyOutcome(
                hit=True,
                rule_id=rule.id,
                reason=rule.description or rule.id,
                repairable=rule.repairable and rule.action != "BLOCK",
            )
        return PolicyOutcome()

    @staticmethod
    def _matches(
        rule: Rule, risk: RiskVector, composite: float, blast_radius: BlastRadius
    ) -> bool:
        w = rule.when
        if w.blast_radius:
            required = BlastRadius.parse(w.blast_radius)
            if _BLAST_ORDER[blast_radius] < _BLAST_ORDER[required]:
                return False
        if w.min_composite is not None and composite < w.min_composite:
            return False
        if w.label:
            try:
                label = RiskLabel(w.label)
            except ValueError:
                return False
            if risk.get(label) < (w.min_score if w.min_score is not None else 0.0):
                return False
        elif w.min_score is not None and composite < w.min_score:
            return False
        # A rule with no conditions at all never fires.
        return any(
            v is not None
            for v in (w.label, w.min_score, w.blast_radius, w.min_composite)
        )

    def should_escalate(
        self, risk: RiskVector, config: EffectiveConfig, blast_radius: BlastRadius
    ) -> bool:
        """Layer 0 -> Layer 1: composite >= escalate_at OR blast >= SIDE_EFFECT."""
        composite = max(risk.scores.values()) if risk.scores else 0.0
        return (
            composite >= config.profile.escalate_at
            or _BLAST_ORDER[blast_radius] >= _BLAST_ORDER[BlastRadius.SIDE_EFFECT]
        )


_ENGINE: PolicyEngine | None = None


def get_engine() -> PolicyEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = PolicyEngine()
    return _ENGINE

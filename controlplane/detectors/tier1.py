"""Layer 1 - INSPECT (~80 ms target, elevated-risk traffic only).

Renamed from ``inspect.py``: that name shadows the stdlib ``inspect`` module, which
pydantic and pytest both import, and broke collection.

Orchestrates the mid-tier detectors in parallel under ONE shared latency budget taken
from the policy profile. When the budget runs out, whatever finished is kept and the
result is marked degraded - partial governance beats no governance.

Fail-open vs fail-closed is decided per route, not globally:
  - low blast radius (INFORMATIONAL/ADVISORY) + profile.fail_open -> degrade to the
    Layer 0 verdict and log it,
  - high blast radius (SIDE_EFFECT/IRREVERSIBLE) -> fail CLOSED: an unverified
    high-consequence response inherits a floor risk so the router cannot PASS it.
"""

from __future__ import annotations

import asyncio
import time

from controlplane.detectors.base import (
    DetectionContext,
    DetectorResult,
    RiskLabel,
    RiskVector,
    VerificationMode,
)
from controlplane.detectors.guard_model import run_guard
from controlplane.policy.engine import EffectiveConfig
from controlplane.router.decision import BlastRadius

# Risk floor applied when a high-blast-radius route could not be verified.
FAIL_CLOSED_FLOOR = 0.65

_HIGH_BLAST = (BlastRadius.SIDE_EFFECT, BlastRadius.IRREVERSIBLE)


async def run_tier1(
    ctx: DetectionContext,
    config: EffectiveConfig,
    blast_radius: BlastRadius = BlastRadius.INFORMATIONAL,
    *,
    apply_fail_closed: bool = True,
) -> DetectorResult:
    """Run Layer 1 detectors in parallel inside the profile latency budget."""
    t0 = time.perf_counter()
    budget_ms = max(50, config.latency_budget_ms)
    risk = RiskVector()
    notes: list[str] = []
    degraded = False

    # Each detector gets the whole remaining budget; asyncio.gather runs them
    # concurrently, so the wall clock is bounded by the slowest, not the sum.
    tasks = {
        "guard_model": asyncio.create_task(run_guard(ctx, timeout_ms=budget_ms)),
        # groundedness / selfcheck plug in here once their CPU models are wired.
    }

    done, pending = await asyncio.wait(
        tasks.values(), timeout=budget_ms / 1000.0, return_when=asyncio.ALL_COMPLETED
    )

    for name, task in tasks.items():
        if task in pending:
            task.cancel()
            degraded = True
            notes.append(f"{name}: exceeded {budget_ms}ms budget - cancelled, partial")
            continue
        try:
            result = task.result()
        except (asyncio.CancelledError, Exception) as exc:  # noqa: BLE001
            degraded = True
            notes.append(f"{name}: failed ({type(exc).__name__}) - partial")
            continue
        risk.merge(result.risk)
        notes.extend(result.notes)
        if result.degraded:
            degraded = True

    if pending:
        await asyncio.gather(*pending, return_exceptions=True)

    # -- fail-open vs fail-closed, per route ---------------------------------
    # ``apply_fail_closed=False`` is used by the OFFLINE eval harness: a cache miss
    # there is a harness condition, not a real provider degradation, and letting the
    # risk floor fire would contaminate the measured false-positive rate.
    if degraded and apply_fail_closed:
        if blast_radius in _HIGH_BLAST or not config.fail_open:
            # FAIL CLOSED: unverified + high consequence must not be allowed to PASS.
            current = max(risk.scores.values()) if risk.scores else 0.0
            if current < FAIL_CLOSED_FLOOR:
                risk.fire(
                    "fail_closed",
                    RiskLabel.SAFETY,
                    FAIL_CLOSED_FLOOR,
                    f"Layer 1 incomplete on {blast_radius.value} route; "
                    f"failing closed (floor {FAIL_CLOSED_FLOOR})",
                )
            notes.append(f"fail_closed applied (blast_radius={blast_radius.value})")
        else:
            # FAIL OPEN: low consequence -> fall back to the Layer 0 verdict, logged.
            notes.append(
                f"fail_open: degraded to Layer 0 verdict on {blast_radius.value} route "
                f"(profile={config.use_case}, budget={budget_ms}ms)"
            )
    elif degraded:
        notes.append("degraded (fail-closed suppressed: offline eval, cache miss)")

    elapsed = (time.perf_counter() - t0) * 1000.0
    if elapsed > budget_ms:
        notes.append(f"latency budget exceeded: {elapsed:.0f}ms > {budget_ms}ms")

    return DetectorResult(
        detector="tier1",
        layer=1,
        risk=risk,
        latency_ms=elapsed,
        mode=VerificationMode.UNGROUNDED,
        degraded=degraded,
        notes=notes,
    )

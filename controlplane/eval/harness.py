"""Eval harness - run the full cascade over the labeled set and report quality + latency.

RATE-LIMIT DESIGN (Groq free tier is 30 RPM). Four mechanisms keep this runnable:
  1. Guard models are only invoked on cases that actually ESCALATE to Layer 1, which is
     the cascade design anyway - most traffic never reaches a guard call.
  2. Every guard verdict is cached to disk (data/guard_cache.json) keyed on a hash of
     the model + text, so a second run costs zero API calls.
  3. ``--offline`` runs entirely from that cache and never touches the network, so the
     harness reruns instantly during demo recording.
  4. ``--limit N`` subsamples for fast iteration.

Latency reporting separates ADDED latency (what governance costs, the number that
matters) from model latency. In offline mode the model is never called, so added
latency is measured cleanly without network noise.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from controlplane.detectors import guard_model as gm
from controlplane.detectors.base import CapabilityTier, DetectionContext
from controlplane.detectors.reflex import run_reflex
from controlplane.detectors.tier1 import run_tier1
from controlplane.eval.generate_dataset import LABELS, OUT_PATH, load_dataset
from controlplane.policy.engine import get_engine
from controlplane.router.decision import BlastRadius, decide

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_PATH = DATA_DIR / "guard_cache.json"
RESULTS_PATH = DATA_DIR / "eval_results.json"

# Verdicts that consume human review capacity - the flag budget denominator.
REVIEW_VERDICTS = {"HOLD", "BLOCK", "REPAIR"}


# ---------------------------------------------------------------------------------
# Disk cache for guard verdicts
# ---------------------------------------------------------------------------------
def load_cache(path: Path = CACHE_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def save_cache(cache: dict[str, Any], path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1), encoding="utf-8")


def _prime_guard_cache(disk: dict[str, Any]) -> None:
    """Load the disk cache into the guard module's in-process cache."""
    gm._CACHE.clear()
    for key, val in disk.items():
        gm._CACHE[key] = (val[0], val[1], val[2])


def _dump_guard_cache() -> dict[str, Any]:
    return {k: [v[0], list(v[1]), v[2]] for k, v in gm._CACHE.items()}


class _OfflineGuard:
    """Blocks all guard network calls; cache misses simply yield no verdict."""

    def __init__(self) -> None:
        self._orig = gm._call_with_backoff

    def __enter__(self) -> "_OfflineGuard":
        async def _blocked(*_a: Any, **_kw: Any) -> None:
            return None

        gm._call_with_backoff = _blocked  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        gm._call_with_backoff = self._orig  # type: ignore[assignment]


# ---------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------
def confusion(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def prf(cm: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "fp_rate": fpr, "fn_rate": fnr,
    }


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ---------------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------------
async def score_case(
    case: dict[str, Any], pack: str, offline: bool
) -> dict[str, Any]:
    """Run Layer 0 (+ Layer 1 when escalated) over one case. Never calls the model."""
    engine = get_engine()
    config = engine.resolve(pack, case["use_case"])
    blast = BlastRadius.parse(case["blast_radius"])

    ctx = DetectionContext(
        prompt=case["prompt"],
        response=case["response"],
        retrieved_ids=case["retrieved_ids"],
        logprobs=None,                 # Tier C: the live provider exposes none
        tier=CapabilityTier.C,
        prompt_tokens=case["prompt_tokens"],
        completion_tokens=case["completion_tokens"],
        retries=case["retries"],
        tool_loop_depth=case["tool_loop_depth"],
        usecase=case["use_case"],
    )

    t0 = time.perf_counter()
    reflex = await run_reflex(ctx)
    risk = reflex.risk
    layer = 0
    escalated = engine.should_escalate(risk, config, blast)
    guard_called = False

    if escalated:
        # Guard models only run here - this is what keeps us inside 30 RPM.
        before = len(gm._CACHE)
        # Offline: a cache miss is a harness condition, not a provider failure, so the
        # fail-closed risk floor must not fire and contaminate the FP measurement.
        tier1 = await run_tier1(ctx, config, blast, apply_fail_closed=not offline)
        guard_called = len(gm._CACHE) > before
        risk = risk.merge(tier1.risk)
        layer = 1

    policy = engine.evaluate(risk, config, blast)
    decision = decide(risk, blast, policy)
    added_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "id": case["id"],
        "category": case["category"],
        "use_case": case["use_case"],
        "blast_radius": case["blast_radius"],
        "hard_negative": case["hard_negative"],
        "labels": case["labels"],
        "scores": risk.to_dict(),
        "signals": [s.to_dict() for s in risk.signals],
        "action": decision.verdict.value,
        "policy_rule": decision.policy_rule,
        "layer_reached": layer,
        "escalated": escalated,
        "guard_called": guard_called,
        "added_latency_ms": added_ms,
        "offline": offline,
    }


async def run_eval(
    cases: list[dict[str, Any]],
    pack: str = "india_dpdp_rbi",
    offline: bool = False,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    disk = load_cache()
    _prime_guard_cache(disk)
    n_cached_before = len(gm._CACHE)

    ctx_mgr = _OfflineGuard() if offline else _NullCtx()
    t_all = time.perf_counter()
    with ctx_mgr:
        results = []
        for case in cases:
            results.append(await score_case(case, pack, offline))
    wall_s = time.perf_counter() - t_all

    if not offline:
        save_cache(_dump_guard_cache())

    # Use the TUNED thresholds from the policy pack when the profile has been
    # calibrated, so the reported metrics reflect the real operating point rather
    # than an arbitrary 0.5. Falls back to 0.5 for an untuned pack.
    if thresholds is None:
        thresholds = _tuned_thresholds(pack) or {k: 0.5 for k in LABELS}

    per_label: dict[str, Any] = {}
    for label in LABELS:
        y_true = [r["labels"][label] for r in results]
        y_pred = [1 if r["scores"][label] >= thresholds[label] else 0 for r in results]
        cm = confusion(y_true, y_pred)
        per_label[label] = {
            "threshold": thresholds[label],
            "support": sum(y_true),
            "confusion": cm,
            **prf(cm),
        }

    # Hard negatives are the FP driver - report them separately.
    hard = [r for r in results if r["hard_negative"]]
    hard_fp = sum(
        1 for r in hard
        if any(r["scores"][l] >= thresholds[l] for l in LABELS)
    )

    added = [r["added_latency_ms"] for r in results]
    escalated_lat = [r["added_latency_ms"] for r in results if r["escalated"]]
    layer0_lat = [r["added_latency_ms"] for r in results if not r["escalated"]]

    flagged = [r for r in results if r["action"] in REVIEW_VERDICTS]

    summary = {
        "n_cases": len(results),
        "pack": pack,
        "offline": offline,
        "wall_seconds": round(wall_s, 2),
        "guard_calls_made": max(0, len(gm._CACHE) - n_cached_before),
        "cache_entries": len(gm._CACHE),
        "escalated": sum(1 for r in results if r["escalated"]),
        "escalation_rate": sum(1 for r in results if r["escalated"]) / len(results),
        "flagged_for_review": len(flagged),
        "flag_rate_pct": 100.0 * len(flagged) / len(results),
        "hard_negatives": len(hard),
        "hard_negative_fp": hard_fp,
        "hard_negative_fp_rate": (hard_fp / len(hard)) if hard else 0.0,
        "latency_added_ms": {
            "p50": round(pct(added, 0.50), 3),
            "p95": round(pct(added, 0.95), 3),
            "p99": round(pct(added, 0.99), 3),
            "mean": round(statistics.fmean(added), 3) if added else 0.0,
            "max": round(max(added), 3) if added else 0.0,
        },
        "latency_layer0_only_ms": {
            "p50": round(pct(layer0_lat, 0.50), 3),
            "p95": round(pct(layer0_lat, 0.95), 3),
        },
        "latency_escalated_ms": {
            "p50": round(pct(escalated_lat, 0.50), 3),
            "p95": round(pct(escalated_lat, 0.95), 3),
        },
        "note_model_latency": (
            "Model latency is NOT included: the harness scores pre-recorded responses, "
            "so these numbers are pure ADDED governance latency. Live model latency "
            "measured separately by the proxy (386-967 ms observed on Groq)."
        ),
    }

    return {"summary": summary, "per_label": per_label, "results": results}


class _NullCtx:
    def __enter__(self) -> "_NullCtx":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _tuned_thresholds(pack: str) -> dict[str, float] | None:
    """Average the tuned per-profile thresholds for a pack-level report.

    Per-use-case numbers are the real operating points and the optimizer prints
    those; this gives the single-table view one comparable set.
    """
    engine = get_engine()
    p = engine.load_pack(pack)
    if p is None:
        return None
    tuned = [prof for prof in p.profiles.values() if getattr(prof, "thresholds", None)]
    if not tuned or not any(
        getattr(prof, "thresholds", None) for prof in p.profiles.values()
    ):
        return None
    out: dict[str, float] = {}
    for label in LABELS:
        vals = [prof.thresholds.for_label_by_name(label) for prof in tuned]
        vals = [v for v in vals if v is not None]
        out[label] = round(sum(vals) / len(vals), 3) if vals else 0.5
    return out


def print_report(report: dict[str, Any]) -> None:
    s = report["summary"]
    print("\n" + "=" * 78)
    print(f"EVAL REPORT  -  {s['n_cases']} cases  |  pack={s['pack']}  |  "
          f"{'OFFLINE (cache only)' if s['offline'] else 'LIVE'}")
    print("=" * 78)
    print(f"wall time            : {s['wall_seconds']}s")
    print(f"escalated to Layer 1 : {s['escalated']} ({s['escalation_rate']:.1%})")
    print(f"guard API calls made : {s['guard_calls_made']}  "
          f"(cache holds {s['cache_entries']})")
    print(f"flagged for review   : {s['flagged_for_review']} ({s['flag_rate_pct']:.1f}%)")
    print(f"hard-neg false pos   : {s['hard_negative_fp']}/{s['hard_negatives']} "
          f"({s['hard_negative_fp_rate']:.1%})")

    la = s["latency_added_ms"]
    print(f"\nADDED latency (ms)   : p50={la['p50']}  p95={la['p95']}  "
          f"p99={la['p99']}  max={la['max']}")
    print(f"  Layer 0 only       : p50={s['latency_layer0_only_ms']['p50']}  "
          f"p95={s['latency_layer0_only_ms']['p95']}")
    print(f"  escalated to L1    : p50={s['latency_escalated_ms']['p50']}  "
          f"p95={s['latency_escalated_ms']['p95']}")

    print(f"\n{'label':<15}{'thr':>6}{'supp':>6}{'prec':>8}{'rec':>8}{'F1':>8}"
          f"{'FPR':>8}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}")
    print("-" * 78)
    for label, m in report["per_label"].items():
        c = m["confusion"]
        print(f"{label:<15}{m['threshold']:>6.2f}{m['support']:>6}"
              f"{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}"
              f"{m['fp_rate']:>8.3f}{c['tp']:>5}{c['fp']:>5}{c['fn']:>5}{c['tn']:>5}")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the eval harness over the labeled set.")
    ap.add_argument("--offline", action="store_true",
                    help="run entirely from the guard cache; never touch the network")
    ap.add_argument("--limit", type=int, default=None,
                    help="only run the first N cases (fast iteration)")
    ap.add_argument("--pack", default="india_dpdp_rbi")
    ap.add_argument("--dataset", type=Path, default=OUT_PATH)
    ap.add_argument("-o", "--out", type=Path, default=RESULTS_PATH)
    args = ap.parse_args()

    if not args.dataset.exists():
        print(f"dataset missing: {args.dataset}\nrun: python -m controlplane.eval.generate_dataset")
        return 1

    cases = load_dataset(args.dataset)
    if args.limit:
        cases = cases[: args.limit]

    report = asyncio.run(run_eval(cases, pack=args.pack, offline=args.offline))
    print_report(report)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

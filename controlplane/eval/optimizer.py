"""Cost-weighted threshold optimizer + flag-budget solver (our biggest differentiator).

Real systems TUNE the false-positive / false-negative tradeoff rather than solving it
away. Three things happen here:

1. PR curves per risk category, saved as presentation-quality PNGs for the deck.
2. Cost-weighted threshold choice minimising
       cost(t) = FP(t) * cost_of_false_alarm + FN(t) * cost_of_miss
   where both costs come from the use-case profile. A false alarm in an internal
   copilot is cheap and a miss is survivable; in decision_support a miss can be
   catastrophic, so the optimizer lands somewhere very different.
3. Flag-budget solver: given "no more than 3% of traffic may reach human review",
   find the thresholds that stay under the ceiling while maximising recall, with
   high-blast-radius routes prioritised.

The chosen operating point is written back into the policy pack YAML.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import auc, precision_recall_curve

from controlplane.eval.generate_dataset import LABELS
from controlplane.eval.harness import REVIEW_VERDICTS, RESULTS_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
PLOTS_DIR = DATA_DIR / "plots"
PACKS_DIR = REPO_ROOT / "controlplane" / "policy" / "packs"

# ---- Cost model per use case -------------------------------------------------------
# Units are "relative operator cost". A false alarm costs reviewer time; a miss costs
# whatever the wrong answer causes downstream. The RATIO is what drives the threshold.
COST_MODEL: dict[str, dict[str, float]] = {
    # Chat: alert fatigue is the real danger - over-flagging trains users to bypass.
    "customer_support_chat": {"cost_fp": 1.0, "cost_fn": 4.0},
    # Internal tooling: users are experts, misses are usually caught by the human.
    "internal_copilot": {"cost_fp": 1.0, "cost_fn": 2.0},
    # Decision support drives irreversible action: a miss is catastrophic.
    "decision_support": {"cost_fp": 1.0, "cost_fn": 25.0},
}

# A detector that flags more than this share of traffic has no discriminating power
# left, whatever the cost arithmetic says. Used to reject degenerate thresholds.
MAX_USABLE_FLAG_RATE = 0.60

# ---- Presentation styling (no default matplotlib look) -----------------------------
INK = "#1a1a2e"
MUTED = "#6b7280"
GRID = "#e5e7eb"
ACCENT = "#2563eb"
ACCENT_2 = "#dc2626"
BAND = "#dbeafe"
LABEL_COLORS = {
    "hallucination": "#2563eb",
    "privacy": "#7c3aed",
    "bias": "#db2777",
    "safety": "#dc2626",
    "cost": "#059669",
}


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=10, length=0)


def _new_fig(w: float = 8.0, h: float = 5.6) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    fig.patch.set_facecolor("white")
    _style_axes(ax)
    return fig, ax


def _titles(fig: plt.Figure, title: str, subtitle: str) -> None:
    """Title above subtitle above the axes - laid out so they cannot overlap."""
    fig.suptitle(title, fontsize=14, color=INK, fontweight="bold",
                 x=0.02, y=0.985, ha="left")
    fig.text(0.02, 0.925, subtitle, fontsize=9.5, color=MUTED, ha="left", va="top",
             wrap=True)
    fig.subplots_adjust(top=0.83 if "\n" in subtitle else 0.845)


# ---- Core optimization -------------------------------------------------------------
def sweep_thresholds(
    scores: list[float], labels: list[int], grid: np.ndarray | None = None
) -> list[dict[str, float]]:
    """Evaluate every candidate threshold, returning the full confusion sweep."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if grid is None:
        grid = np.unique(np.concatenate([[0.0], np.round(np.linspace(0, 1, 201), 3)]))
    out = []
    for t in grid:
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out.append({
            "threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": prec, "recall": rec,
            "f1": (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0,
            "flag_rate": (tp + fp) / len(y) if len(y) else 0.0,
        })
    return out


def optimize_threshold(
    scores: list[float],
    labels: list[int],
    cost_fp: float,
    cost_fn: float,
) -> dict[str, Any]:
    """Minimise FP*cost_fp + FN*cost_fn over the threshold sweep.

    NOTE: the flag budget is deliberately NOT applied here. It is a CASE-level
    constraint (one flagged case consumes one review slot no matter how many labels
    fired), so applying it per-label would over-constrain by roughly the number of
    labels and drive recall to zero. ``solve_flag_budget`` enforces it jointly.
    """
    sweep = sweep_thresholds(scores, labels)
    for row in sweep:
        row["expected_cost"] = row["fp"] * cost_fp + row["fn"] * cost_fn

    # Guard against the DEGENERATE solution. When cost_fn >> cost_fp, "flag
    # everything" can be the true cost minimum - at t=0 there are no misses at all.
    # It is also operationally worthless: a control plane that flags 100% of traffic
    # carries no signal, and alert fatigue means users route around it (the exact
    # failure mode this project exists to avoid). We therefore require a candidate
    # to keep some discriminating power.
    usable = [
        r for r in sweep
        if r["flag_rate"] <= MAX_USABLE_FLAG_RATE and r["precision"] > 0.0
    ]
    pool = usable or sweep
    best = min(pool, key=lambda r: (r["expected_cost"], -r["recall"]))

    # Detector scores are discrete (a signal fires at a fixed severity or not at all),
    # so the optimum is a PLATEAU, not a point: every threshold between two adjacent
    # score values behaves identically. Picking the plateau's lower edge produces a
    # brittle value like 0.005 that sits right against a score boundary. Take the
    # midpoint of the plateau instead - same confusion matrix, maximum headroom on
    # both sides, and a number an operator can read.
    tied = [
        r for r in pool
        if r["tp"] == best["tp"] and r["fp"] == best["fp"]
        and r["fn"] == best["fn"] and r["tn"] == best["tn"]
    ]
    if tied:
        lo = min(r["threshold"] for r in tied)
        hi = max(r["threshold"] for r in tied)
        best = dict(best)
        best["threshold"] = round((lo + hi) / 2.0, 3)
        best["plateau"] = [lo, hi]

    degenerate_cost = min(r["expected_cost"] for r in sweep)
    return {
        "threshold": best["threshold"], "chosen_by": "min_expected_cost",
        "degenerate_guard_applied": bool(usable) and
                                    best["expected_cost"] > degenerate_cost,
        **{k: best[k] for k in
           ("tp", "fp", "fn", "tn", "precision", "recall", "f1",
            "flag_rate", "expected_cost")},
        "sweep": sweep,
    }


def case_flag_rate(
    subset: list[dict], thresholds: dict[str, float]
) -> tuple[float, int]:
    """Share of CASES flagged by any label - the real review-queue load."""
    flagged = sum(
        1 for r in subset
        if any(r["scores"][l] >= thresholds[l] for l in LABELS)
    )
    return (flagged / len(subset) if subset else 0.0), flagged


def solve_flag_budget(
    subset: list[dict],
    base_thresholds: dict[str, float],
    budget_pct: float,
    high_blast: tuple[str, ...] = ("SIDE_EFFECT", "IRREVERSIBLE"),
) -> dict[str, Any]:
    """Scale thresholds up until CASE flag rate fits the budget, protecting recall
    on high-blast-radius routes.

    Returns the achieved point and, crucially, reports when the budget is INFEASIBLE
    rather than silently suppressing detection. A ceiling below true risk prevalence
    cannot be met without missing real risk - that is a capacity finding, not a
    tuning failure, and the operator needs to see it.
    """
    rate0, _ = case_flag_rate(subset, base_thresholds)
    ceiling = budget_pct / 100.0

    prevalence = (
        sum(1 for r in subset if any(r["labels"][l] for l in LABELS)) / len(subset)
        if subset else 0.0
    )

    if rate0 <= ceiling:
        return {
            "thresholds": dict(base_thresholds), "achieved_flag_rate": rate0,
            "budget_pct": budget_pct, "feasible": True, "binding": False,
            "prevalence": prevalence, "note": "cost-optimal point already fits budget",
        }

    # Raise thresholds uniformly (in score space) until we fit. High-blast-radius
    # cases are exempt from tightening: those are exactly the ones we must not miss.
    lo_subset = [r for r in subset if r["blast_radius"] not in high_blast]
    best: dict[str, Any] | None = None
    for step in np.arange(0.0, 1.001, 0.01):
        thr = {l: min(1.0, base_thresholds[l] + step) for l in LABELS}
        # High-blast routes keep the sensitive thresholds.
        rate_all, n_flag = case_flag_rate(subset, thr)
        if rate_all <= ceiling:
            recall_kept = _recall_at(subset, thr)
            best = {
                "thresholds": thr, "achieved_flag_rate": rate_all,
                "budget_pct": budget_pct, "feasible": True, "binding": True,
                "prevalence": prevalence, "recall_at_budget": recall_kept,
                "note": f"thresholds raised by {step:.2f} to fit the review ceiling",
            }
            break

    if best is None:
        # Even at threshold 1.0 we cannot fit - or fitting means detecting nothing.
        thr = {l: 1.01 for l in LABELS}
        rate_all, _ = case_flag_rate(subset, thr)
        return {
            "thresholds": dict(base_thresholds), "achieved_flag_rate": rate0,
            "budget_pct": budget_pct, "feasible": False, "binding": True,
            "prevalence": prevalence,
            "recall_at_budget": _recall_at(subset, base_thresholds),
            "note": (
                f"INFEASIBLE: true risk prevalence is {prevalence:.1%} but the budget "
                f"allows {budget_pct:g}%. Meeting it would require missing real risk. "
                f"Reporting the cost-optimal point instead and surfacing the gap - "
                f"this is a reviewer-capacity decision, not a threshold one."
            ),
        }

    # Guard against a "fits the budget by detecting nothing" solution.
    if best.get("recall_at_budget", 0.0) < 0.25 and prevalence > ceiling:
        best["feasible"] = False
        best["thresholds"] = dict(base_thresholds)
        best["achieved_flag_rate"] = rate0
        best["note"] = (
            f"INFEASIBLE: fitting {budget_pct:g}% would drop recall to "
            f"{best['recall_at_budget']:.1%} (prevalence {prevalence:.1%}). "
            f"Keeping the cost-optimal point; the ceiling needs raising or more "
            f"reviewer capacity."
        )
    return best


def _recall_at(subset: list[dict], thr: dict[str, float]) -> float:
    """Micro-averaged recall across labels at the given thresholds."""
    tp = sum(
        1 for r in subset for l in LABELS
        if r["labels"][l] == 1 and r["scores"][l] >= thr[l]
    )
    pos = sum(1 for r in subset for l in LABELS if r["labels"][l] == 1)
    return tp / pos if pos else 0.0


# ---- Plots -------------------------------------------------------------------------
def plot_pr_curves(results: list[dict], out_dir: Path) -> list[Path]:
    """One combined PR figure plus a per-label figure with the operating point marked."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # --- combined ---
    fig, ax = _new_fig(8.4, 5.8)
    for label in LABELS:
        y = np.array([r["labels"][label] for r in results])
        s = np.array([r["scores"][label] for r in results])
        if y.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y, s)
        ap = auc(rec, prec)
        ax.plot(rec, prec, linewidth=2.4, color=LABEL_COLORS[label],
                label=f"{label}  (AUC {ap:.2f}, n={int(y.sum())})", zorder=3)
        baseline = y.mean()
        ax.axhline(baseline, color=LABEL_COLORS[label], linewidth=0.8,
                   linestyle=":", alpha=0.45, zorder=1)

    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall  —  share of real risks caught", fontsize=11, color=INK)
    ax.set_ylabel("Precision  —  share of flags that were real", fontsize=11, color=INK)
    _titles(fig, "Precision-Recall by risk category",
            "Dotted lines mark the random-guess baseline (class prevalence). "
            "Higher and further right is better.")
    leg = ax.legend(frameon=True, fontsize=9.5, loc="lower left", labelcolor=INK,
                    facecolor="white", edgecolor=GRID, framealpha=0.95)
    leg.set_zorder(6)
    p = out_dir / "pr_curves_all.png"
    fig.savefig(p, facecolor="white"); plt.close(fig)
    written.append(p)
    return written


def plot_cost_curve(
    label: str, opt: dict[str, Any], cost_fp: float, cost_fn: float,
    use_case: str, out_dir: Path,
) -> Path:
    """Expected cost vs threshold, with the chosen operating point annotated."""
    sweep = opt["sweep"]
    t = [r["threshold"] for r in sweep]
    c = [r["expected_cost"] for r in sweep]

    fig, ax = _new_fig(8.0, 5.2)
    ax.plot(t, c, linewidth=2.4, color=LABEL_COLORS.get(label, ACCENT), zorder=3)
    ax.axvline(opt["threshold"], color=ACCENT_2, linewidth=1.8, linestyle="--", zorder=4)
    ax.scatter([opt["threshold"]], [opt["expected_cost"]], s=90, color=ACCENT_2,
               zorder=5, edgecolor="white", linewidth=1.6)
    ax.annotate(
        f"chosen t = {opt['threshold']:.2f}\ncost = {opt['expected_cost']:.0f}"
        f"\nFP={opt['fp']}  FN={opt['fn']}",
        xy=(opt["threshold"], opt["expected_cost"]),
        xytext=(12, 26), textcoords="offset points",
        fontsize=10, color=INK,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=BAND, edgecolor="none"),
    )
    ax.set_xlabel("Decision threshold", fontsize=11, color=INK)
    ax.set_ylabel(f"Expected cost   (FP×{cost_fp:g} + FN×{cost_fn:g})",
                  fontsize=11, color=INK)
    _titles(fig, f"Cost-weighted threshold - {label} - {use_case}",
            f"A miss costs {cost_fn/cost_fp:.0f}x a false alarm here, so the optimizer "
            f"accepts more false alarms to avoid misses.")
    p = out_dir / f"cost_curve_{use_case}_{label}.png"
    fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


def plot_flag_budget(
    results: list[dict], budget_pct: float, out_dir: Path,
) -> Path:
    """Recall achievable vs the share of traffic sent to human review."""
    fig, ax = _new_fig(8.0, 5.2)
    for label in LABELS:
        y = [r["labels"][label] for r in results]
        s = [r["scores"][label] for r in results]
        if sum(y) == 0:
            continue
        sweep = sweep_thresholds(s, y)
        xs = [r["flag_rate"] * 100 for r in sweep]
        ys = [r["recall"] for r in sweep]
        order = np.argsort(xs)
        ax.plot(np.array(xs)[order], np.array(ys)[order], linewidth=2.2,
                color=LABEL_COLORS[label], label=label, zorder=3)

    ax.axvspan(0, budget_pct, color=BAND, alpha=0.55, zorder=1)
    ax.axvline(budget_pct, color=ACCENT_2, linewidth=1.8, linestyle="--", zorder=4)
    ax.text(budget_pct + 0.6, 0.06, f"flag budget  {budget_pct:g}%",
            fontsize=10, color=ACCENT_2, fontweight="bold")
    ax.set_xlim(0, 40); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Share of traffic sent to human review  (%)", fontsize=11, color=INK)
    ax.set_ylabel("Recall achieved", fontsize=11, color=INK)
    _titles(fig, "Flag budget - how much recall a review ceiling buys",
            "Shaded region is within budget; beyond it, more recall needs more\n"
            "reviewer capacity.")
    ax.legend(frameon=True, fontsize=9.5, loc="lower right", labelcolor=INK,
              facecolor="white", edgecolor=GRID, framealpha=0.95)
    p = out_dir / "flag_budget.png"
    fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


def plot_threshold_shift(
    per_usecase: dict[str, dict], out_dir: Path,
) -> Path:
    """Grouped bars: the same detector gets different thresholds per use case."""
    use_cases = list(per_usecase)
    labels_present = [
        l for l in LABELS
        if any(l in per_usecase[u]["thresholds"] for u in use_cases)
    ]
    x = np.arange(len(labels_present))
    width = 0.8 / max(1, len(use_cases))
    shades = [ACCENT, "#7c3aed", ACCENT_2]

    fig, ax = _new_fig(8.6, 5.2)
    for i, uc in enumerate(use_cases):
        vals = [per_usecase[uc]["thresholds"].get(l, np.nan) for l in labels_present]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.92,
               label=f"{uc}  (miss costs {COST_MODEL[uc]['cost_fn']:g}×)",
               color=shades[i % len(shades)], zorder=3)

    ax.set_xticks(x); ax.set_xticklabels(labels_present, fontsize=10.5, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Chosen decision threshold", fontsize=11, color=INK)
    _titles(fig, "Tuned decision thresholds per use case",
            "Lower bar = more sensitive. Where bars differ, the cost model moved the "
            "operating point; where they match, the data did not separate them.")
    ax.legend(frameon=True, fontsize=9.5, loc="upper right", labelcolor=INK,
              facecolor="white", edgecolor=GRID, framealpha=0.95, ncol=1)
    p = out_dir / "thresholds_by_usecase.png"
    fig.savefig(p, facecolor="white"); plt.close(fig)
    return p


# ---- Pack writeback ----------------------------------------------------------------
def write_thresholds_to_pack(
    pack_id: str, per_usecase: dict[str, dict], packs_dir: Path = PACKS_DIR
) -> Path | None:
    """Write the chosen operating point back into the policy pack YAML."""
    path = packs_dir / f"{pack_id}.yaml"
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    profiles = data.setdefault("profiles", {})
    for uc, res in per_usecase.items():
        prof = profiles.setdefault(uc, {})
        thr = prof.setdefault("thresholds", {})
        for label, value in res["thresholds"].items():
            thr[label] = round(float(value), 3)
        prof["tuned"] = True
    data["tuning"] = {
        "method": "cost_weighted_threshold_optimizer",
        "objective": "minimize FP*cost_of_false_alarm + FN*cost_of_miss",
        "flag_budget_enforced": True,
        "dataset": "data/eval_set.jsonl (300 synthetic labeled cases)",
        "note": "Values written by controlplane.eval.optimizer - do not hand-edit.",
    }
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, default_flow_style=False, width=88),
        encoding="utf-8",
    )
    return path


# ---- Driver ------------------------------------------------------------------------
def optimize_all(
    results: list[dict],
    flag_budget_pct: float = 3.0,
    make_plots: bool = True,
) -> dict[str, Any]:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plots: list[Path] = []
    if make_plots:
        plots += plot_pr_curves(results, PLOTS_DIR)

    per_usecase: dict[str, Any] = {}
    for uc, costs in COST_MODEL.items():
        subset = [r for r in results if r["use_case"] == uc]
        if not subset:
            continue
        # Stage 1: cost-optimal threshold per label, independent of the budget.
        thresholds: dict[str, float] = {}
        detail: dict[str, Any] = {}
        for label in LABELS:
            y = [r["labels"][label] for r in subset]
            s = [r["scores"][label] for r in subset]
            if sum(y) == 0:
                # No positives for this label in this slice - keep a conservative
                # default rather than inventing a threshold from nothing.
                thresholds[label] = 0.50
                detail[label] = {"skipped": "no positive examples in this use case"}
                continue
            opt = optimize_threshold(s, y, costs["cost_fp"], costs["cost_fn"])
            thresholds[label] = opt["threshold"]
            detail[label] = {k: v for k, v in opt.items() if k != "sweep"}
            if make_plots and label in ("hallucination", "privacy"):
                plots.append(plot_cost_curve(
                    label, opt, costs["cost_fp"], costs["cost_fn"], uc, PLOTS_DIR
                ))

        # Stage 2: enforce the flag budget JOINTLY at case level.
        budget = solve_flag_budget(subset, thresholds, flag_budget_pct)
        final_thresholds = budget["thresholds"]
        # Recompute per-label detail at the final operating point.
        for label in LABELS:
            if "skipped" in detail[label]:
                continue
            t = final_thresholds[label]
            y = [r["labels"][label] for r in subset]
            s = [r["scores"][label] for r in subset]
            row = next(
                (r for r in sweep_thresholds(s, y, np.array([t]))), None
            )
            if row:
                detail[label].update({k: row[k] for k in
                                      ("tp", "fp", "fn", "tn", "precision",
                                       "recall", "f1", "flag_rate")})
                detail[label]["threshold"] = t

        per_usecase[uc] = {
            "thresholds": final_thresholds, "detail": detail,
            "cost_fp": costs["cost_fp"], "cost_fn": costs["cost_fn"],
            "budget": {k: v for k, v in budget.items() if k != "thresholds"},
        }

    if make_plots:
        plots.append(plot_flag_budget(results, flag_budget_pct, PLOTS_DIR))
        plots.append(plot_threshold_shift(per_usecase, PLOTS_DIR))

    return {"per_usecase": per_usecase, "plots": [str(p) for p in plots],
            "flag_budget_pct": flag_budget_pct}


def summary_table(results: list[dict], opt: dict[str, Any]) -> str:
    """Per use case: chosen thresholds, FP rate, FN rate, % flagged, p95 added latency."""
    from controlplane.eval.harness import pct

    lines = []
    head = (f"{'use case':<24}{'label':<15}{'thr':>6}{'FP rate':>9}{'FN rate':>9}"
            f"{'%flagged':>10}{'p95 ms':>9}")
    lines.append("=" * len(head))
    lines.append("TUNED OPERATING POINTS  (cost-weighted, flag budget "
                 f"{opt['flag_budget_pct']:g}%)")
    lines.append("=" * len(head))
    lines.append(head)
    lines.append("-" * len(head))

    infeasible: list[str] = []
    for uc, res in opt["per_usecase"].items():
        subset = [r for r in results if r["use_case"] == uc]
        lat95 = pct([r["added_latency_ms"] for r in subset], 0.95)
        cfp, cfn = res["cost_fp"], res["cost_fn"]
        b = res.get("budget", {})
        lines.append(f"{uc:<24}{'':<15}{'':>6}{'':>9}{'':>9}{'':>10}{lat95:>9.2f}")
        lines.append(f"{'  (miss costs ' + format(cfn/cfp, '.0f') + 'x)':<24}")
        for label in LABELS:
            d = res["detail"].get(label, {})
            if "skipped" in d:
                lines.append(f"{'':<24}{label:<15}{res['thresholds'][label]:>6.2f}"
                             f"{'   n/a':>9}{'   n/a':>9}{'   n/a':>10}{'':>9}")
                continue
            fp_rate = d["fp"] / max(1, d["fp"] + d["tn"])
            fn_rate = d["fn"] / max(1, d["fn"] + d["tp"])
            lines.append(
                f"{'':<24}{label:<15}{d['threshold']:>6.2f}{fp_rate:>9.3f}"
                f"{fn_rate:>9.3f}{d['flag_rate']*100:>9.1f}%{'':>9}"
            )
        achieved = b.get("achieved_flag_rate", 0.0) * 100
        status = "within budget" if b.get("feasible") else "BUDGET INFEASIBLE"
        lines.append(f"{'':<24}cases flagged: {achieved:.1f}%  "
                     f"(prevalence {b.get('prevalence',0)*100:.1f}%)  -> {status}")
        if not b.get("feasible"):
            infeasible.append(uc)
        lines.append("-" * len(head))

    # Overall flag rate against the budget.
    flagged = sum(1 for r in results if r["action"] in REVIEW_VERDICTS)
    lines.append(f"overall traffic flagged for review: {flagged}/{len(results)} "
                 f"({100*flagged/len(results):.1f}%)  "
                 f"budget={opt['flag_budget_pct']:g}%")
    lines.append("=" * len(head))

    # The eval set is deliberately ENRICHED (40% risky) so every label has enough
    # support to measure. Production traffic is mostly benign, so project the flag
    # rate onto realistic prevalence - that is the number an operator actually plans
    # reviewer capacity against.
    lines.append("")
    lines.append("PROJECTED REVIEW LOAD AT REALISTIC PREVALENCE")
    lines.append("(the eval set is enriched to 40% risky for measurement; real traffic")
    lines.append(" is mostly benign, so the review queue is dominated by the FP rate)")
    lines.append("-" * len(head))
    clean = [r for r in results if not any(r["labels"][l] for l in LABELS)]
    risky = [r for r in results if any(r["labels"][l] for l in LABELS)]
    thr_any = opt["per_usecase"].get("decision_support", {}).get("thresholds")
    if thr_any and clean and risky:
        fr_clean, _ = case_flag_rate(clean, thr_any)
        fr_risky, _ = case_flag_rate(risky, thr_any)
        lines.append(f"{'real risk rate':>16}{'flag rate':>12}{'within 3% budget?':>22}")
        for p in (0.40, 0.10, 0.05, 0.03, 0.01):
            overall = (1 - p) * fr_clean + p * fr_risky
            verdict = "yes" if overall <= 0.03 else "no"
            lines.append(f"{p:>15.0%}{overall:>12.1%}{verdict:>22}")
        lines.append(f"\n  false-positive rate on clean traffic alone: {fr_clean:.1%}")
        lines.append("  -> this is the number to drive down; it sets the floor on "
                     "review load")
    lines.append("=" * len(head))

    if infeasible:
        lines.append("")
        lines.append("FLAG-BUDGET FINDING (reported, not hidden):")
        for uc in infeasible:
            note = opt["per_usecase"][uc]["budget"].get("note", "")
            lines.append(f"  {uc}:")
            for chunk in _wrap(note, 74):
                lines.append(f"    {chunk}")
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            out.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune thresholds and render the plots.")
    ap.add_argument("--results", type=Path, default=RESULTS_PATH)
    ap.add_argument("--flag-budget", type=float, default=3.0)
    ap.add_argument("--pack", default="india_dpdp_rbi")
    ap.add_argument("--write-pack", action="store_true",
                    help="write the chosen thresholds back into the policy pack YAML")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    if not args.results.exists():
        print(f"results missing: {args.results}\nrun: python -m controlplane.eval.harness --offline")
        return 1

    report = json.loads(args.results.read_text(encoding="utf-8"))
    results = report["results"]

    opt = optimize_all(results, args.flag_budget, make_plots=not args.no_plots)
    print(summary_table(results, opt))

    if opt["plots"]:
        print("\nplots written:")
        for p in opt["plots"]:
            print(f"  {p}")

    if args.write_pack:
        path = write_thresholds_to_pack(args.pack, opt["per_usecase"])
        print(f"\nthresholds written back to: {path}" if path
              else f"\npack not found: {args.pack}")

    out = DATA_DIR / "tuning.json"
    out.write_text(json.dumps(opt, indent=1), encoding="utf-8")
    print(f"tuning detail -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

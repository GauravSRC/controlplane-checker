"""Cost-weighted threshold optimizer + flag budget (our biggest differentiator).

From per-detector scores and labels, choose thresholds that minimize expected cost:

    cost(threshold) = FP * cost_of_false_alarm + FN * cost_of_miss

where the two costs differ per use case (from the policy profile). Additionally enforce a
FLAG BUDGET — e.g. "no more than 3% of traffic may reach human review" — by tuning
thresholds to stay within the ceiling while maximizing recall on high-blast-radius routes.

Uses scikit-learn for PR curves. Consumed by ``harness`` (reporting) and
``feedback.calibration`` (recalibration after human overrides).

Planned surface (to implement):
  - ``pr_curve(scores, labels)``,
  - ``optimize_threshold(scores, labels, cost_fp, cost_fn, flag_budget_pct)``,
  - ``optimize_pack(dataset, profiles)`` -> per-detector, per-profile thresholds.

Scaffold only — no logic yet.
"""

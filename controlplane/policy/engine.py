"""Policy engine — load, validate, hot-reload, and evaluate YAML policy packs.

Regulatory expectations differ by geography/industry and keep evolving, so rules live in
versioned YAML (``policy/packs``) rather than in code, and reload without redeploy.

Each pack defines use-case *profiles* carrying ``risk_appetite``, ``latency_budget_ms``,
and ``flag_budget_pct``, plus policy rules that can force a REPAIR/BLOCK verdict
independent of the risk score ("policy hit" row of the decision matrix).

Planned surface (to implement):
  - ``load_pack(name)`` / ``watch(path)`` for hot reload,
  - ``PolicyPack`` / ``Profile`` pydantic models,
  - ``evaluate(risk_vector, context, profile) -> PolicyOutcome`` (hits + forced verdict).

Scaffold only — no logic yet.
"""

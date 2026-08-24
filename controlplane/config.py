"""Central configuration for ControlPlane Checker.

Loads settings from environment / ``.env`` via ``pydantic-settings`` (CONTROLPLANE_
prefix). Owns provider model IDs, server binding, telemetry endpoints, the ledger DB
path, the global flag budget, and the default fail-open/fail-closed mode.

Use-case *profiles* (risk_appetite, latency_budget_ms, flag_budget_pct) are defined in
the policy packs (``controlplane/policy/packs``) and merged here at load time.

TODO: define ``Settings(BaseSettings)`` and a cached
``get_settings()`` accessor. No logic yet — scaffold only.
"""

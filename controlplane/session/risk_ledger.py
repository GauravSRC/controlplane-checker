"""Session-level risk ledger — multi-turn compounding risk.

Risk accumulates across flagged turns and decays over clean ones, so a conversation that
repeatedly skirts the line is escalated even when no single turn crosses a threshold.

For agents, gate the ACTION, not just the text: run a plan-level check before tool
execution so a benign-looking message that triggers an irreversible tool call is caught
at the SIDE_EFFECT / IRREVERSIBLE blast radius.

Planned surface (to implement):
  - ``SessionRisk`` state (per-label accumulators, decay factor, turn history),
  - ``update(session_id, turn_result)`` and ``current(session_id) -> RiskVector``,
  - ``gate_action(session_id, planned_action) -> Decision``.

Scaffold only — no logic yet.
"""

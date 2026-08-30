"""SQLite evidence + feedback store (zero setup, ships in repo).

Durable record behind the telemetry: verdicts, per-detector scores, evidence packets,
policy hits, latency, and human overrides. Also backs:
  - the operator console (``console/app.py``),
  - the feedback loop / recalibration (``feedback.calibration``),
  - CVCO (Cost per Verified Correct Outcome) reporting.

DB path from settings (``LEDGER_DB_PATH``, default ``data/controlplane.db``, gitignored).

Planned surface (to implement):
  - schema init / migrations,
  - ``record_verdict(...)``, ``record_feedback(...)``,
  - query helpers for the console and optimizer.

Scaffold only — no logic yet.
"""

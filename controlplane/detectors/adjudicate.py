"""Layer 2 — ADJUDICATE (~1 s, small residual share).

Final tier for the few responses that survive Layers 0-1 still contested. Assembles an
evidence packet (claims, retrieved evidence, per-detector scores, policy hits, session
context) and resolves via LLM-as-judge or human handoff.

Human decisions here are the ground-truth signal for the feedback loop
(``feedback.calibration``) and count against the flag budget.

TODO: implement the LLM-judge call and the human-handoff queue
interface. Scaffold only — no logic yet.
"""

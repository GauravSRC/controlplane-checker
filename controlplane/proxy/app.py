"""FastAPI application exposing an OpenAI-compatible ``/v1/chat/completions`` endpoint.

Headline deployability story: any existing app changes only its ``base_url`` and is
instantly governed with zero code changes.

Request flow (to implement):
  1. Receive OpenAI-shaped request; extract messages, retrieved evidence, and the
     declared blast radius (via header/metadata, default INFORMATIONAL).
  2. Call the governed model through LiteLLM (streaming).
  3. Run the detector mesh (``detectors``) in parallel with per-detector timeouts,
     concurrent with token streaming for Layer 0.
  4. Route on the multi-label risk vector x blast radius (``router.decision``).
  5. Apply the verdict (PASS / ANNOTATE / HOLD / REPAIR / BLOCK), honoring the
     rollback window for irreversible side effects.
  6. Emit an evidence span (``ledger.otel``) and persist (``ledger.store``).

TODO: define ``app = FastAPI(...)`` and the route handlers.
Scaffold only — no logic yet.
"""

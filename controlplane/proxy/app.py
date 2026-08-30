"""FastAPI app exposing an OpenAI-compatible /v1/chat/completions endpoint.

Headline deployability story: an existing app changes only its base_url and is instantly
governed, with zero code changes. The response is a valid OpenAI ChatCompletion object
with one additive "controlplane" block - clients that ignore it keep working.

Flow: call the governed model via LiteLLM -> run Layer 0 REFLEX -> route on the
multi-label vector x blast radius -> apply the verdict.

Governance headers (all optional):
  x-controlplane-usecase       profile name (default / support-chat / agent-tools / payments)
  x-controlplane-policy-pack   policy pack id (eu-ai-act / india-dpdp / us-healthcare)
  x-controlplane-blast-radius  explicit override of the profile blast radius
  x-controlplane-retrieved-ids comma-separated retrieved doc IDs (grounding set)
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import litellm
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from controlplane.config import get_settings
from controlplane.detectors.base import CapabilityTier, DetectionContext
from controlplane.detectors.reflex import run_reflex
from controlplane.detectors.tier1 import run_tier1
from controlplane.policy.engine import get_engine
from controlplane.router.decision import BlastRadius, Verdict, decide

settings = get_settings()
app = FastAPI(title="ControlPlane Checker", version="0.1.0")

# Models that reject the logprobs parameter -> capability Tier C (black-box).
# The Capability Negotiation Layer (proxy/capability.py) will probe this properly;
# for now we degrade on the error and record the tier we actually got.
_TIER_C_MODELS = ("gpt-oss", "compound", "qwen")

BLOCK_MESSAGE = (
    "This response was withheld by ControlPlane Checker pending review. "
    "A human reviewer has been notified."
)


def _tier_for(model: str) -> CapabilityTier:
    return (
        CapabilityTier.C
        if any(m in model for m in _TIER_C_MODELS)
        else CapabilityTier.B
    )


def _extract_logprobs(choice: Any) -> list[float] | None:
    """Pull flat token logprobs out of an OpenAI-shaped choice, if present."""
    lp = getattr(choice, "logprobs", None)
    content = getattr(lp, "content", None) if lp else None
    if not content:
        return None
    out = [getattr(tok, "logprob", None) for tok in content]
    return [x for x in out if x is not None] or None


def _annotation(decision: Any) -> str:
    labels = ", ".join(decision.fired_labels) or "none"
    return (
        f"\n\n---\n**ControlPlane notice** - flagged for {labels} "
        f"(risk {decision.risk_score:.2f}). {decision.rationale}"
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": settings.model,
        "tier": _tier_for(settings.model).value,
        "key_configured": bool(settings.groq_api_key),
    }


@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{"id": settings.model, "object": "model", "owned_by": "controlplane"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_controlplane_usecase: str | None = Header(default=None),
    x_controlplane_policy_pack: str | None = Header(default=None),
    x_controlplane_blast_radius: str | None = Header(default=None),
    x_controlplane_retrieved_ids: str | None = Header(default=None),
) -> JSONResponse:
    t_start = time.perf_counter()
    body = await request.json()

    messages = body.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    # Policy resolution: (pack, use_case) -> effective config. Hot-reloads on mtime.
    engine = get_engine()
    usecase = x_controlplane_usecase or "internal_copilot"
    config = engine.resolve(x_controlplane_policy_pack, usecase)
    blast_radius = BlastRadius.parse(
        x_controlplane_blast_radius, default=config.profile.blast_radius
    )
    retrieved_ids = [
        s.strip() for s in (x_controlplane_retrieved_ids or "").split(",") if s.strip()
    ]

    model = body.get("model") or settings.model
    if "/" not in model:
        # Client sent a bare name (e.g. "gpt-4o"); govern our configured model instead.
        model = settings.model

    # Reasoning models spend tokens before emitting content - keep a floor.
    max_tokens = max(int(body.get("max_tokens") or 0), settings.max_tokens_floor)
    tier = _tier_for(model)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": body.get("temperature", 0.7),
        "api_key": settings.groq_api_key,
        "timeout": settings.request_timeout_s,
    }
    if tier is CapabilityTier.B:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 1

    t_model = time.perf_counter()
    try:
        completion = await litellm.acompletion(**kwargs)
    except Exception as exc:  # provider errors surface as OpenAI-shaped errors
        if "logprobs" in str(exc) and tier is CapabilityTier.B:
            # Capability negotiation: degrade to Tier C and retry once.
            kwargs.pop("logprobs", None)
            kwargs.pop("top_logprobs", None)
            tier = CapabilityTier.C
            completion = await litellm.acompletion(**kwargs)
        else:
            raise HTTPException(status_code=502, detail=f"upstream error: {exc}") from exc
    model_ms = (time.perf_counter() - t_model) * 1000.0

    payload = completion.model_dump()
    choice = completion.choices[0]
    text = choice.message.content or ""
    usage = payload.get("usage") or {}

    # ---- Layer 0: REFLEX, on the free signals only --------------------------
    t_gov = time.perf_counter()
    ctx = DetectionContext(
        prompt="\n".join(str(m.get("content", "")) for m in messages),
        response=text,
        retrieved_ids=retrieved_ids,
        logprobs=_extract_logprobs(choice),
        tier=tier,
        expect_json=(body.get("response_format") or {}).get("type") == "json_object",
        prompt_tokens=usage.get("prompt_tokens", 0) or 0,
        completion_tokens=usage.get("completion_tokens", 0) or 0,
        usecase=usecase,
        meta={"policy_pack": x_controlplane_policy_pack or "none"},
    )
    # Layer 0 ALWAYS runs.
    reflex = await run_reflex(ctx)
    risk = reflex.risk
    layer_reached = 0
    tier1_ms = 0.0
    degraded = reflex.degraded
    notes = list(reflex.notes)

    # Cascade: escalate when composite risk >= escalate_at OR blast >= SIDE_EFFECT.
    if engine.should_escalate(risk, config, blast_radius):
        t_t1 = time.perf_counter()
        tier1 = await run_tier1(ctx, config, blast_radius)
        tier1_ms = (time.perf_counter() - t_t1) * 1000.0
        risk = risk.merge(tier1.risk)
        layer_reached = 1
        degraded = degraded or tier1.degraded
        notes.extend(tier1.notes)

    policy_outcome = engine.evaluate(risk, config, blast_radius)
    decision = decide(
        risk_vector=risk,
        blast_radius=blast_radius,
        policy_outcome=policy_outcome,
        session_risk=0.0,
    )
    gov_ms = (time.perf_counter() - t_gov) * 1000.0

    # ---- Apply the verdict ---------------------------------------------------
    if decision.verdict is Verdict.ANNOTATE:
        payload["choices"][0]["message"]["content"] = text + _annotation(decision)
    elif decision.verdict is Verdict.BLOCK:
        payload["choices"][0]["message"]["content"] = BLOCK_MESSAGE
        payload["choices"][0]["finish_reason"] = "content_filter"
    elif decision.verdict is Verdict.HOLD:
        # ABSTAIN, do not silently drop: text still streams, side effects wait.
        payload["choices"][0]["message"]["content"] = text + (
            "\n\n---\n**ControlPlane: held** - the text is shown, but any side effect "
            "is queued pending clearance."
        )

    payload["controlplane"] = {
        "action": decision.verdict.value,
        "risk_vector": decision.risk_vector,
        "fired_labels": decision.fired_labels,
        "signals": [s.to_dict() for s in risk.signals],
        "layer_reached": layer_reached,
        "escalate_to_layer": decision.escalate_to_layer,
        "blast_radius": decision.blast_radius.value,
        "risk_score": round(decision.risk_score, 4),
        "dominant_label": decision.dominant_label.value
        if decision.dominant_label
        else None,
        "verification_mode": reflex.mode.value,
        "capability_tier": tier.value,
        "degraded": degraded,
        "rollback_window": decision.rollback_window,
        "handoff": decision.handoff,
        "usecase": usecase,
        "policy_pack": config.pack_id,
        "regulation": config.regulation,
        "policy_rule": decision.policy_rule,
        "latency_budget_ms": config.latency_budget_ms,
        "fail_open": config.fail_open,
        "rationale": decision.rationale,
        "notes": notes,
        "latency_ms": {
            "added": round(gov_ms, 2),
            "reflex": round(reflex.latency_ms, 2),
            "tier1": round(tier1_ms, 2),
            "model": round(model_ms, 2),
            "total": round((time.perf_counter() - t_start) * 1000.0, 2),
        },
        "trace_id": str(uuid.uuid4()),
    }
    return JSONResponse(content=payload)

"""Layer 0 - REFLEX (~10 ms, majority of traffic, runs concurrent with token streaming).

Only signals that are ALREADY FREE - no extra model calls:
  - sequence confidence from logprobs (degrades to a no-op on Tier C),
  - citation-ID validity (cited IDs vs the retrieved set),
  - schema / JSON conformance,
  - deterministic PII + secret regex,
  - budget counters (tokens / retries / tool-loop depth) -> COST.

Every signal writes into a multi-label RiskVector. A fabricated personal detail fires
HALLUCINATION *and* PRIVACY - the overlap case is a feature, not a bug (CLAUDE.md 3.2).
"""

from __future__ import annotations

import json
import math
import re
import time

from controlplane.detectors.base import (
    DetectionContext,
    Detector,
    DetectorResult,
    RiskLabel,
    RiskVector,
    VerificationMode,
)

# --------------------------------------------------------------------------------------
# Deterministic PII / secret patterns (India-first, per the DPDP policy pack).
# Illustrative regexes over synthetic demo data - not a compliance-grade PII engine.
# Deep NER-based PII lives in detectors/pii.py (Presidio, Layer 1).
# --------------------------------------------------------------------------------------
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    # Indian mobile: optional +91, leading digit 6-9, 10 digits total.
    "phone_in": re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)"),
    # PAN: 5 letters, 4 digits, 1 letter.
    "pan": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    # Aadhaar: 12 digits, first digit 2-9, optionally space/hyphen grouped.
    "aadhaar": re.compile(r"(?<!\d)[2-9]\d{3}[-\s]?\d{4}[-\s]?\d{4}(?!\d)"),
    # Payment card: 13-19 digits, optionally grouped; Luhn-checked below.
    "card": re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
}

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "api_key_openai": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "api_key_groq": re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}

# Severity per PII kind - a leaked Aadhaar is worse than a leaked email.
PII_SEVERITY: dict[str, float] = {
    "email": 0.45,
    "phone_in": 0.55,
    "pan": 0.80,
    "aadhaar": 0.90,
    "card": 0.85,
}

CITATION_RE = re.compile(r"\[(?:doc[:\-\s]?)?([A-Za-z0-9_\-]{1,40})\]", re.IGNORECASE)

# A person named near a fabricated citation is the hallucination x privacy overlap.
PERSON_HINT_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Shri|Smt)\.?\s+[A-Z][a-z]+|\b[A-Z][a-z]+\s+[A-Z][a-z]+\b"
)

# Budget ceilings -> COST label. Overridable per use case via context.meta.
DEFAULT_TOKEN_BUDGET = 1500
DEFAULT_RETRY_BUDGET = 2
DEFAULT_TOOL_LOOP_BUDGET = 3


def _luhn_ok(digits: str) -> bool:
    """Luhn check so ordinary long numbers are not flagged as payment cards."""
    d = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(d) <= 19:
        return False
    total, parity = 0, len(d) % 2
    for i, n in enumerate(d):
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _redact(match: str) -> str:
    """Never echo a raw secret or PII value into the ledger."""
    keep = 2 if len(match) <= 8 else 4
    return f"{match[:keep]}***{match[-2:]} (len={len(match)})"


def sequence_confidence(logprobs: list[float] | None) -> float | None:
    """Mean token logprob -> [0,1] confidence. None when the provider hides logprobs.

    Returning None (rather than a fake 1.0) is what lets the router see Tier C as
    "unmeasured" instead of "confident".
    """
    if not logprobs:
        return None
    finite = [lp for lp in logprobs if lp is not None and math.isfinite(lp)]
    if not finite:
        return None
    return math.exp(sum(finite) / len(finite))


def scan_pii(text: str) -> list[tuple[str, str, float]]:
    """Return (kind, redacted_match, severity) for each PII / secret hit."""
    hits: list[tuple[str, str, float]] = []
    aadhaar_spans = [m.span() for m in PII_PATTERNS["aadhaar"].finditer(text)]
    for kind, pat in PII_PATTERNS.items():
        for m in pat.finditer(text):
            raw = m.group(0)
            if kind == "card":
                if not _luhn_ok(raw):
                    continue
                # An Aadhaar-shaped 12-digit run can also match the card regex;
                # keep the stronger, more specific label only.
                if any(s <= m.start() and m.end() <= e for s, e in aadhaar_spans):
                    continue
            hits.append((kind, _redact(raw), PII_SEVERITY[kind]))
    for kind, pat in SECRET_PATTERNS.items():
        for m in pat.finditer(text):
            hits.append((f"secret:{kind}", _redact(m.group(0)), 0.95))
    return hits


def citation_validity(
    text: str, retrieved_ids: list[str]
) -> tuple[list[str], list[str]]:
    """Split cited IDs into (valid, dangling) against the retrieved set."""
    cited = [m.group(1) for m in CITATION_RE.finditer(text)]
    known = {rid.lower() for rid in retrieved_ids}
    valid = [c for c in cited if c.lower() in known]
    dangling = [c for c in cited if c.lower() not in known]
    return valid, dangling


class ReflexDetector(Detector):
    """Layer 0. Pure Python, no network, no model - budgeted at ~10 ms."""

    name = "reflex"
    layer = 0
    timeout_ms = 10

    async def run(self, ctx: DetectionContext) -> DetectorResult:
        t0 = time.perf_counter()
        risk = RiskVector()
        notes: list[str] = []
        degraded = False
        text = ctx.response or ""

        # -- 1. Sequence confidence (logprobs) ---------------------------------
        conf = sequence_confidence(ctx.logprobs)
        if conf is None:
            degraded = True
            notes.append(
                f"no logprobs (tier {ctx.tier.value}) - sequence_confidence unmeasured; "
                "defer to Layer 1 consistency"
            )
        else:
            notes.append(f"sequence_confidence={conf:.3f}")
            if conf < 0.55:
                # Low mean-token probability => shaky generation.
                risk.fire(
                    "sequence_confidence",
                    RiskLabel.HALLUCINATION,
                    min(1.0, (0.55 - conf) / 0.55),
                    f"mean token confidence {conf:.3f} < 0.55",
                )

        # -- 2. Citation validity ----------------------------------------------
        valid, dangling = citation_validity(text, ctx.retrieved_ids)
        if ctx.retrieved_ids:
            mode = VerificationMode.GROUNDED
        elif valid or dangling:
            mode = VerificationMode.UNGROUNDED
        else:
            mode = VerificationMode.UNVERIFIABLE

        if dangling:
            # A fabricated source ID is a strong, free hallucination signal.
            risk.fire(
                "citation_validity",
                RiskLabel.HALLUCINATION,
                min(1.0, 0.55 + 0.15 * len(dangling)),
                f"cited unknown doc id(s): {', '.join(dangling[:5])}",
            )
            # Overlap case: a fabricated citation attached to a named person is
            # simultaneously a privacy problem.
            if PERSON_HINT_RE.search(text):
                risk.fire(
                    "citation_person_overlap",
                    RiskLabel.PRIVACY,
                    0.5,
                    "unverified claim attached to a named individual",
                )

        # -- 3. Schema / JSON conformance --------------------------------------
        if ctx.expect_json:
            try:
                json.loads(text)
            except (ValueError, TypeError) as exc:
                risk.fire(
                    "schema_conformance",
                    RiskLabel.SAFETY,
                    0.6,
                    f"expected JSON, parse failed: {exc}",
                )

        # -- 4. PII / secrets (deterministic pre-pass) -------------------------
        for kind, redacted, severity in scan_pii(text):
            risk.fire("pii_regex", RiskLabel.PRIVACY, severity, f"{kind}: {redacted}")
            if kind.startswith("secret:"):
                # Leaked credentials are a safety incident too, not just privacy.
                risk.fire(
                    "secret_leak", RiskLabel.SAFETY, 0.8, f"{kind} present in output"
                )
            elif mode is not VerificationMode.GROUNDED and kind in {
                "pan",
                "aadhaar",
                "card",
            }:
                # A sensitive identifier with no retrieved source to support it may
                # be fabricated about a real person -> hallucination + privacy co-fire.
                risk.fire(
                    "unsourced_pii",
                    RiskLabel.HALLUCINATION,
                    0.5,
                    f"{kind} emitted with no retrieved source to support it",
                )

        # -- 5. Budget counters -> COST ----------------------------------------
        token_budget = int(ctx.meta.get("token_budget", DEFAULT_TOKEN_BUDGET))
        retry_budget = int(ctx.meta.get("retry_budget", DEFAULT_RETRY_BUDGET))
        loop_budget = int(ctx.meta.get("tool_loop_budget", DEFAULT_TOOL_LOOP_BUDGET))
        total_tokens = ctx.prompt_tokens + ctx.completion_tokens

        if token_budget > 0 and total_tokens > token_budget:
            over = (total_tokens - token_budget) / token_budget
            risk.fire(
                "token_budget",
                RiskLabel.COST,
                min(1.0, 0.4 + over),
                f"{total_tokens} tokens > budget {token_budget}",
            )
        if ctx.retries > retry_budget:
            risk.fire(
                "retry_budget",
                RiskLabel.COST,
                min(1.0, 0.5 + 0.2 * (ctx.retries - retry_budget)),
                f"{ctx.retries} retries > budget {retry_budget}",
            )
        if ctx.tool_loop_depth > loop_budget:
            risk.fire(
                "tool_loop_depth",
                RiskLabel.COST,
                min(1.0, 0.6 + 0.2 * (ctx.tool_loop_depth - loop_budget)),
                f"tool loop depth {ctx.tool_loop_depth} > budget {loop_budget}",
            )

        return DetectorResult(
            detector=self.name,
            layer=self.layer,
            risk=risk,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            mode=mode,
            degraded=degraded,
            notes=notes,
        )


async def run_reflex(ctx: DetectionContext) -> DetectorResult:
    """Convenience entry point used by the proxy."""
    return await ReflexDetector().run(ctx)

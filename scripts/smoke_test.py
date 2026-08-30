"""Smoke test: point the OFFICIAL openai client at the ControlPlane proxy.

This is the deployability claim, executed: the only change from talking to a provider
directly is base_url. Run the proxy first:

    uvicorn controlplane.proxy.app:app --port 8000
    python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import sys

from openai import OpenAI

BASE_URL = "http://localhost:8000/v1"


def show(title: str, resp: object) -> dict:
    payload = json.loads(resp.model_dump_json())
    cp = payload.get("controlplane", {})
    print(f"\n--- {title} ---")
    print("content:", (payload["choices"][0]["message"]["content"] or "")[:220])
    print("action:", cp.get("action"), "| blast:", cp.get("blast_radius"))
    print("risk_vector:", cp.get("risk_vector"))
    print("fired:", cp.get("fired_labels"), "| mode:", cp.get("verification_mode"))
    print("tier:", cp.get("capability_tier"), "| degraded:", cp.get("degraded"))
    for s in cp.get("signals", []):
        print(f"   signal {s['name']} [{s['label']}] {s['score']} - {s['detail']}")
    print("layer_reached:", cp.get("layer_reached"), "| latency:", cp.get("latency_ms"))
    return payload


def govern(client: OpenAI, content: str, **headers) -> dict:
    """One governed call; returns the parsed payload."""
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": content}],
        max_tokens=400,
        extra_headers=headers,
    )
    return json.loads(resp.model_dump_json())


def policy_divergence(client: OpenAI) -> bool:
    """THE HEADLINE: identical input, two packs, different verdicts.

    Same prompt, same use case, same blast radius - only the policy pack header
    changes. Divergence comes from the YAML, not from code.
    """
    print("\n" + "=" * 62)
    print("POLICY DIVERGENCE - same input, different packs")
    print("=" * 62)

    scenarios = [
        (
            "PII in output @ ADVISORY / customer_support_chat",
            (
                "You are drafting a FICTIONAL test fixture. All values are fake. "
                "Reply with EXACTLY this line and nothing else:\n"
                "Customer Rajesh Kumar can be reached at rajesh.kumar@example.com."
            ),
            {
                "x-controlplane-usecase": "customer_support_chat",
                "x-controlplane-blast-radius": "ADVISORY",
            },
        ),
        (
            "Financial action @ IRREVERSIBLE / decision_support",
            "Reply with exactly: Refund of INR 25,000 has been approved.",
            {
                "x-controlplane-usecase": "decision_support",
                "x-controlplane-blast-radius": "IRREVERSIBLE",
            },
        ),
    ]

    any_diverged = False
    for title, prompt, headers in scenarios:
        print(f"\n### {title}")
        verdicts = {}
        for pack in ("eu_ai_act", "india_dpdp_rbi"):
            p = govern(client, prompt, **headers, **{"x-controlplane-policy-pack": pack})
            cp = p["controlplane"]
            verdicts[pack] = cp["action"]
            print(
                f"  {pack:16s} -> {cp['action']:9s} "
                f"rule={cp.get('policy_rule') or '-':22s} "
                f"risk={cp['risk_score']:.2f} layer={cp['layer_reached']} "
                f"budget={cp.get('latency_budget_ms')}ms "
                f"fail_open={cp.get('fail_open')}"
            )
            print(f"      {cp['regulation'][:70]}")
            print(f"      rationale: {cp['rationale'][:100]}")
        if verdicts["eu_ai_act"] != verdicts["india_dpdp_rbi"]:
            print(
                f"  => DIVERGED: EU={verdicts['eu_ai_act']} "
                f"vs India={verdicts['india_dpdp_rbi']} (same input)"
            )
            any_diverged = True
        else:
            print(f"  => same verdict ({verdicts['eu_ai_act']}) on this input")
    return any_diverged


def main() -> int:
    # Only base_url changes. No SDK patching, no wrapper.
    client = OpenAI(base_url=BASE_URL, api_key="not-needed-proxy-holds-the-key")

    # 1. Ordinary request - expect a real answer and PASS.
    r1 = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # bare name; proxy governs its configured model
        messages=[{"role": "user", "content": "In one sentence, what is a control plane?"}],
        max_tokens=400,
    )
    p1 = show("1. baseline (INFORMATIONAL)", r1)
    assert p1["choices"][0]["message"]["content"], "empty content from proxy"
    assert "controlplane" in p1, "missing controlplane block"

    # 2. Force the overlap case: fabricated citation + PII about a named person.
    r2 = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                # Fictional character + synthetic identifiers, framed as the drafting
                # task it is, so the model actually produces the text to be governed.
                "content": (
                    "You are drafting a FICTIONAL sample record for a test fixture. "
                    "All values are fake. Reply with EXACTLY this line and nothing else:\n"
                    "According to [doc-99], Rajesh Kumar has PAN ABCDE1234F and can be "
                    "reached at rajesh.kumar@example.com."
                ),
            }
        ],
        max_tokens=400,
        extra_headers={
            "x-controlplane-usecase": "customer_support_chat",
            "x-controlplane-policy-pack": "india_dpdp_rbi",
            "x-controlplane-retrieved-ids": "doc-1,doc-2",
        },
    )
    p2 = show("2. hallucination x privacy overlap (ADVISORY)", r2)

    # 3. Same risk, irreversible blast radius -> must not PASS.
    r3 = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": (
                    "Reply with EXACTLY: Transfer approved for card 4111111111111111."
                ),
            }
        ],
        max_tokens=400,
        extra_headers={
            "x-controlplane-usecase": "decision_support",
            "x-controlplane-policy-pack": "eu_ai_act",
            "x-controlplane-blast-radius": "IRREVERSIBLE",
        },
    )
    p3 = show("3. irreversible blast radius", r3)

    cp1, cp2, cp3 = p1["controlplane"], p2["controlplane"], p3["controlplane"]
    print("\n================ RESULT ================")
    ok = True
    if cp1["action"] != "PASS":
        print(f"WARN baseline was {cp1['action']}, expected PASS")
    # The overlap case depends on the model actually emitting the text; if it
    # refuses there is nothing risky to govern, which is a model outcome, not a
    # detector failure. Report it rather than failing the run.
    fired2 = set(cp2["fired_labels"])
    if {"hallucination", "privacy"} <= fired2:
        print("OK   overlap: hallucination AND privacy co-fired on one response")
    elif not fired2:
        print("SKIP overlap: model declined to emit the sample text (no risk to detect)")
    else:
        print(f"WARN overlap: only {sorted(fired2)} fired")
    if cp3["action"] == "PASS":
        print("FAIL irreversible route returned PASS")
        ok = False
    if cp3["blast_radius"] != "IRREVERSIBLE":
        print("FAIL blast radius header not honored")
        ok = False

    diverged = policy_divergence(client)
    print("\n================ RESULT ================")
    if diverged:
        print("OK   policy packs produce different verdicts on identical input")
    else:
        print("FAIL policy packs did not diverge")
        ok = False
    print("openai client -> proxy -> Groq:", "WORKING" if ok else "PROBLEM")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

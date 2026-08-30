"""Capability Negotiation Layer — probe what each provider exposes and degrade gracefully.

Enterprises consume foundation models via API, so we cannot assume hidden states or even
logprobs. This module classifies a provider/model into a capability tier and selects the
detection strategy accordingly:

  - Tier A (self-hosted open-weight): hidden states + logprobs available.
  - Tier B (API with logprobs, e.g. OpenAI): token entropy + sampled consistency.
  - Tier C (API, text-only, e.g. Anthropic/Groq): black-box only — SelfCheckGPT-style
    NLI consistency across resampled responses, plus retrieval verification.

Detection quality degrades gracefully with provider transparency instead of assuming
access we won't have.

TODO: ``CapabilityTier`` enum + ``probe(model) -> Capabilities``.
Scaffold only — no logic yet.
"""

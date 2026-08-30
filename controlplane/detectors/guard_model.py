"""Guard-model detector — safety and prompt-injection/jailbreak (SAFETY label).

Wraps the Groq-hosted guard models:
  - ``meta-llama/llama-guard-4-12b`` for content safety on input and output,
  - Prompt Guard for injection / jailbreak detection on input.

Maps hits to OWASP LLM Top 10 IDs (e.g. LLM01 prompt injection) for the evidence ledger.

TODO: implement as ``Detector`` subclass(es) calling the guard
models via LiteLLM. Scaffold only — no logic yet.
"""

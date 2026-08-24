"""OpenTelemetry emission — GenAI-semconv spans for every verdict.

Each verdict emits an OTel span using ``gen_ai.*`` attributes, including
``gen_ai.evaluation.score.value`` and ``gen_ai.evaluation.score.label``, exported over
OTLP to a local Arize Phoenix instance. Spans are annotated with NIST AI RMF
subcategories and OWASP LLM Top 10 IDs so oversight becomes the audit artifact, not a tax.

Planned surface (to implement):
  - tracer/provider setup + OTLP HTTP exporter to the Phoenix endpoint,
  - ``emit_verdict(context, risk_vector, decision, mappings)`` span writer.

Scaffold only — no logic yet.
"""

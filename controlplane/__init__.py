"""ControlPlane Checker — a risk-adaptive verification mesh for governing LLM output.

Sits between any model and the application via an OpenAI-compatible proxy and spends
verification compute proportional to consequence (confidence x blast radius). See
the project README for the full architecture.
"""

__version__ = "0.1.0"

"""Central configuration for ControlPlane Checker.

Loads from environment / .env. Provider model IDs live here so swapping the governed
model is a one-line change (the Groq catalog moves fast - see ASSUMPTIONS.md).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency on find_dotenv stack walking)."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


@dataclass
class Settings:
    groq_api_key: str = ""
    model: str = "groq/openai/gpt-oss-20b"
    host: str = "127.0.0.1"
    port: int = 8000
    max_tokens_floor: int = 512  # reasoning models burn tokens before emitting content
    request_timeout_s: float = 60.0
    # Per-use-case defaults; policy packs override these at load time.
    profiles: dict[str, dict] = field(
        default_factory=lambda: {
            "default": {"blast_radius": "INFORMATIONAL", "latency_budget_ms": 250},
            "support-chat": {"blast_radius": "ADVISORY", "latency_budget_ms": 300},
            "agent-tools": {"blast_radius": "SIDE_EFFECT", "latency_budget_ms": 500},
            "payments": {"blast_radius": "IRREVERSIBLE", "latency_budget_ms": 1000},
        }
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv()
    return Settings(
        groq_api_key=os.environ.get("GROQ_API_KEY", ""),
        model=os.environ.get("CONTROLPLANE_MODEL", "groq/openai/gpt-oss-20b"),
        host=os.environ.get("CONTROLPLANE_HOST", "127.0.0.1"),
        port=int(os.environ.get("CONTROLPLANE_PORT", "8000")),
    )

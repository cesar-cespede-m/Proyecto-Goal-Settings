"""Agentes de negocio con propuestas y ejecución supervisada persistente."""

from core.ai_engine.agents.context import build_context
from core.ai_engine.agents.runner import AgentRunner

__all__ = ["AgentRunner", "build_context"]

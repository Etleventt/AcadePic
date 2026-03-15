"""
OpenAI-compatible provider wrapper.
"""

from .evolink import ClientError, EvolinkProvider, OpenAICompatibleProvider

__all__ = ["ClientError", "OpenAICompatibleProvider", "EvolinkProvider"]

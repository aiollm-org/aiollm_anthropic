"""Anthropic provider: its client, API and error resolver."""

from __future__ import annotations

from aiollm_anthropic.__meta__ import __version__
from aiollm_anthropic.api import (
    ANTHROPIC_API,
    API_VERSION,
    AnthropicAPI,
)
from aiollm_anthropic.client import (
    AnthropicClient,
)
from aiollm_anthropic.resolver import (
    AnthropicErrorResolver,
)

__all__ = [
    "ANTHROPIC_API",
    "API_VERSION",
    "AnthropicAPI",
    "AnthropicClient",
    "AnthropicErrorResolver",
    "__version__",
]

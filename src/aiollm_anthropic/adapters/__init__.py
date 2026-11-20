"""Anthropic's adapters between shared types and its methods."""

from __future__ import annotations

from aiollm_anthropic.adapters.conversions import (
    message_text,
    message_thinking,
    message_tool_calls,
)
from aiollm_anthropic.adapters.files import (
    AnthropicFilesAdapter,
)
from aiollm_anthropic.adapters.generation import (
    AnthropicGenerationAdapter,
)

__all__ = [
    "AnthropicFilesAdapter",
    "AnthropicGenerationAdapter",
    "message_text",
    "message_thinking",
    "message_tool_calls",
]

"""Conversions of Anthropic's generation adapter can be swapped.

The adapter takes its input, usage and stop conversions as keyword
arguments; a swapped one is used for the request, the response and the
stream alike, and gets the provider's own types.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from aiollm.adapters import BaseConversion, IConversion, Stop
from aiollm.enums import IncompleteReason
from aiollm.providers.anthropic.adapters import AnthropicGenerationAdapter
from aiollm.providers.anthropic.adapters.conversions import (
    INPUT_CONVERSION as INPUT,
)
from aiollm.providers.anthropic.adapters.conversions import (
    STOP_CONVERSION,
    USAGE_CONVERSION,
)
from aiollm.providers.anthropic.types import Delta, Message, Usage
from aiollm.types.requests import InputItem
from aiollm.types.responses import CompletedEvent, TokenUsage
from tests.providers.anthropic import test_generation as anthropic

USAGE = TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3)

STOP = Stop(IncompleteReason.OTHER, "swapped", "because")


class Recording:
    """Swapped conversions that record what they were given."""

    def __init__(self) -> None:
        """Start with nothing seen."""
        self.seen: list[object] = []

    def usage(self, source: object, /) -> TokenUsage:
        """Record the source and give fixed counts."""
        self.seen.append(source)
        return USAGE

    def stop(self, source: object, /) -> Stop:
        """Record the source and give a fixed stop."""
        self.seen.append(source)
        return STOP


class AnthropicInput(
    BaseConversion[str | Sequence[InputItem], list[dict[str, Any]]]
):
    """Sends a fixed turn instead of the request's input."""

    def __call__(
        self, source: str | Sequence[InputItem], /
    ) -> list[dict[str, Any]]:
        """Ignore the input."""
        return [{"role": "user", "content": "swapped"}]


def test_default_conversions_are_conversions() -> None:
    defaults: list[IConversion[Any, Any]] = [
        INPUT,
        USAGE_CONVERSION,
        STOP_CONVERSION,
    ]
    adapter = AnthropicGenerationAdapter()

    assert all(isinstance(each, BaseConversion) for each in defaults)
    assert adapter.input_conversion is INPUT
    assert adapter.usage_conversion is USAGE_CONVERSION
    assert adapter.stop_conversion is STOP_CONVERSION


def check_completed(events: list[Any]) -> None:
    """Check that the completion went through the swapped conversions."""
    completed = events[-1]
    assert isinstance(completed, CompletedEvent)
    assert completed.usage == USAGE
    assert completed.incomplete_reason is STOP.incomplete_reason
    assert completed.raw_incomplete_reason == STOP.raw_incomplete_reason
    assert completed.refusal == STOP.refusal


async def test_swapped_conversions_are_used() -> None:
    recording = Recording()
    adapter = AnthropicGenerationAdapter(
        input_conversion=AnthropicInput(),
        usage_conversion=recording.usage,
        stop_conversion=recording.stop,
    )

    method = adapter.to_method(anthropic.REQUEST)
    response = adapter.to_response(anthropic.message(), anthropic.REQUEST)
    events = [
        event
        async for event in adapter.to_events(
            anthropic.events(anthropic.STREAM), anthropic.REQUEST
        )
    ]

    assert method.get_body_map()["messages"] == [
        {"role": "user", "content": "swapped"}
    ]
    assert response.usage == USAGE
    assert response.refusal == STOP.refusal
    check_completed(events)
    # A stream gives the same types as a message: a real Usage, and the
    # Delta collected from the message_delta events.
    assert [type(source) for source in recording.seen] == [
        Message,
        Usage,
        Delta,
        Usage,
    ]

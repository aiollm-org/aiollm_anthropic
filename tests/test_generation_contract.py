"""Shared generation behaves for Anthropic as for every provider.

The scenarios of ``aiollm.testing.contract.generation`` run against
``AnthropicClient.generation`` on the local server, with the payloads of
Anthropic's reference examples. Only the wire format is the provider's
own; the shared results and errors must be the same.
"""

from __future__ import annotations

import json
from typing import Any

from aiollm.providers.anthropic import AnthropicClient
from aiollm.providers.anthropic.api import PROVIDER as ANTHROPIC_PROVIDER
from aiollm.providers.anthropic.client import (
    session_factory as anthropic_factory,
)
from aiollm.providers.anthropic.methods import StreamMessage
from aiollm.testing.contract.generation import (
    CALLS,
    SCHEMA,
    GenerationCase,
    GenerationContract,
)
from aiollm.types.responses import TokenUsage
from tests.providers.anthropic import test_messages as anthropic


def anthropic_called() -> dict[str, Any]:
    """An Anthropic message with text and ``CALLS``."""
    blocks = [
        {"type": "tool_use", "id": call_id, "name": name, "input": arguments}
        for call_id, name, arguments in CALLS
    ]
    return anthropic.MESSAGE | {
        "content": [{"type": "text", "text": "Checking."}, *blocks],
        "stop_reason": "tool_use",
        "stop_details": None,
    }


def anthropic_called_stream() -> list[dict[str, Any]]:
    """An Anthropic stream of ``CALLS``: the first call's arguments in
    two pieces, the second's in none, as Anthropic sends ``{}``."""
    start, *_, stop = anthropic.STREAM
    events: list[dict[str, Any]] = [start]
    for index, (call_id, name, arguments) in enumerate(CALLS):
        events.append(
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": {},
                },
            }
        )
        if arguments:
            text = json.dumps(arguments)
            events += [
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": part,
                    },
                }
                for part in (text[:3], text[3:])
            ]
        events.append({"type": "content_block_stop", "index": index})
    events += [
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 15},
        },
        stop,
    ]
    return events


ANTHROPIC = GenerationCase(
    provider=ANTHROPIC_PROVIDER,
    client=AnthropicClient,
    factory=anthropic_factory,
    path="/v1/messages",
    response=anthropic.MESSAGE,
    text="Hi! My name is Claude.",
    usage=TokenUsage(
        input_tokens=6197,
        output_tokens=503,
        total_tokens=6700,
        cache_read_tokens=2051,
        cache_write_tokens=2051,
        reasoning_tokens=0,
    ),
    truncated=anthropic.MESSAGE | {"stop_reason": "max_tokens"},
    refused=anthropic.MESSAGE | {"stop_reason": "refusal"},
    refusal=anthropic.MESSAGE["stop_details"]["explanation"],
    stream=anthropic.STREAM,
    deltas=["Hello", "!"],
    stream_usage=TokenUsage(
        input_tokens=25, output_tokens=15, total_tokens=40
    ),
    overloaded={
        "type": "error",
        "error": {"type": "overloaded_error", "message": "Overloaded"},
    },
    quota={
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": "Spend cap reached",
            "details": {"error_code": "enforced_spend_limit_reached"},
        },
    },
    bad_request={
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "Bad"},
    },
    raw_stream=lambda: StreamMessage(**anthropic.request()),
    options={"stop_sequences": ["END"]},
    other_options={"stop_sequences": ["STOP", "HALT"]},
    bad_option={"stop_sequences": "END"},
    effort={"output_config": {"effort": "high"}},
    output_schema={
        "output_config": {
            "effort": "high",
            "format": {
                "type": "json_schema",
                "schema": {"type": "object", "required": ["n"]},
            },
        }
    },
    tools={
        "tools": [
            {
                "name": "get_weather",
                "description": "Current weather",
                "input_schema": SCHEMA,
                "strict": True,
            },
            {
                "name": "get_time",
                "input_schema": {"type": "object", "properties": {}},
            },
        ],
        "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
    },
    forced={"tool_choice": {"type": "tool", "name": "get_time"}},
    only_parallel={
        "tool_choice": {"type": "auto", "disable_parallel_tool_use": False}
    },
    called=anthropic_called(),
    called_stream=anthropic_called_stream(),
    cut_off={"stop_reason": "max_tokens"},
    input_field="messages",
    continued_streamed=[
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": arguments,
                }
                for call_id, name, arguments in CALLS
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": f"result of {name}",
                }
                for call_id, name, _ in CALLS
            ],
        },
    ],
    continued=[
        {"role": "assistant", "content": anthropic_called()["content"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": f"result of {name}",
                }
                for call_id, name, _ in CALLS
            ],
        },
    ],
    continued_shared=[
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking."},
                *(
                    {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": arguments,
                    }
                    for call_id, name, arguments in CALLS
                ),
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": f"result of {name}",
                }
                for call_id, name, _ in CALLS
            ],
        },
    ],
    parts={
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in these?"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "iVBORw0K",
                },
            },
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/cat.jpg",
                },
            },
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi0xLjc=",
                },
                "title": "report.pdf",
            },
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "JVBERi0xLjc=",
                },
            },
            {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": "https://example.com/report.pdf",
                },
            },
        ],
    },
    after_results=[
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "noon",
                },
                {"type": "text", "text": "What is in these?"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "iVBORw0K",
                    },
                },
            ],
        }
    ],
)


class TestAnthropicGeneration(GenerationContract):
    case = ANTHROPIC

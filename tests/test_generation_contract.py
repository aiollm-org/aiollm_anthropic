"""Shared generation behaves for Anthropic as for every provider.

The scenarios of ``aiollm.testing.contract.generation`` run against
``AnthropicClient.generation`` on the local server, with the payloads of
Anthropic's reference examples. Only the wire format is the provider's
own; the shared results and errors must be the same.
"""

from __future__ import annotations

import json
from typing import Any

from aiollm.testing.contract.generation import (
    CALLS,
    SCHEMA,
    SEARCH_ID,
    SEARCH_QUERIES,
    BuiltinCase,
    GenerationCase,
    GenerationContract,
)
from aiollm.types.responses import TokenUsage

from aiollm_anthropic import AnthropicClient
from aiollm_anthropic.api import PROVIDER as ANTHROPIC_PROVIDER
from aiollm_anthropic.client import (
    session_factory as anthropic_factory,
)
from aiollm_anthropic.methods import StreamMessage
from tests import test_messages as anthropic


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


SEARCH_CALL: dict[str, Any] = {
    "type": "server_tool_use",
    "id": SEARCH_ID,
    "name": "web_search",
    "input": {"query": SEARCH_QUERIES[0]},
}
"""The block of the search the model makes."""

SEARCH_RESULT: dict[str, Any] = {
    "type": "web_search_tool_result",
    "tool_use_id": SEARCH_ID,
    "content": [
        {
            "type": "web_search_result",
            "url": "https://example.com/paris",
            "title": "Paris weather",
            "encrypted_content": "EqgfCioIARgBIiQ",
            "page_age": "April 30, 2025",
        }
    ],
}
"""The block with the pages it found."""

SEARCH_FAILED: dict[str, Any] = {
    "type": "web_search_tool_result",
    "tool_use_id": SEARCH_ID,
    "content": {
        "type": "web_search_tool_result_error",
        "error_code": "unavailable",
    },
}
"""The block of a search that failed."""

CITING_TEXT: dict[str, Any] = {
    "type": "text",
    "text": "Hi! My name is Claude.",
    "citations": [
        {
            "type": "web_search_result_location",
            "url": "https://example.com/paris",
            "title": "Paris weather",
            "encrypted_index": "Eo8BCioIAhgB",
            "cited_text": "It is sunny in Paris.",
        }
    ],
}
"""The text with its citation."""


def anthropic_message(
    content: list[dict[str, Any]], stop_reason: str = "end_turn"
) -> dict[str, Any]:
    """An Anthropic message of the given content blocks."""
    return anthropic.MESSAGE | {
        "content": content,
        "stop_reason": stop_reason,
        "stop_details": None,
    }


def anthropic_searched_stream() -> list[dict[str, Any]]:
    """An Anthropic stream of one web search and the text it leads to:
    the call's input in two pieces, the result whole, the citation in
    a ``citations_delta``."""
    start, *_, stop = anthropic.STREAM
    text = json.dumps(SEARCH_CALL["input"])
    return [
        start,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": SEARCH_CALL | {"input": {}},
        },
        *(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": part},
            }
            for part in (text[:5], text[5:])
        ),
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": SEARCH_RESULT,
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "text", "text": "", "citations": None},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "text_delta", "text": "Hi! My name "},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {
                "type": "citations_delta",
                "citation": CITING_TEXT["citations"][0],
            },
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "text_delta", "text": "is Claude."},
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 15},
        },
        stop,
    ]


ANTHROPIC_BUILTIN = BuiltinCase(
    tools={
        "tools": [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "allowed_domains": ["example.com"],
                "user_location": {
                    "type": "approximate",
                    "country": "FR",
                    "city": "Paris",
                },
            },
            {"type": "code_execution_20250825", "name": "code_execution"},
        ]
    },
    mixed={
        "tools": [
            {
                "name": "get_time",
                "input_schema": {"type": "object", "properties": {}},
            },
            {"type": "web_search_20250305", "name": "web_search"},
        ]
    },
    searched=anthropic_message([SEARCH_CALL, SEARCH_RESULT, CITING_TEXT]),
    searched_stream=anthropic_searched_stream(),
    failed=anthropic_message(
        [SEARCH_CALL, SEARCH_FAILED, {"type": "text", "text": "No luck."}]
    ),
    paused=anthropic_message([SEARCH_CALL], "pause_turn"),
    continued=[{"role": "assistant", "content": [SEARCH_CALL]}],
)

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
    builtin=ANTHROPIC_BUILTIN,
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

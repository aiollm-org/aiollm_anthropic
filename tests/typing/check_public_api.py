# pyright: reportUnnecessaryTypeIgnoreComment=true
"""Static checks of Anthropic's public API.

Never run: mypy (through ``pytest --mypy``) and pyright check it. Every
``# type: ignore[...]`` marks a line that must stay a type error; mypy's
``warn_unused_ignores`` and pyright's ``reportUnnecessaryTypeIgnoreComment``
fail the check if that error disappears.
"""

from __future__ import annotations

from typing import assert_type

from aiollm import GenerationStream, IGenerationClient, Provider, Stream
from aiollm.adapters import IConversion
from aiollm.adapters import Stop as GenerationStop
from aiollm.client import BaseClient
from aiollm.providers.anthropic import AnthropicClient
from aiollm.providers.anthropic.adapters import AnthropicGenerationAdapter
from aiollm.providers.anthropic.methods import CreateMessage, StreamMessage
from aiollm.providers.anthropic.methods.params import UserMessageParam
from aiollm.providers.anthropic.types import (
    Delta,
    FileMetadata,
    Message,
    MessageStreamEvent,
    Usage,
)
from aiollm.providers.anthropic.types.options import (
    AnthropicGenerationOptions,
)
from aiollm.resources import GenerationResource, IFilesClient
from aiollm.session.aiohttp import AiohttpSession
from aiollm.types.requests import DocumentPart, GenerationRequest, ProviderFile
from aiollm.types.responses import GenerationResponse, StoredFile, TokenUsage


async def messages_infer_types(session: AiohttpSession) -> None:
    turn = UserMessageParam(role="user", content="x")
    create = CreateMessage(model="m", max_tokens=16, messages=[turn])
    assert_type(await session.send_request(create), Message)
    stream = StreamMessage(model="m", max_tokens=16, messages=[turn])
    async with session.stream(stream) as events:
        assert_type(events, Stream[MessageStreamEvent])
    CreateMessage(model="m", max_tokens=16, messages=[turn], temperature=1)  # type: ignore[call-arg]


async def generation_infers_its_types(client: AnthropicClient) -> None:
    request = GenerationRequest(model="m", input="x")
    response = await client.generation.create(request)
    assert_type(response, GenerationResponse[Message])
    async with client.generation.stream(request) as events:
        assert_type(events, GenerationStream[MessageStreamEvent])
    AnthropicClient("sk", default_max_tokens="1")  # type: ignore[arg-type]


def no_usage(source: Usage, /) -> TokenUsage:
    """A plain function is a conversion of Anthropic's counts."""
    return TokenUsage(input_tokens=source.input_tokens)


def text_usage(source: str, /) -> TokenUsage:
    """A conversion of something else than Anthropic's counts."""
    return TokenUsage(input_tokens=len(source))


def conversions_are_typed(
    stop: IConversion[Message | Delta, GenerationStop],
    message_only: IConversion[Message, GenerationStop],
    other: IConversion[str, GenerationStop],
) -> None:
    AnthropicGenerationAdapter(usage_conversion=no_usage, stop_conversion=stop)
    # A stream ends with the Delta of message_delta, not a Message.
    AnthropicGenerationAdapter(stop_conversion=message_only)  # type: ignore[arg-type]
    AnthropicGenerationAdapter(usage_conversion=text_usage)  # type: ignore[arg-type]
    AnthropicGenerationAdapter(stop_conversion=other)  # type: ignore[arg-type]


def clients_take_session_arguments() -> None:
    AnthropicClient("sk", default_max_tokens=8, follow_redirects=True)
    AnthropicClient("sk", trust_env="yes")  # type: ignore[arg-type]


def any_generation_client(client: IGenerationClient) -> IGenerationClient:
    return client


def any_files_client(client: IFilesClient) -> IFilesClient:
    return client


async def clients_are_shared_clients(anthropic: AnthropicClient) -> None:
    any_generation_client(anthropic)
    any_files_client(anthropic)
    any_files_client(BaseClient(anthropic.session))  # type: ignore[arg-type]

    stored = await anthropic.files.upload(b"%PDF", expires_in_seconds=3600)
    assert_type(stored, StoredFile[FileMetadata])
    assert_type(stored.ref, ProviderFile)
    async with anthropic.files.download(stored.ref) as content:
        assert_type(content, Stream[bytes])
    DocumentPart(file=stored.ref)


def clients_bind_their_resources() -> None:
    assert_type(
        AnthropicClient("sk").generation,
        GenerationResource[
            Message, MessageStreamEvent, AnthropicGenerationOptions
        ],
    )
    assert_type(AnthropicClient.provider, Provider)


async def provider_options_are_typed(request: GenerationRequest) -> None:
    anthropic = AnthropicClient("sk").generation
    await anthropic.create(
        request, provider_options={"stop_sequences": ["END"]}
    )
    await anthropic.create(request, provider_options={"store": False})  # type: ignore[call-overload]

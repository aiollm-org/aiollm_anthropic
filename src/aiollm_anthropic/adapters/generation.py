"""Adapter of shared generation to Anthropic's Messages API."""

from __future__ import annotations

import copy
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, Self

from pydantic import MISSING, BaseModel

from aiollm.adapters.base import BaseCompositeAdapter
from aiollm.adapters.validation import (
    check_builtin_tools,
    check_conversation,
    check_files,
    checked_options,
    merge_options,
    request_errors,
)
from aiollm.enums import ServerToolPhase, ToolChoice
from aiollm.exceptions import RequestValidationError
from aiollm.types.requests.builtin import (
    BuiltinTool,
    CodeExecution,
    WebSearch,
)
from aiollm.types.requests.generation import (
    ForcedTool,
    GenerationRequest,
    InputItem,
    Tool,
    ToolCallTurn,
)
from aiollm.types.responses.events import (
    CompletedEvent,
    GenerationEvent,
    OtherEvent,
    ReasoningDeltaEvent,
    ServerToolCallEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallDoneEvent,
    ToolCallStartEvent,
)
from aiollm.types.responses.generation import GenerationResponse

from aiollm_anthropic.adapters.conversions import (
    INPUT_CONVERSION,
    STOP_CONVERSION,
    USAGE_CONVERSION,
    StreamState,
    block_citations,
    message_citations,
    message_native,
    message_server_calls,
    message_text,
    message_thinking,
    message_tool_calls,
    server_tool_calls,
)
from aiollm_anthropic.api import PROVIDER
from aiollm_anthropic.methods.messages import (
    CreateMessage,
    StreamMessage,
)
from aiollm_anthropic.types import (
    CitationsDelta,
    InputJSONDelta,
    Message,
    MessageStreamEvent,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    ServerToolUseBlock,
    SignatureDelta,
    TextDelta,
    ThinkingDelta,
    ToolUseBlock,
)
from aiollm_anthropic.types.options import AnthropicGenerationOptions

if TYPE_CHECKING:
    from aiollm.adapters.protocols import IConversion
    from aiollm.adapters.stop import Stop
    from aiollm.types.responses.generation import TokenUsage

    from aiollm_anthropic.types import Delta, Usage

DEFAULT_MAX_TOKENS: Final = 16000
"""``max_tokens`` sent when the request has no ``max_output_tokens``.

Thinking is always on for the Claude 5.x models and counts toward
``max_tokens``, so the default leaves room for it; 16000 is the value
Anthropic's own examples send.
"""

SHARED_FIELDS: Final[Mapping[str, str]] = {
    "messages": "input",
    "max_tokens": "max_output_tokens",
    "system": "instructions",
    "metadata": "end_user_id",
}
"""Shared request field of each method field with another name."""


class AnthropicGenerationAdapter(
    BaseCompositeAdapter[
        GenerationRequest,
        Message,
        GenerationResponse[Message],
        MessageStreamEvent,
        GenerationEvent[MessageStreamEvent],
    ],
):
    """Generates text with ``POST /messages``.

    A text ``input`` becomes one ``user`` turn, a list of turns the
    ``messages`` as they are, and ``instructions`` the ``system``
    prompt. Anthropic requires ``max_tokens``, so a request without
    ``max_output_tokens`` gets ``default_max_tokens``. A request that
    Anthropic would reject raises ``RequestValidationError`` before
    anything is sent, such as a conversation that does not end with a
    ``user`` turn. ``end_user_id`` becomes ``metadata.user_id``.

    The conversions of the input, the token counts and the stop are
    keyword arguments, so any of them can be swapped without a subclass;
    a stream goes through the same ones as a message.

    Attributes:
        default_max_tokens: ``max_tokens`` of a request without
            ``max_output_tokens``.
        input_conversion: Turns the shared ``input`` into ``messages``.
        usage_conversion: Turns Anthropic's ``Usage`` into shared counts.
        stop_conversion: Tells how a message, or a stream, stopped.

    Example:
        >>> generation = GenerationResource(
        ...     session, AnthropicGenerationAdapter(default_max_tokens=4096)
        ... )
    """

    provider = PROVIDER

    def __init__(
        self,
        *,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        input_conversion: IConversion[
            str | Sequence[InputItem], list[dict[str, Any]]
        ] = INPUT_CONVERSION,
        usage_conversion: IConversion[Usage, TokenUsage] = USAGE_CONVERSION,
        stop_conversion: IConversion[Message | Delta, Stop] = STOP_CONVERSION,
    ) -> None:
        """Store the default token limit and the conversions.

        Args:
            default_max_tokens: ``max_tokens`` of a request without
                ``max_output_tokens``; positive.
            input_conversion: Turns the shared ``input`` into
                ``messages``.
            usage_conversion: Turns Anthropic's ``Usage`` into shared
                counts.
            stop_conversion: Tells how a message, or a stream, stopped.

        Raises:
            ValueError: If ``default_max_tokens`` is not positive.
        """
        if default_max_tokens < 1:
            message = (
                "default_max_tokens must be positive, "
                f"got {default_max_tokens}"
            )
            raise ValueError(message)
        self.default_max_tokens = default_max_tokens
        self.input_conversion = input_conversion
        self.usage_conversion = usage_conversion
        self.stop_conversion = stop_conversion
        self.provider_options: dict[str, Any] = {}

    def with_options(self, options: AnthropicGenerationOptions, /) -> Self:
        """Return a copy that sends the options with every method.

        Args:
            options: Fields of ``CreateMessage`` that the shared request
                does not hold, such as ``{"stop_sequences": ["END"]}``.

        Returns:
            A copy of the adapter; this one is left as it is.

        Raises:
            RequestValidationError: A key is not one of the options.
        """
        adapter = copy.copy(self)
        adapter.provider_options = checked_options(
            options, AnthropicGenerationOptions, self.provider
        )
        return adapter

    def to_method(self, request: GenerationRequest) -> CreateMessage:
        """Build the method that creates a message at once.

        Args:
            request: The shared request.

        Returns:
            ``CreateMessage`` for the request.

        Raises:
            RequestValidationError: Anthropic cannot take the request.
        """
        fields = self._fields(request)
        with request_errors(self.provider, SHARED_FIELDS):
            return CreateMessage(**fields)

    def to_response(
        self,
        raw: Message,
        request: GenerationRequest,
    ) -> GenerationResponse[Message]:
        """Convert a finished message.

        Args:
            raw: Anthropic's message.
            request: The request it answers.

        Returns:
            The generated text with its token counts, the calls of the
            built-in tools Anthropic ran (``server_tool_calls``) and the
            sources its text cites (``citations``); a message stopped
            early, by ``max_tokens`` for example, keeps its text and gets
            ``incomplete_reason`` (``PAUSED`` for ``pause_turn``), and a
            refusal gets ``refusal``.

        Raises:
            APIResponseError: The message has no stop reason.
        """
        stop = self.stop_conversion(raw)
        return GenerationResponse[Message](
            id=raw.id,
            provider=self.provider,
            model=raw.model,
            text=message_text(raw),
            usage=self.usage_conversion(raw.usage),
            incomplete_reason=stop.incomplete_reason,
            raw_incomplete_reason=stop.raw_incomplete_reason,
            refusal=stop.refusal,
            reasoning=message_thinking(raw),
            tool_calls=message_tool_calls(raw),
            server_tool_calls=message_server_calls(raw),
            citations=message_citations(raw),
            native=message_native(raw),
            raw=raw,
        )

    def to_stream_method(self, request: GenerationRequest) -> StreamMessage:
        """Build the method that streams a message.

        Args:
            request: The shared request.

        Returns:
            ``StreamMessage`` for the request.

        Raises:
            RequestValidationError: Anthropic cannot take the request.
        """
        fields = self._fields(request)
        with request_errors(self.provider, SHARED_FIELDS):
            return StreamMessage(**fields)

    async def to_events(
        self,
        items: AsyncIterator[MessageStreamEvent],
        request: GenerationRequest,
    ) -> AsyncGenerator[GenerationEvent[MessageStreamEvent]]:
        """Convert the events of a message stream.

        Text deltas become ``TextDeltaEvent`` and non-empty thinking
        deltas ``ReasoningDeltaEvent``; a ``tool_use`` block becomes
        ``ToolCallStartEvent``, a ``ToolCallDeltaEvent`` per
        ``input_json_delta`` and ``ToolCallDoneEvent`` at its
        ``content_block_stop``; a built-in tool's ``server_tool_use``
        block becomes a ``ServerToolCallEvent`` that starts the call and
        its result block one that ends it; ``message_stop`` becomes
        ``CompletedEvent`` with the counts and stop reason collected from
        ``message_start`` and ``message_delta``, the built-in tool calls
        and citations, and the content as ``native``; every other event
        (the pieces of a built-in tool's input, citations) is an
        ``OtherEvent``.

        Args:
            items: The events Anthropic sent.
            request: The request the stream answers.

        Yields:
            One shared event per Anthropic event.

        Raises:
            APIResponseError: ``message_stop`` came without a stop
                reason or without ``message_start``, or arguments came
                for a block no ``tool_use`` or ``server_tool_use``
                opened. The stream's end is
                checked by ``StreamMessage`` and by the use case.
        """
        state = StreamState()
        async for item in items:
            event = _block_event(state, item)
            if event is not None:
                yield event
                continue
            match item:
                case RawMessageStartEvent():
                    state.start(item)
                case RawMessageDeltaEvent():
                    state.update(item)
                case RawMessageStopEvent():
                    stop = self.stop_conversion(state.delta)
                    message = state.message()
                    native = state.native()
                    blocks = [] if native is None else native.items
                    yield CompletedEvent[MessageStreamEvent](
                        id=message.id,
                        provider=self.provider,
                        model=message.model,
                        text=state.text(),
                        usage=self.usage_conversion(state.usage()),
                        incomplete_reason=stop.incomplete_reason,
                        raw_incomplete_reason=stop.raw_incomplete_reason,
                        refusal=stop.refusal,
                        reasoning=state.thinking(),
                        tool_calls=state.tool_calls(),
                        server_tool_calls=server_tool_calls(blocks),
                        citations=block_citations(blocks),
                        native=native,
                        raw=item,
                    )
                    continue
                case _:
                    pass
            yield OtherEvent[MessageStreamEvent](raw=item)

    def _fields(self, request: GenerationRequest) -> dict[str, Any]:
        """Take the method fields of a request.

        Args:
            request: The shared request.

        Returns:
            The fields that both methods take, the provider options of
            ``with_options`` included.

        Raises:
            RequestValidationError: The conversation is out of order (see
                ``check_conversation``), a part names another provider's
                file, a built-in tool is one Anthropic does not run (see
                ``check_builtin_tools``), or a call of a turn without
                Anthropic's own content has arguments that are not a
                JSON object, which ``tool_use.input`` must be.
        """
        check_conversation(request, self.provider)
        check_files(request, self.provider)
        check_builtin_tools(request, self.provider, (WebSearch, CodeExecution))
        _check_arguments(request)
        fields = {
            "model": request.model,
            "messages": self.input_conversion(request.input),
            "max_tokens": (
                self.default_max_tokens
                if request.max_output_tokens is MISSING
                else request.max_output_tokens
            ),
            "system": request.instructions,
            "metadata": (
                MISSING
                if request.end_user_id is MISSING
                else {"user_id": request.end_user_id}
            ),
            "output_config": _output_config(request),
            "tools": _tools(request),
            "tool_choice": _tool_choice(request),
        }
        return merge_options(fields, self.provider_options, self.provider)


def _block_event(
    state: StreamState,
    item: MessageStreamEvent,
) -> GenerationEvent[MessageStreamEvent] | None:
    """Keep a content block event in the state and convert it.

    Args:
        state: The stream's state.
        item: An event of the stream.

    Returns:
        The shared event of a text, thinking or tool call event; ``None``
        for an event that is no block event or has no shared
        counterpart (a signature, a citation, an empty thinking delta, the
        input of a built-in tool, a block that is no call starting or
        stopping), which becomes ``OtherEvent``.

    Raises:
        APIResponseError: Arguments came for a block that is no
            ``tool_use`` or ``server_tool_use``.
    """
    match item:
        case RawContentBlockStartEvent(content_block=block):
            state.start_block(item.index, block)
            return _start_event(state, item, block)
        case RawContentBlockDeltaEvent(delta=TextDelta() as delta):
            state.add_piece(item.index, "text", delta.text)
            state.add_text(delta.text)
            return TextDeltaEvent[MessageStreamEvent](
                text=delta.text, raw=item
            )
        case RawContentBlockDeltaEvent(delta=ThinkingDelta() as delta):
            state.add_piece(item.index, "thinking", delta.thinking)
            if delta.thinking:
                state.add_thinking(delta.thinking)
                return ReasoningDeltaEvent[MessageStreamEvent](
                    text=delta.thinking, raw=item
                )
        case RawContentBlockDeltaEvent(delta=SignatureDelta() as delta):
            state.add_piece(item.index, "signature", delta.signature)
        case RawContentBlockDeltaEvent(delta=InputJSONDelta() as delta):
            call_id = state.add_arguments(item.index, delta.partial_json)
            if call_id is not None:
                return ToolCallDeltaEvent[MessageStreamEvent](
                    call_id=call_id, arguments=delta.partial_json, raw=item
                )
        case RawContentBlockDeltaEvent(delta=CitationsDelta() as delta):
            state.add_citation(item.index, delta.citation)
        case RawContentBlockStopEvent():
            call = state.call(item.index)
            return (
                None
                if call is None
                else ToolCallDoneEvent[MessageStreamEvent](call=call, raw=item)
            )
        case _:
            pass
    return None


def _start_event(
    state: StreamState,
    item: RawContentBlockStartEvent,
    block: BaseModel,
) -> GenerationEvent[MessageStreamEvent] | None:
    """Convert the start of a block that is a call or a call's result.

    Args:
        state: The stream's state, which has the block.
        item: The ``content_block_start`` event.
        block: The block it starts.

    Returns:
        ``ToolCallStartEvent`` for a ``tool_use`` block, the
        ``STARTED`` event of a built-in tool call for a
        ``server_tool_use`` block and its ``DONE`` event for the block
        with the call's result; ``None`` for any other block.
    """
    if isinstance(block, ToolUseBlock):
        return ToolCallStartEvent[MessageStreamEvent](
            call_id=block.id, name=block.name, raw=item
        )
    if isinstance(block, ServerToolUseBlock):
        return ServerToolCallEvent[MessageStreamEvent](
            phase=ServerToolPhase.STARTED,
            call=state.server_started(item.index),
            raw=item,
        )
    done = state.server_done(item.index)
    if done is None:
        return None
    return ServerToolCallEvent[MessageStreamEvent](
        phase=ServerToolPhase.DONE, call=done, raw=item
    )


def _check_arguments(request: GenerationRequest) -> None:
    """Check that the calls Anthropic gets as ``tool_use`` parse.

    A turn with Anthropic's own content is sent as it came; the calls of
    any other turn are sent as ``tool_use`` blocks, whose ``input`` is a
    JSON object, so their arguments must parse as one.

    Args:
        request: The shared request.

    Raises:
        RequestValidationError: A call's arguments are not a JSON object.
    """
    if isinstance(request.input, str):
        return
    for item in request.input:
        if not isinstance(item, ToolCallTurn):
            continue
        if item.native is not MISSING and item.native.provider == PROVIDER:
            continue
        for call in item.tool_calls:
            try:
                call.parse_arguments()
            except ValueError as error:
                raise RequestValidationError(
                    str(error), provider=PROVIDER, field="input"
                ) from error


TOOL_CHOICES: Final[Mapping[ToolChoice, str]] = {
    ToolChoice.AUTO: "auto",
    ToolChoice.NONE: "none",
    ToolChoice.REQUIRED: "any",
}
"""Anthropic's ``tool_choice`` ``type`` of each shared mode."""


def _tools(request: GenerationRequest) -> list[dict[str, Any]] | MISSING:
    """Build ``tools`` from the function and the built-in tools.

    Args:
        request: The shared request.

    Returns:
        The function tools, then the built-in ones, or ``MISSING``
        without any.
    """
    functions = [] if request.tools is MISSING else request.tools
    tools = [_tool(tool) for tool in functions]
    tools += [_builtin_tool(tool) for tool in request.builtin_tools]
    return tools or MISSING


WEB_SEARCH_TYPE: Final = "web_search_20250305"
"""The version of Anthropic's web search that ``WebSearch`` sends.

The first one, which has no side effects on other tools (the newer
ones filter the results with code execution); a newer version goes
through ``aiollm_anthropic.tools.web_search``.
"""

CODE_EXECUTION_TYPE: Final = "code_execution_20250825"
"""The version of Anthropic's code execution that ``CodeExecution`` sends.

A newer version goes through ``aiollm_anthropic.tools.code_execution``.
"""


def _builtin_tool(tool: BuiltinTool) -> dict[str, Any]:
    """Build Anthropic's tool from a shared built-in tool.

    Args:
        tool: The shared tool; ``check_builtin_tools`` has accepted it.

    Returns:
        ``WebSearch`` as a ``web_search_20250305`` tool with its
        domains and its location as an ``approximate`` one,
        ``CodeExecution`` as a ``code_execution_20250825`` tool, and a
        ``ProviderTool`` (the only other kind) as its own definition.
    """
    match tool:
        case WebSearch():
            search: dict[str, Any] = {
                "type": WEB_SEARCH_TYPE,
                "name": "web_search",
            }
            if tool.allowed_domains is not MISSING:
                search["allowed_domains"] = list(tool.allowed_domains)
            location = tool.user_location
            if location is not MISSING:
                search["user_location"] = {
                    "type": "approximate",
                    **{
                        name: getattr(location, name)
                        for name in ("country", "region", "city", "timezone")
                        if getattr(location, name) is not MISSING
                    },
                }
            return search
        case CodeExecution():
            return {"type": CODE_EXECUTION_TYPE, "name": "code_execution"}
        case _:
            return copy.deepcopy(dict(tool.definition))


def _tool(tool: Tool) -> dict[str, Any]:
    """Build Anthropic's custom tool from a shared tool.

    Args:
        tool: The shared tool.

    Returns:
        The tool, its ``parameters`` as ``input_schema``; ``description``
        and ``strict`` only when given (Anthropic is not strict by
        default).
    """
    custom: dict[str, Any] = {
        "name": tool.name,
        "input_schema": tool.parameters,
    }
    if tool.description is not MISSING:
        custom["description"] = tool.description
    if tool.strict is not MISSING:
        custom["strict"] = tool.strict
    return custom


def _tool_choice(request: GenerationRequest) -> dict[str, Any] | MISSING:
    """Build Anthropic's ``tool_choice`` from the choice and parallel calls.

    Anthropic has no ``parallel_tool_calls``: it is
    ``disable_parallel_tool_use`` inside ``tool_choice``, so a request
    that gives only it gets ``{"type": "auto", ...}``, the default mode.
    The shared request refuses it with ``ToolChoice.NONE``, whose
    ``tool_choice`` takes no such key.

    Args:
        request: The shared request.

    Returns:
        ``tool_choice``, or ``MISSING`` when the request sets neither.
    """
    choice = request.tool_choice
    parallel = request.parallel_tool_calls
    if choice is MISSING and parallel is MISSING:
        return MISSING
    if isinstance(choice, ForcedTool):
        config: dict[str, Any] = {"type": "tool", "name": choice.name}
    else:
        mode = ToolChoice.AUTO if choice is MISSING else choice
        config = {"type": TOOL_CHOICES[mode]}
    if parallel is not MISSING:
        config["disable_parallel_tool_use"] = not parallel
    return config


def _output_config(request: GenerationRequest) -> dict[str, Any] | MISSING:
    """Build ``output_config`` from the shared effort and output schema.

    Args:
        request: The shared request.

    Returns:
        The fields of ``output_config`` the request gives, or ``MISSING``
        when it gives neither, so nothing is sent.
    """
    config: dict[str, Any] = {}

    if request.effort is not MISSING:
        config["effort"] = request.effort.value
    if request.output_schema is not MISSING:
        config["format"] = {
            "type": "json_schema",
            "schema": request.output_schema,
        }

    return config or MISSING

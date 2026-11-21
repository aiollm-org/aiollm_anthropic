"""Conversions of Anthropic's Messages types into shared types."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, cast

from pydantic import MISSING, BaseModel

from aiollm.adapters.base import BaseConversion
from aiollm.adapters.stop import Stop
from aiollm.enums import (
    CitationKind,
    IncompleteReason,
    Role,
    ServerToolKind,
    ServerToolStatus,
)
from aiollm.http.exceptions.response import APIResponseError
from aiollm.types.requests.content import (
    ContentPart,
    DocumentPart,
    ImagePart,
    ProviderFile,
)
from aiollm.types.requests.generation import (
    InputItem,
    NativeOutput,
    ToolCall,
    ToolCallTurn,
    ToolResult,
    Turn,
)
from aiollm.types.responses.builtin import Citation, ServerToolCall
from aiollm.types.responses.generation import TokenUsage

from aiollm_anthropic.api import PROVIDER
from aiollm_anthropic.types import (
    Delta,
    Message,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    ServerToolUseBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)

FINISHED: Final = frozenset({"end_turn", "stop_sequence", "tool_use"})
"""Stop reasons of a generation that reached its end.

``tool_use`` is one: the model waits for the tool results, as an
OpenAI response with function calls is ``completed``. Any other reason
(``max_tokens``, ``refusal``, ``model_context_window_exceeded``,
``pause_turn``, a reason Anthropic adds later) stopped it early.
"""

INCOMPLETE_REASONS: Final[Mapping[str, IncompleteReason]] = {
    "max_tokens": IncompleteReason.MAX_TOKENS,
    "refusal": IncompleteReason.REFUSAL,
    "pause_turn": IncompleteReason.PAUSED,
}
"""Shared reason of each early stop reason that has one.

Every other early stop is ``IncompleteReason.OTHER``.
"""

REFUSAL: Final = "refusal"
"""The stop reason of a refusal."""


class InputConversion(
    BaseConversion[str | Sequence[InputItem], list[dict[str, Any]]],
):
    """Converts the shared ``input`` into ``messages``.

    A text becomes one ``user`` turn, and so does a text turn of a list.
    An ``assistant`` turn with Anthropic's own content (``native``: a
    response that used built-in tools, or paused) is sent as that
    content, which holds the encrypted search results Anthropic needs
    back; another provider's ``native`` is not read and the turn's text
    is sent. A turn of calls becomes an ``assistant`` message: its Anthropic
    content blocks when it has them (``native``, thinking with it), else
    a ``text`` block and a ``tool_use`` block per call. The tool results
    after it become one ``user`` message of ``tool_result`` blocks, and a
    ``user`` turn right after them joins that message, after the results,
    as Anthropic requires. The adapter checks first that the arguments
    of the calls are JSON objects.
    """

    def __call__(
        self, source: str | Sequence[InputItem], /
    ) -> list[dict[str, Any]]:
        """Convert the input.

        Args:
            source: The text or the conversation of the shared request.

        Returns:
            The messages.
        """
        if isinstance(source, str):
            return [{"role": Role.USER.value, "content": source}]
        messages: list[dict[str, Any]] = []
        results: list[dict[str, Any]] | None = None
        for item in source:
            match item:
                case ToolResult():
                    block = {
                        "type": "tool_result",
                        "tool_use_id": item.call_id,
                        "content": item.output,
                    }
                    if results is None:
                        results = [block]
                        messages.append(
                            {"role": Role.USER.value, "content": results}
                        )
                    else:
                        results.append(block)
                    continue
                case Turn(role=Role.USER) if results is not None:
                    results += _blocks(item.content)
                case Turn(native=NativeOutput() as native) if (
                    item.role is Role.ASSISTANT and native.provider == PROVIDER
                ):
                    messages.append(
                        {
                            "role": Role.ASSISTANT.value,
                            "content": [dict(b) for b in native.items],
                        }
                    )
                case Turn(content=str()):
                    messages.append(
                        {"role": item.role.value, "content": item.content}
                    )
                case Turn():
                    messages.append(
                        {
                            "role": item.role.value,
                            "content": _blocks(item.content),
                        }
                    )
                case _:
                    messages.append(
                        {
                            "role": Role.ASSISTANT.value,
                            "content": _assistant_blocks(item),
                        }
                    )
            results = None
        return messages


def _blocks(content: str | Sequence[ContentPart]) -> list[dict[str, Any]]:
    """Convert a turn's content into content blocks.

    Args:
        content: The text, or the parts.

    Returns:
        The blocks: one ``text`` block for a text, else one per part.
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [_block(part) for part in content]


def _block(part: ContentPart) -> dict[str, Any]:
    """Convert a part of a turn into a content block.

    Text becomes a ``text`` block, an image an ``image`` block and a
    document a ``document`` block, with a ``base64`` source for bytes,
    a ``url`` source for a URL and a ``file`` source for a stored file,
    and a document's ``filename`` as its ``title``. The adapter has
    checked that a file is Anthropic's.

    Args:
        part: The part.

    Returns:
        The block.
    """
    match part:
        case ImagePart() | DocumentPart():
            if isinstance(part.data, bytes):
                source = {
                    "type": "base64",
                    "media_type": part.media_type,
                    "data": base64.b64encode(part.data).decode(),
                }
            elif isinstance(part.file, ProviderFile):
                source = {"type": "file", "file_id": part.file.id}
            else:
                source = {"type": "url", "url": part.url}
            block: dict[str, Any] = {"type": part.type, "source": source}
            if isinstance(part, DocumentPart) and part.filename is not MISSING:
                block["title"] = part.filename
            return block
        case _:
            return {"type": "text", "text": part.text}


def _assistant_blocks(turn: ToolCallTurn) -> list[dict[str, Any]]:
    """Build the content of a turn of calls.

    Args:
        turn: The turn.

    Returns:
        Its Anthropic content blocks when it has them, else a ``text``
        block and a ``tool_use`` block per call, the arguments parsed.
    """
    native = turn.native
    if native is not MISSING and native.provider == PROVIDER:
        return [dict(block) for block in native.items]
    blocks: list[dict[str, Any]] = []
    if turn.text is not MISSING:
        blocks.append({"type": "text", "text": turn.text})
    blocks += [
        {
            "type": "tool_use",
            "id": call.id,
            "name": call.name,
            "input": call.parse_arguments(),
        }
        for call in turn.tool_calls
    ]
    return blocks


class UsageConversion(BaseConversion[Usage, TokenUsage]):
    """Converts Anthropic's token counts.

    Anthropic counts input read from and written to the cache apart from
    the rest; the shared ``input_tokens`` is their sum, and
    ``total_tokens`` adds the output.
    """

    def __call__(self, source: Usage, /) -> TokenUsage:
        """Convert the counts of a message.

        Args:
            source: Token counts of a message.

        Returns:
            The shared token counts.
        """
        cache_write = source.cache_creation_input_tokens
        cache_read = source.cache_read_input_tokens
        input_tokens = (
            source.input_tokens + (cache_write or 0) + (cache_read or 0)
        )
        details = source.output_tokens_details
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=source.output_tokens,
            total_tokens=input_tokens + source.output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            reasoning_tokens=(
                None if details is None else details.thinking_tokens
            ),
        )


class StopConversion(BaseConversion[Message | Delta, Stop]):
    """Tells how a message stopped.

    It takes a ``Message``, or for a stream the ``Delta`` that
    ``StreamState`` collects: both carry ``stop_reason`` and
    ``stop_details``.
    """

    def __call__(self, source: Message | Delta, /) -> Stop:
        """Convert the stop of a finished message.

        Args:
            source: The message, or the collected changes of a stream.

        Returns:
            Nothing set if the message reached its end; else the shared and
            Anthropic's reason, and for a refusal its explanation, else its
            category.

        Raises:
            APIResponseError: There is no stop reason, which Anthropic
                sends on every finished message.
        """
        stop_reason = source.stop_reason
        if stop_reason is None:
            message = "Anthropic message finished without a stop_reason"
            raise APIResponseError(message)
        if stop_reason in FINISHED:
            return Stop()
        details = source.stop_details
        refusal = None
        if stop_reason == REFUSAL and details is not None:
            refusal = details.explanation or details.category
        return Stop(
            INCOMPLETE_REASONS.get(stop_reason, IncompleteReason.OTHER),
            stop_reason,
            refusal,
        )


def message_text(message: Message) -> str | None:
    """Join the text of a message.

    Args:
        message: A message.

    Returns:
        The text of all ``text`` blocks, or ``None`` if there are none.
    """
    texts = [
        block.text for block in message.content if isinstance(block, TextBlock)
    ]
    return "".join(texts) if texts else None


def tool_call(block: ToolUseBlock) -> ToolCall:
    """Turn a ``tool_use`` block into a shared call.

    Args:
        block: The block.

    Returns:
        The call, its ``input`` written as JSON text.
    """
    arguments = json.dumps(block.input, ensure_ascii=False)
    return ToolCall(id=block.id, name=block.name, arguments=arguments)


def message_tool_calls(message: Message) -> list[ToolCall]:
    """Collect the tool calls of a message.

    Args:
        message: A message.

    Returns:
        A shared call per ``tool_use`` block, in content order; empty
        without calls.
    """
    return [
        tool_call(block)
        for block in message.content
        if isinstance(block, ToolUseBlock)
    ]


def message_blocks(message: Message) -> list[dict[str, Any]]:
    """Dump a message's content blocks as Anthropic sent them.

    Args:
        message: A message.

    Returns:
        Each block with only the fields Anthropic sent.
    """
    return [
        block.model_dump(mode="json", by_alias=True, exclude_unset=True)
        for block in message.content
    ]


def message_native(message: Message) -> NativeOutput | None:
    """Keep a message's content as Anthropic's next request takes it.

    Args:
        message: A message.

    Returns:
        Its content blocks with only the fields Anthropic sent (thinking
        with its ``signature``, text, ``tool_use``, the calls and results
        of built-in tools), which an ``assistant`` message takes back;
        ``None`` without content.
    """
    blocks = message_blocks(message)
    return NativeOutput(provider=PROVIDER, items=blocks) if blocks else None


SERVER_TOOL_KINDS: Final[Mapping[str, ServerToolKind]] = {
    "web_search": ServerToolKind.WEB_SEARCH,
    "web_fetch": ServerToolKind.WEB_FETCH,
    "code_execution": ServerToolKind.CODE_EXECUTION,
    "bash_code_execution": ServerToolKind.CODE_EXECUTION,
    "text_editor_code_execution": ServerToolKind.CODE_EXECUTION,
}
"""Shared kind of each built-in tool Anthropic names in ``server_tool_use``.

Any other name (the tool search tools, one Anthropic adds later) is
``ServerToolKind.OTHER``.
"""

CODE_INPUTS: Final[Mapping[str, str]] = {
    "code_execution": "code",
    "bash_code_execution": "command",
}
"""The ``input`` field that holds what a code execution runs, by tool."""

RESULT_SUFFIX: Final = "_tool_result"
"""Ends the ``type`` of every block that holds a built-in tool's result."""

ERROR_SUFFIX: Final = "_tool_result_error"
"""Ends the ``type`` of a result's ``content`` that tells a failure."""

DOCUMENT_LOCATIONS: Final = frozenset(
    {"char_location", "page_location", "content_block_location"}
)
"""Citation types that point into a document the request sent."""


def server_tool_calls(
    blocks: Sequence[Mapping[str, Any]],
) -> list[ServerToolCall]:
    """Collect the built-in tool calls of content blocks.

    A call is a ``server_tool_use`` block and, after it, the result block
    that names its id in ``tool_use_id``.

    Args:
        blocks: The content blocks of a message, as Anthropic sent them.

    Returns:
        A call per ``server_tool_use`` block, in order. It is
        ``COMPLETED`` once its result came, ``FAILED`` (with the result's
        ``error_code``) when the result is an error, and ``INCOMPLETE``
        without a result, such as a paused or cut off message. ``raw``
        holds the call block and its result block.
    """
    results = {
        block["tool_use_id"]: block
        for block in blocks
        if str(block.get("type", "")).endswith(RESULT_SUFFIX)
        and "tool_use_id" in block
    }
    return [
        _server_call(dict(block), results.get(block["id"]))
        for block in blocks
        if block.get("type") == "server_tool_use"
    ]


def _server_call(
    block: dict[str, Any], result: Mapping[str, Any] | None
) -> ServerToolCall:
    """Build the call of a ``server_tool_use`` block and its result.

    Args:
        block: The ``server_tool_use`` block.
        result: The block with its result, if it came.

    Returns:
        The shared call.
    """
    name = str(block.get("name", ""))
    arguments = block.get("input")
    arguments = arguments if isinstance(arguments, dict) else {}
    status = ServerToolStatus.INCOMPLETE
    error = None
    if result is not None:
        content = result.get("content")
        failed = isinstance(content, dict) and str(
            content.get("type", "")
        ).endswith(ERROR_SUFFIX)
        status = (
            ServerToolStatus.FAILED if failed else ServerToolStatus.COMPLETED
        )
        if failed and isinstance(content, dict):
            error = str(content.get("error_code") or content["type"])
    query = arguments.get("query")
    field = CODE_INPUTS.get(name)
    code = arguments.get(field) if field else None
    return ServerToolCall(
        id=block["id"],
        kind=SERVER_TOOL_KINDS.get(name, ServerToolKind.OTHER),
        status=status,
        queries=[query]
        if name == "web_search" and isinstance(query, str)
        else [],
        code=code if isinstance(code, str) else None,
        error=error,
        raw=[block] if result is None else [block, dict(result)],
    )


def block_citations(blocks: Sequence[Mapping[str, Any]]) -> list[Citation]:
    """Collect the citations of the text blocks.

    Anthropic attaches a citation to the whole ``text`` block, so
    ``start_index`` and ``end_index`` are where that block sits in the
    joined text of the message. A page of a web search is a ``URL``
    citation, a part of a stored file a ``FILE`` one; a part of a
    document that came inline, or of search results the request sent,
    has no stable source and stays in ``raw`` and ``native``.

    Args:
        blocks: The content blocks of a message, as Anthropic sent them.

    Returns:
        The citations, in order.
    """
    citations: list[Citation] = []
    offset = 0
    for block in blocks:
        if block.get("type") != "text":
            continue
        text = str(block.get("text", ""))
        start, offset = offset, offset + len(text)
        for item in block.get("citations") or []:
            citation = _citation(item, start, offset)
            if citation is not None:
                citations.append(citation)
    return citations


def _citation(
    item: Mapping[str, Any], start: int, end: int
) -> Citation | None:
    """Turn a citation of a text block into a shared one.

    Args:
        item: The citation as Anthropic sent it.
        start: Where its text block starts in the message's text.
        end: Where its text block ends.

    Returns:
        The shared citation, or ``None`` for a source with no stable
        identity.
    """
    kind = item.get("type")
    if kind == "web_search_result_location":
        return Citation(
            kind=CitationKind.URL,
            url=item.get("url"),
            title=item.get("title"),
            cited_text=item.get("cited_text"),
            start_index=start,
            end_index=end,
        )
    if kind in DOCUMENT_LOCATIONS and item.get("file_id"):
        return Citation(
            kind=CitationKind.FILE,
            title=item.get("document_title"),
            file_id=item["file_id"],
            cited_text=item.get("cited_text"),
            start_index=start,
            end_index=end,
        )
    return None


def message_server_calls(message: Message) -> list[ServerToolCall]:
    """Collect the built-in tool calls of a message.

    Args:
        message: A message.

    Returns:
        Its calls, as ``server_tool_calls`` gives them; empty without.
    """
    return server_tool_calls(message_blocks(message))


def message_citations(message: Message) -> list[Citation]:
    """Collect the citations of a message.

    Args:
        message: A message.

    Returns:
        Its citations, as ``block_citations`` gives them; empty without.
    """
    return block_citations(message_blocks(message))


def message_thinking(message: Message) -> str | None:
    """Join the thinking of a message.

    Args:
        message: A message.

    Returns:
        The text of all ``thinking`` blocks, or ``None`` if there are
        none or all are empty (thinking whose display is ``omitted``).
    """
    thinking = "".join(
        block.thinking
        for block in message.content
        if isinstance(block, ThinkingBlock)
    )
    return thinking or None


INPUT_CONVERSION: Final = InputConversion()
"""The default conversion of the shared ``input``."""

USAGE_CONVERSION: Final = UsageConversion()
"""The default conversion of token counts."""

STOP_CONVERSION: Final = StopConversion()
"""The default conversion of a message's stop."""


@dataclass
class StreamState:
    """What a message stream has told so far about the whole message.

    ``message_start`` carries the message's ``Usage``, and every
    ``message_delta`` the stop reason and the cumulative counts, so the
    counts are replaced, never added up. At the end the state gives a
    real ``Usage`` and ``Delta``, so a stream goes through the
    same conversions as a message, and the message's id, model and
    whole text. It also keeps every content block, from its
    ``content_block_start`` and its deltas, to give the message's
    content back as ``native`` (thinking with its ``signature`` among
    it, the citations of its text, the calls and results of built-in
    tools) and each tool call once its block stops, and each built-in
    tool call when it starts and when its result comes.

    Attributes:
        delta: The stop reason and details sent so far.
    """

    delta: Delta = field(default_factory=Delta)
    _message: Message | None = field(default=None, init=False, repr=False)
    _texts: list[str] = field(default_factory=list, init=False, repr=False)
    _thinking: list[str] = field(default_factory=list, init=False, repr=False)
    _calls: dict[int, ToolUseBlock] = field(
        default_factory=dict[int, ToolUseBlock], init=False, repr=False
    )
    _arguments: dict[int, list[str]] = field(
        default_factory=dict[int, list[str]], init=False, repr=False
    )
    _blocks: dict[int, dict[str, Any]] = field(
        default_factory=dict[int, dict[str, Any]], init=False, repr=False
    )
    _pieces: dict[int, dict[str, list[str]]] = field(
        default_factory=dict[int, dict[str, list[str]]],
        init=False,
        repr=False,
    )
    _servers: dict[int, ServerToolUseBlock] = field(
        default_factory=dict[int, ServerToolUseBlock],
        init=False,
        repr=False,
    )

    def start(self, event: RawMessageStartEvent) -> None:
        """Take the message and the counts of ``message_start``.

        Args:
            event: The first event of the stream.
        """
        self._message = event.message

    def add_text(self, text: str) -> None:
        """Keep a text delta for the whole text.

        Args:
            text: The new text of a ``text_delta``.
        """
        self._texts.append(text)

    def start_block(self, index: int, block: BaseModel) -> None:
        """Keep a block that a ``content_block_start`` opened.

        Args:
            index: The block's index in the message.
            block: The block as it starts: a ``tool_use`` with an empty
                ``input``, text and thinking without their deltas yet.
        """
        self._blocks[index] = block.model_dump(
            mode="json", by_alias=True, exclude_unset=True
        )
        self._pieces[index] = {}
        if isinstance(block, ToolUseBlock):
            self._calls[index] = block
            self._arguments[index] = []
        elif isinstance(block, ServerToolUseBlock):
            self._servers[index] = block
            self._arguments[index] = []

    def add_piece(self, index: int, key: str, piece: str) -> None:
        """Keep a piece of a block's text, thinking or signature.

        A block whose ``content_block_start`` did not come (some of the
        documented streams leave it out) is opened from the piece: a
        ``text`` block for text, a ``thinking`` block for thinking or a
        signature, so its content still comes back whole.

        Args:
            index: The block's index.
            key: The field the piece belongs to: ``text``, ``thinking``
                or ``signature``.
            piece: The piece.
        """
        if index not in self._pieces:
            kind = "text" if key == "text" else "thinking"
            self._blocks[index] = {"type": kind}
            self._pieces[index] = {}
        self._pieces[index].setdefault(key, []).append(piece)

    def add_citation(self, index: int, citation: BaseModel) -> None:
        """Keep a citation of a text block.

        A text block whose ``content_block_start`` did not come is
        opened from the citation, as ``add_piece`` does.

        Args:
            index: The text block's index.
            citation: The citation of a ``citations_delta``.
        """
        if index not in self._pieces:
            self._blocks[index] = {"type": "text"}
            self._pieces[index] = {}
        block = self._blocks[index]
        block["citations"] = [
            *(block.get("citations") or []),
            citation.model_dump(
                mode="json", by_alias=True, exclude_unset=True
            ),
        ]

    def server_started(self, index: int) -> ServerToolCall:
        """Give the built-in tool call that a block starts.

        Args:
            index: The index of a ``server_tool_use`` block that just
                started.

        Returns:
            The call as it is known at ``content_block_start``, its
            input not streamed yet and its status ``IN_PROGRESS``.
        """
        [call] = server_tool_calls([self._blocks[index]])
        return call.model_copy(update={"status": ServerToolStatus.IN_PROGRESS})

    def server_done(self, index: int) -> ServerToolCall | None:
        """Give the built-in tool call that a result block finishes.

        Args:
            index: The index of the block that just started.

        Returns:
            The whole call, its input from the pieces streamed for it,
            when the block is the result of a ``server_tool_use`` block;
            ``None`` for any other block.

        Raises:
            APIResponseError: The call's input pieces do not join into a
                JSON object.
        """
        result = self._blocks[index]
        if not str(result.get("type", "")).endswith(RESULT_SUFFIX):
            return None
        for opened, block in self._servers.items():
            if block.id == result.get("tool_use_id"):
                [call] = server_tool_calls([self._finished(opened), result])
                return call
        return None

    def call(self, index: int) -> ToolCall | None:
        """Give the tool call of a block, once its block stops.

        Args:
            index: The block's index.

        Returns:
            The call, or ``None`` if the block is no ``tool_use``.
        """
        if index not in self._calls:
            return None
        return self._call(index)

    def _call(self, index: int) -> ToolCall:
        """Build the call of a ``tool_use`` block.

        Args:
            index: The block's index.

        Returns:
            The call, its arguments the pieces joined, or the block's own
            ``input`` when no piece came.
        """
        block = self._calls[index]
        pieces = self._arguments[index]
        if not pieces:
            return tool_call(block)
        return ToolCall(
            id=block.id, name=block.name, arguments="".join(pieces)
        )

    def native(self) -> NativeOutput | None:
        """Give the streamed content as Anthropic's next request takes it.

        Returns:
            Each block as it started, with its text, thinking and
            signature pieces joined into it and a ``tool_use`` block's
            ``input`` parsed from its pieces; ``None`` without blocks.

        Raises:
            APIResponseError: A ``tool_use`` block's pieces do not join
                into a JSON object.
        """
        if not self._blocks:
            return None
        blocks = [self._finished(index) for index in sorted(self._blocks)]
        return NativeOutput(provider=PROVIDER, items=blocks)

    def _finished(self, index: int) -> dict[str, Any]:
        """Join a block's pieces into it.

        Args:
            index: The block's index.

        Returns:
            The block as it started, with its text, thinking and
            signature pieces joined into it, and the ``input`` of a call
            parsed from its pieces.

        Raises:
            APIResponseError: A call's pieces do not join into a JSON
                object.
        """
        block = dict(self._blocks[index])
        for key, pieces in self._pieces[index].items():
            block[key] = block.get(key, "") + "".join(pieces)
        if index in self._calls:
            block["input"] = _input(self._call(index))
        elif self._arguments.get(index):
            block["input"] = _object("".join(self._arguments[index]))
        return block

    def add_arguments(self, index: int, partial: str) -> str | None:
        """Keep a piece of a call's arguments.

        Args:
            index: The index of the call's block.
            partial: The ``partial_json`` of an ``input_json_delta``.

        Returns:
            The id of the call the piece belongs to; ``None`` for a piece
            of a built-in tool's input, which is no call of yours.

        Raises:
            APIResponseError: No ``tool_use`` or ``server_tool_use``
                block opened at the index.
        """
        if index not in self._arguments:
            message = (
                f"Anthropic stream sent input_json_delta for block {index} "
                "without its tool_use"
            )
            raise APIResponseError(message)
        self._arguments[index].append(partial)
        return self._calls[index].id if index in self._calls else None

    def tool_calls(self) -> list[ToolCall]:
        """Give the tool calls streamed so far.

        Returns:
            A shared call per ``tool_use`` block, in block order, its
            arguments the pieces joined, or the block's own ``input``
            when no piece came.
        """
        return [self._call(index) for index in sorted(self._calls)]

    def add_thinking(self, thinking: str) -> None:
        """Keep a thinking delta for the whole reasoning.

        Args:
            thinking: The new text of a ``thinking_delta``.
        """
        self._thinking.append(thinking)

    def thinking(self) -> str | None:
        """Join the thinking deltas.

        Returns:
            The thinking received, or ``None`` if there was none.
        """
        return "".join(self._thinking) or None

    def update(self, event: RawMessageDeltaEvent) -> None:
        """Take the stop reason and counts of a ``message_delta``.

        Args:
            event: A ``message_delta`` event.

        Raises:
            APIResponseError: No ``message_start`` came before it to
                take its counts into.
        """
        changes = _sent(event.delta)
        if changes:
            self.delta = self.delta.model_copy(update=changes)
        if self._message is None:
            message = (
                "Anthropic stream sent message_delta before message_start"
            )
            raise APIResponseError(message)
        counts = self._message.usage.model_copy(update=_sent(event.usage))
        self._message = self._message.model_copy(update={"usage": counts})

    def usage(self) -> Usage:
        """Give the counts of the whole message.

        Returns:
            The counts of ``message_start`` with the last cumulative
            counts of ``message_delta``.

        Raises:
            APIResponseError: The stream had no ``message_start``.
        """
        return self.message().usage

    def message(self) -> Message:
        """Give the message of ``message_start``.

        Returns:
            The message as the stream started it, for its id and model,
            with the counts of the last ``message_delta``.

        Raises:
            APIResponseError: The stream had no ``message_start``.
        """
        if self._message is None:
            message = "Anthropic stream finished without message_start"
            raise APIResponseError(message)
        return self._message

    def text(self) -> str | None:
        """Give the whole text streamed so far.

        Returns:
            The text deltas joined, or ``None`` if none came, as
            ``message_text`` gives for a message without text.
        """
        return "".join(self._texts) if self._texts else None


def _input(call: ToolCall) -> dict[str, Any]:
    """Parse a streamed call's arguments into ``tool_use.input``.

    Args:
        call: The call.

    Returns:
        Its arguments as an object.

    Raises:
        APIResponseError: The arguments are not a JSON object.
    """
    try:
        return call.parse_arguments()
    except ValueError as error:
        raise APIResponseError(str(error)) from error


def _object(text: str) -> dict[str, Any]:
    """Parse the joined pieces of a built-in tool's input.

    Args:
        text: The pieces joined; blank when Anthropic sent only empty
            ones, as it does for an input with no field.

    Returns:
        The input as an object.

    Raises:
        APIResponseError: The text is not a JSON object.
    """
    try:
        value = json.loads(text) if text.strip() else {}
    except ValueError as error:
        raise APIResponseError(str(error)) from error
    if not isinstance(value, dict):
        message = "Anthropic stream sent a tool input that is no JSON object"
        raise APIResponseError(message)
    return cast(dict[str, Any], value)


def _sent(model: BaseModel) -> dict[str, object]:
    """Take the fields of a model that have a value.

    The values stay as they are, nested models included, so they can
    update another model with ``model_copy``.

    Args:
        model: A partial update, such as a ``message_delta``'s counts.

    Returns:
        Each field that is not ``None``, with its value.
    """
    return {
        name: value
        for name in type(model).model_fields
        if (value := getattr(model, name)) is not None
    }

"""Built-in tools at Anthropic: the tools, their calls and their sources.

The shared scenarios of the test kit cover a web search end to end; these
cover what is Anthropic's own: the other server tools, the versions, the
shape of a citation, and a stream that sends its pieces in odd ways.
"""

from __future__ import annotations

from typing import Any

import pytest

from aiollm.enums import (
    CitationKind,
    IncompleteReason,
    Provider,
    Role,
    ServerToolKind,
    ServerToolPhase,
    ServerToolStatus,
)
from aiollm.exceptions import RequestValidationError
from aiollm.http.exceptions.response import APIResponseError
from aiollm.types.requests import (
    CodeExecution,
    GenerationRequest,
    ProviderFile,
    ProviderTool,
    TextPart,
    Turn,
    UserLocation,
    WebSearch,
)
from aiollm.types.requests.generation import NativeOutput
from aiollm.types.responses import (
    CompletedEvent,
    OtherEvent,
    ServerToolCallEvent,
)

from aiollm_anthropic import tools
from aiollm_anthropic.adapters.conversions import StreamState, block_files
from aiollm_anthropic.api import PROVIDER
from aiollm_anthropic.methods import CreateMessage
from aiollm_anthropic.methods.params import UserLocationParam
from tests.test_generation import REQUEST, adapter, convert, message
from tests.test_generation_contract import (
    CITING_TEXT,
    SEARCH_CALL,
    SEARCH_RESULT,
    anthropic_message,
    anthropic_searched_stream,
)
from tests.test_messages import STREAM


def request(*tool_list: object, **fields: Any) -> GenerationRequest:
    """The reference request with tools."""
    data = REQUEST.model_dump(by_alias=True) | {"tools": list(tool_list)}
    return GenerationRequest.model_validate(data | fields)


def body_tools(*tool_list: object, **fields: Any) -> list[dict[str, Any]]:
    """The ``tools`` of the method the adapter builds."""
    method = adapter.to_method(request(*tool_list, **fields))
    assert isinstance(method, CreateMessage)
    return method.get_body_map()["tools"]


def code_call(
    name: str, arguments: dict[str, Any], content: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Blocks of a call of a code execution tool, and its result."""
    call = {
        "type": "server_tool_use",
        "id": "srv_1",
        "name": name,
        "input": arguments,
    }
    if content is None:
        return [call]
    return [
        call,
        {
            "type": f"{name}_tool_result",
            "tool_use_id": "srv_1",
            "content": content,
        },
    ]


class TestRequest:
    def test_a_web_search_sends_what_it_has(self) -> None:
        search = WebSearch(
            allowed_domains=["example.com"],
            user_location=UserLocation(
                country="FR", region="IDF", city="Paris", timezone="CET"
            ),
        )

        assert body_tools(search) == [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "allowed_domains": ["example.com"],
                "user_location": {
                    "type": "approximate",
                    "country": "FR",
                    "region": "IDF",
                    "city": "Paris",
                    "timezone": "CET",
                },
            }
        ]

    def test_a_code_execution_sends_its_first_current_version(self) -> None:
        assert body_tools(CodeExecution()) == [
            {"type": "code_execution_20250825", "name": "code_execution"}
        ]

    def test_a_provider_tool_goes_as_its_definition(self) -> None:
        fetch = tools.web_fetch(max_uses=2, citations=True)

        assert body_tools(fetch) == [
            {
                "name": "web_fetch",
                "type": "web_fetch_20250910",
                "citations": {"enabled": True},
                "max_uses": 2,
            }
        ]
        assert fetch.definition["max_uses"] == 2

    def test_a_request_of_built_in_tools_only_has_no_tool_choice(
        self,
    ) -> None:
        method = adapter.to_method(request(WebSearch()))

        assert "tool_choice" not in method.get_body_map()

    def test_a_tool_of_another_provider_is_refused(self) -> None:
        other = ProviderTool(
            provider=Provider("someone-else"), definition={"type": "x"}
        )

        with pytest.raises(RequestValidationError) as caught:
            adapter.to_method(request(other))

        assert caught.value.field == "tools"

    def test_the_definition_is_copied(self) -> None:
        tool = tools.web_search(allowed_domains=["a.com"])

        [sent] = body_tools(tool)
        sent["allowed_domains"].append("b.com")

        assert tool.definition["allowed_domains"] == ["a.com"]


class TestTools:
    @pytest.mark.parametrize("version", ["20250305", "20260209", "20260318"])
    def test_web_search_has_every_version(self, version: Any) -> None:
        tool = tools.web_search(version=version)

        assert tool.provider == PROVIDER
        assert tool.definition == {
            "name": "web_search",
            "type": f"web_search_{version}",
        }

    @pytest.mark.parametrize(
        "version", ["20250910", "20260209", "20260309", "20260318"]
    )
    def test_web_fetch_has_every_version(self, version: Any) -> None:
        assert tools.web_fetch(version=version).definition["type"] == (
            f"web_fetch_{version}"
        )

    @pytest.mark.parametrize(
        "version", ["20250522", "20250825", "20260120", "20260521"]
    )
    def test_code_execution_has_every_version(self, version: Any) -> None:
        assert tools.code_execution(version=version).definition == {
            "name": "code_execution",
            "type": f"code_execution_{version}",
        }

    def test_web_search_takes_anthropics_options(self) -> None:
        tool = tools.web_search(
            blocked_domains=["spam.com"],
            max_uses=3,
            user_location=UserLocationParam(city="Paris"),
        )

        assert tool.definition == {
            "name": "web_search",
            "type": "web_search_20250305",
            "blocked_domains": ["spam.com"],
            "max_uses": 3,
            "user_location": {"type": "approximate", "city": "Paris"},
        }

    def test_web_fetch_takes_its_options(self) -> None:
        allowed = tools.web_fetch(
            allowed_domains=["example.com"], max_content_tokens=1000
        )
        blocked = tools.web_fetch(blocked_domains=["spam.com"])

        assert allowed.definition["allowed_domains"] == ["example.com"]
        assert allowed.definition["max_content_tokens"] == 1000
        assert blocked.definition["blocked_domains"] == ["spam.com"]

    @pytest.mark.parametrize("function", [tools.web_search, tools.web_fetch])
    def test_the_two_lists_of_domains_do_not_mix(self, function: Any) -> None:
        with pytest.raises(ValueError, match="together"):
            function(allowed_domains=["a.com"], blocked_domains=["b.com"])

    @pytest.mark.parametrize(
        "function",
        [tools.web_search, tools.web_fetch, tools.code_execution],
    )
    def test_an_unknown_version_is_refused(self, function: Any) -> None:
        with pytest.raises(ValueError, match="no version"):
            function(version="1999")

    def test_a_value_the_version_refuses_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_uses"):
            tools.web_search(max_uses="many")  # type: ignore[arg-type]


class TestCalls:
    def test_a_bash_call_has_its_command_as_code(self) -> None:
        blocks = code_call(
            "bash_code_execution",
            {"command": "echo 1"},
            {
                "type": "bash_code_execution_result",
                "stdout": "1\n",
                "stderr": "",
                "return_code": 0,
                "content": [],
            },
        )

        response = adapter.to_response(
            message(content=[*blocks, {"type": "text", "text": "Done."}]),
            REQUEST,
        )

        [call] = response.server_tool_calls
        assert call.kind is ServerToolKind.CODE_EXECUTION
        assert call.status is ServerToolStatus.COMPLETED
        assert call.code == "echo 1"
        assert call.queries == []
        assert [raw["type"] for raw in call.raw] == [
            "server_tool_use",
            "bash_code_execution_tool_result",
        ]

    def test_a_python_call_has_its_code(self) -> None:
        blocks = code_call("code_execution", {"code": "print(1)"}, None)

        response = adapter.to_response(message(content=blocks), REQUEST)

        [call] = response.server_tool_calls
        assert call.kind is ServerToolKind.CODE_EXECUTION
        assert call.code == "print(1)"
        assert call.status is ServerToolStatus.INCOMPLETE

    def test_a_text_editor_call_has_no_code(self) -> None:
        blocks = code_call(
            "text_editor_code_execution", {"command": "view"}, None
        )

        [call] = adapter.to_response(
            message(content=blocks), REQUEST
        ).server_tool_calls

        assert call.kind is ServerToolKind.CODE_EXECUTION
        assert call.code is None

    def test_a_failed_result_gives_its_error_code(self) -> None:
        blocks = code_call(
            "code_execution",
            {"code": "x"},
            {
                "type": "code_execution_tool_result_error",
                "error_code": "unavailable",
            },
        )

        [call] = adapter.to_response(
            message(content=blocks), REQUEST
        ).server_tool_calls

        assert call.status is ServerToolStatus.FAILED
        assert call.error == "unavailable"

    def test_a_code_that_exits_badly_is_still_a_completed_call(self) -> None:
        blocks = code_call(
            "bash_code_execution",
            {"command": "false"},
            {
                "type": "bash_code_execution_result",
                "stdout": "",
                "stderr": "",
                "return_code": 1,
                "content": [],
            },
        )

        [call] = adapter.to_response(
            message(content=blocks), REQUEST
        ).server_tool_calls

        assert call.status is ServerToolStatus.COMPLETED
        assert call.error is None

    def test_a_fetch_is_a_web_fetch(self) -> None:
        blocks = [
            {
                "type": "server_tool_use",
                "id": "srv_2",
                "name": "web_fetch",
                "input": {"url": "https://example.com"},
            },
            {
                "type": "web_fetch_tool_result",
                "tool_use_id": "srv_2",
                "content": {
                    "type": "web_fetch_tool_result_error",
                    "error_code": "url_not_accessible",
                },
            },
        ]

        [call] = adapter.to_response(
            message(content=blocks), REQUEST
        ).server_tool_calls

        assert call.kind is ServerToolKind.WEB_FETCH
        assert call.status is ServerToolStatus.FAILED
        assert call.error == "url_not_accessible"
        assert call.queries == []

    def test_an_unknown_tool_is_other(self) -> None:
        blocks = [
            {
                "type": "server_tool_use",
                "id": "srv_3",
                "name": "tool_search_tool_regex",
                "input": {"pattern": "x"},
            },
            {
                "type": "tool_search_tool_result",
                "tool_use_id": "srv_3",
                "content": {
                    "type": "tool_search_tool_search_result",
                    "tool_references": [],
                },
            },
        ]

        [call] = adapter.to_response(
            message(content=blocks), REQUEST
        ).server_tool_calls

        assert call.kind is ServerToolKind.OTHER
        assert call.status is ServerToolStatus.COMPLETED

    def test_a_search_without_a_string_query_has_no_queries(self) -> None:
        blocks = [SEARCH_CALL | {"input": {"query": 5}}]

        [call] = adapter.to_response(
            message(content=blocks), REQUEST
        ).server_tool_calls

        assert call.queries == []

    def test_two_searches_keep_their_order(self) -> None:
        second = SEARCH_CALL | {"id": "search_2", "input": {"query": "b"}}
        results = SEARCH_RESULT | {"tool_use_id": "search_2"}

        response = adapter.to_response(
            message(content=[SEARCH_CALL, second, results, SEARCH_RESULT]),
            REQUEST,
        )

        assert [
            (call.id, call.queries, call.status)
            for call in response.server_tool_calls
        ] == [
            ("search_1", ["weather in Paris"], ServerToolStatus.COMPLETED),
            ("search_2", ["b"], ServerToolStatus.COMPLETED),
        ]

    def test_a_pause_is_paused(self) -> None:
        response = adapter.to_response(
            message(content=[SEARCH_CALL], stop_reason="pause_turn"),
            REQUEST,
        )

        assert response.incomplete_reason is IncompleteReason.PAUSED
        assert response.raw_incomplete_reason == "pause_turn"
        assert response.text is None
        turn = response.to_turn()
        assert isinstance(turn, Turn)
        assert turn.native == NativeOutput(
            provider=PROVIDER, items=[SEARCH_CALL]
        )


class TestCitations:
    def test_a_page_of_a_search_is_a_url(self) -> None:
        response = adapter.to_response(
            anthropic_message_of([SEARCH_CALL, SEARCH_RESULT, CITING_TEXT]),
            REQUEST,
        )

        [citation] = response.citations
        assert citation.kind is CitationKind.URL
        assert citation.url == "https://example.com/paris"
        assert citation.title == "Paris weather"
        assert citation.cited_text == "It is sunny in Paris."
        assert (citation.start_index, citation.end_index) == (
            0,
            len("Hi! My name is Claude."),
        )

    def test_each_text_block_has_its_own_span(self) -> None:
        first = {"type": "text", "text": "One. "}
        second = CITING_TEXT | {"text": "Two."}

        response = adapter.to_response(
            anthropic_message_of([first, SEARCH_CALL, second]), REQUEST
        )

        [citation] = response.citations
        assert response.text == "One. Two."
        assert (citation.start_index, citation.end_index) == (5, 9)
        assert response.text is not None
        assert response.text[citation.start_index : citation.end_index] == (
            "Two."
        )

    @pytest.mark.parametrize(
        "location",
        [
            {
                "type": "char_location",
                "start_char_index": 0,
                "end_char_index": 4,
            },
            {
                "type": "page_location",
                "start_page_number": 1,
                "end_page_number": 2,
            },
            {
                "type": "content_block_location",
                "start_block_index": 0,
                "end_block_index": 1,
            },
        ],
    )
    def test_a_part_of_a_stored_file_is_a_file(
        self, location: dict[str, Any]
    ) -> None:
        text = {
            "type": "text",
            "text": "From the file.",
            "citations": [
                location
                | {
                    "cited_text": "words",
                    "document_index": 0,
                    "document_title": "Report",
                    "file_id": "file_1",
                }
            ],
        }

        [citation] = adapter.to_response(
            anthropic_message_of([text]), REQUEST
        ).citations

        assert citation.kind is CitationKind.FILE
        assert citation.file_id == "file_1"
        assert citation.title == "Report"
        assert citation.cited_text == "words"

    def test_a_part_of_an_inline_document_is_left_out(self) -> None:
        text = {
            "type": "text",
            "text": "From the document.",
            "citations": [
                {
                    "type": "char_location",
                    "cited_text": "words",
                    "document_index": 0,
                    "start_char_index": 0,
                    "end_char_index": 5,
                },
                {
                    "type": "search_result_location",
                    "cited_text": "more",
                    "search_result_index": 0,
                    "source": "my-database",
                    "start_block_index": 0,
                    "end_block_index": 1,
                },
            ],
        }

        response = adapter.to_response(anthropic_message_of([text]), REQUEST)

        assert response.citations == []
        assert response.native is not None
        assert response.native.items[0]["citations"] == text["citations"]


def anthropic_message_of(content: list[dict[str, Any]]) -> Any:
    """A parsed message of the given blocks."""
    return message(**anthropic_message(content))


def stream_of(
    *blocks: list[dict[str, Any]], stop_reason: str = "end_turn"
) -> list[dict[str, Any]]:
    """The reference stream with other events between its ends."""
    start, *_, stop = STREAM
    return [
        start,
        *(event for events in blocks for event in events),
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": {"output_tokens": 3},
        },
        stop,
    ]


def block_events(
    index: int, block: dict[str, Any], *deltas: dict[str, Any]
) -> list[dict[str, Any]]:
    """A content block: its start, its deltas and its stop."""
    return [
        {
            "type": "content_block_start",
            "index": index,
            "content_block": block,
        },
        *(
            {"type": "content_block_delta", "index": index, "delta": delta}
            for delta in deltas
        ),
        {"type": "content_block_stop", "index": index},
    ]


class TestStream:
    async def test_a_search_streams_as_started_then_done(self) -> None:
        received = await convert(anthropic_searched_stream())

        tool_events = [
            e for e in received if isinstance(e, ServerToolCallEvent)
        ]
        assert [(e.phase, e.call.status) for e in tool_events] == [
            (ServerToolPhase.STARTED, ServerToolStatus.IN_PROGRESS),
            (ServerToolPhase.DONE, ServerToolStatus.COMPLETED),
        ]
        started, done = tool_events
        assert started.call.queries == []
        assert done.call.queries == ["weather in Paris"]
        assert [raw["type"] for raw in done.call.raw] == [
            "server_tool_use",
            "web_search_tool_result",
        ]
        completed = received[-1]
        assert isinstance(completed, CompletedEvent)
        assert completed.text == "Hi! My name is Claude."
        [citation] = completed.citations
        assert (citation.start_index, citation.end_index) == (0, 22)
        assert completed.native is not None
        assert (
            completed.native.items[2]["citations"]
            == (CITING_TEXT["citations"])
        )
        assert completed.native.items[0]["input"] == {
            "query": "weather in Paris"
        }

    async def test_the_pieces_of_a_tools_input_are_other_events(
        self,
    ) -> None:
        received = await convert(anthropic_searched_stream())

        deltas = [
            e
            for e in received
            if isinstance(e, OtherEvent)
            and e.raw.type == "content_block_delta"
            and e.raw.delta.type in {"input_json_delta", "citations_delta"}
        ]
        assert len(deltas) == 3

    async def test_a_calls_empty_input_is_an_empty_object(self) -> None:
        call = code_call("code_execution", {}, None)[0]
        blocks = block_events(
            0,
            call,
            {"type": "input_json_delta", "partial_json": ""},
        )

        received = await convert(stream_of(blocks))

        completed = received[-1]
        assert isinstance(completed, CompletedEvent)
        assert completed.native is not None
        assert completed.native.items[0]["input"] == {}
        [server_call] = completed.server_tool_calls
        assert server_call.status is ServerToolStatus.INCOMPLETE

    async def test_a_call_without_pieces_keeps_its_input(self) -> None:
        call = code_call("code_execution", {"code": "print(1)"}, None)[0]

        received = await convert(stream_of(block_events(0, call)))

        completed = received[-1]
        assert isinstance(completed, CompletedEvent)
        assert completed.server_tool_calls[0].code == "print(1)"

    @pytest.mark.parametrize("pieces", [["[1,", "2]"], ["{", ""]])
    async def test_an_input_that_is_no_object_raises(
        self, pieces: list[str]
    ) -> None:
        call = code_call("code_execution", {}, None)[0]
        blocks = block_events(
            0,
            call,
            *({"type": "input_json_delta", "partial_json": p} for p in pieces),
        )

        with pytest.raises(APIResponseError):
            await convert(stream_of(blocks))

    async def test_a_result_of_no_known_call_is_an_other_event(self) -> None:
        result = SEARCH_RESULT | {"tool_use_id": "nobody"}

        received = await convert(stream_of(block_events(0, result)))

        assert not any(isinstance(e, ServerToolCallEvent) for e in received)
        completed = received[-1]
        assert isinstance(completed, CompletedEvent)
        assert completed.server_tool_calls == []

    async def test_a_pause_ends_the_stream_as_paused(self) -> None:
        blocks = block_events(0, SEARCH_CALL | {"input": {}})

        received = await convert(stream_of(blocks, stop_reason="pause_turn"))

        completed = received[-1]
        assert isinstance(completed, CompletedEvent)
        assert completed.incomplete_reason is IncompleteReason.PAUSED
        assert completed.text is None
        assert completed.native is not None

    async def test_a_citation_opens_a_text_block_that_did_not_start(
        self,
    ) -> None:
        events = stream_of(
            [
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "citations_delta",
                        "citation": CITING_TEXT["citations"][0],
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Cited."},
                },
            ]
        )

        completed = (await convert(events))[-1]

        assert isinstance(completed, CompletedEvent)
        assert completed.native is not None
        assert (
            completed.native.items[0]["citations"]
            == (CITING_TEXT["citations"])
        )
        assert completed.citations[0].end_index == len("Cited.")


class TestStreamState:
    def test_a_piece_of_a_tools_input_has_no_call_id(self) -> None:
        state = StreamState()
        state.start_block(
            0, message(content=[SEARCH_CALL | {"input": {}}]).content[0]
        )

        assert state.add_arguments(0, "{}") is None
        assert state.tool_calls() == []
        assert state.call(0) is None

    def test_a_piece_without_its_block_raises(self) -> None:
        with pytest.raises(APIResponseError, match="without its tool_use"):
            StreamState().add_arguments(3, "{}")

    def test_only_a_result_of_a_call_finishes_it(self) -> None:
        state = StreamState()
        state.start_block(0, message().content[0])
        state.start_block(
            1, message(content=[SEARCH_CALL | {"input": {}}]).content[0]
        )
        state.start_block(
            2,
            message(
                content=[SEARCH_RESULT | {"tool_use_id": "other"}]
            ).content[0],
        )

        assert state.server_done(0) is None
        assert state.server_done(2) is None
        assert state.server_started(1).status is ServerToolStatus.IN_PROGRESS


class TestInput:
    def test_a_turn_with_anthropics_content_goes_back_as_it_is(self) -> None:
        blocks = [SEARCH_CALL, SEARCH_RESULT, CITING_TEXT]
        turn = Turn(
            role=Role.ASSISTANT,
            content="Hi! My name is Claude.",
            native=NativeOutput(provider=PROVIDER, items=blocks),
        )

        method = adapter.to_method(
            GenerationRequest(
                model="m", input=[Turn.user("Hi"), turn, Turn.user("More")]
            )
        )

        assert method.get_body_map()["messages"] == [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": blocks},
            {"role": "user", "content": "More"},
        ]

    def test_content_of_another_provider_is_not_read(self) -> None:
        turn = Turn(
            role=Role.ASSISTANT,
            content="Hi there.",
            native=NativeOutput(
                provider=Provider("someone-else"), items=[{"x": 1}]
            ),
        )

        method = adapter.to_method(
            GenerationRequest(
                model="m", input=[Turn.user("Hi"), turn, Turn.user("More")]
            )
        )

        assert method.get_body_map()["messages"][1] == {
            "role": "assistant",
            "content": "Hi there.",
        }

    def test_a_paused_turn_may_end_the_conversation(self) -> None:
        paused = adapter.to_response(
            message(content=[SEARCH_CALL], stop_reason="pause_turn"),
            REQUEST,
        )

        method = adapter.to_method(
            GenerationRequest(
                model="m", input=[Turn.user("Hi"), paused.to_turn()]
            )
        )

        assert method.get_body_map()["messages"][-1] == {
            "role": "assistant",
            "content": [SEARCH_CALL],
        }


class TestCodeFiles:
    """The files a code execution reads, and the files it makes."""

    stored = ProviderFile(provider=PROVIDER, id="file_in")

    def test_files_go_into_the_last_user_message(self) -> None:
        paused = adapter.to_response(
            message(content=[SEARCH_CALL], stop_reason="pause_turn"),
            REQUEST,
        )
        parts = [TextPart(text="Chart this")]

        method = adapter.to_method(
            request(
                CodeExecution(files=[self.stored]),
                input=[Turn.user(parts), paused.to_turn()],
            )
        )

        assert method.get_body_map()["messages"][0] == {
            "role": "user",
            "content": [
                {"type": "text", "text": "Chart this"},
                {"type": "container_upload", "file_id": "file_in"},
            ],
        }

    def test_files_without_a_user_message_are_refused(self) -> None:
        paused = adapter.to_response(
            message(content=[SEARCH_CALL], stop_reason="pause_turn"),
            REQUEST,
        )

        with pytest.raises(RequestValidationError) as caught:
            adapter.to_method(
                request(
                    CodeExecution(files=[self.stored]),
                    input=[paused.to_turn()],
                )
            )

        assert caught.value.field == "tools"

    def test_only_outputs_with_a_file_are_files(self) -> None:
        result = {
            "type": "bash_code_execution_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": {
                "type": "bash_code_execution_result",
                "stdout": "",
                "stderr": "",
                "return_code": 0,
                "content": [
                    {"type": "bash_code_execution_output", "file_id": "f1"},
                    {"type": "something_else", "file_id": "f2"},
                ],
            },
        }

        files = block_files([result, SEARCH_RESULT])

        assert [(f.id, f.source_call_id) for f in files] == [
            ("f1", "srvtoolu_1")
        ]
        assert files[0].container_id is None

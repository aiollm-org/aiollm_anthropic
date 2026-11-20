"""Anthropic's Files methods on the wire.

The client runs against the local recorder: what goes out (path,
query, form parts, headers) and how results and errors come back.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from aiollm.http.exceptions.status import BadRequestError, NotFoundError
from aiollm.testing import TEST_TOKEN, LocalAPI, Reply
from aiollm.testing.aiohttp import Recorder
from aiollm.types.files import BytesFile

from aiollm_anthropic import API_VERSION, AnthropicClient
from aiollm_anthropic import methods as anthropic
from aiollm_anthropic.client import session_factory

CONTRACTS = Path(__file__).parent / "contracts"

PDF = b"%PDF-1.4\n%%EOF\n"
"""Bytes the upload sends; the server never looks into them."""


def documented(operation: str) -> Any:
    """Read a documented result of the Files snapshot.

    Args:
        operation: Such as ``createFile``.

    Returns:
        Its body.
    """
    path = CONTRACTS / "files" / "responses.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    (body,) = [item["body"] for item in items if item["id"] == operation]
    return body


ANTHROPIC_FILE = documented("createFile")


@pytest.fixture
def local(api: LocalAPI, monkeypatch: pytest.MonkeyPatch) -> LocalAPI:
    """Point the client at the recorder."""
    monkeypatch.setattr(session_factory, "api", api)
    return api


@pytest.mark.usefixtures("local")
async def test_upload_sends_the_version_and_parses_dates(
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(json=ANTHROPIC_FILE))

    async with AnthropicClient(TEST_TOKEN) as client:
        file = await client.send_request(
            anthropic.CreateFile(
                file=BytesFile(PDF, "document.pdf", "application/pdf"),
                expires_in_seconds=3600,
            )
        )

    assert recorder.last.raw_path == "/v1/files"
    assert recorder.last.headers["anthropic-version"] == API_VERSION
    parts = {part.name: part for part in recorder.last.parts}
    assert list(parts) == ["file", "expires_in_seconds"]
    assert parts["file"].content_type == "application/pdf"
    assert parts["expires_in_seconds"].data == b"3600"
    assert file.created_at == datetime(
        2025, 4, 15, 18, 37, 24, 100435, tzinfo=UTC
    )
    assert file.downloadable is False


@pytest.mark.usefixtures("local")
async def test_list_by_ids_repeats_the_bracketed_key(
    recorder: Recorder,
) -> None:
    recorder.reply(Reply(json=documented("listFiles")))

    async with AnthropicClient(TEST_TOKEN) as client:
        page = await client.send_request(
            anthropic.ListFiles(ids=["file_1", "file_2"])
        )

    assert recorder.last.query.getall("ids[]") == ["file_1", "file_2"]
    assert recorder.last.headers["anthropic-version"] == API_VERSION
    assert page.next_page is None or isinstance(page.next_page, str)


@pytest.mark.usefixtures("local")
async def test_a_file_that_cannot_be_downloaded_names_its_code(
    recorder: Recorder,
) -> None:
    # The live answer (2026-11-19) to downloading an uploaded file.
    recorder.reply(
        Reply(
            status=400,
            json={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "File cannot be downloaded",
                    "details": {"error_code": "file_not_downloadable"},
                },
                "request_id": "req_1",
            },
        )
    )

    async with AnthropicClient(TEST_TOKEN) as client:
        method = anthropic.DownloadFile(file_id="file_1")
        with pytest.raises(BadRequestError) as caught:
            async with client.stream(method):
                pass  # pragma: no cover

    assert caught.value.code == "file_not_downloadable"
    assert caught.value.transient is False
    assert len(recorder.requests) == 1


@pytest.mark.usefixtures("local")
async def test_an_unknown_file_is_not_found(recorder: Recorder) -> None:
    recorder.reply(
        Reply(
            status=404,
            json={
                "type": "error",
                "error": {
                    "type": "not_found_error",
                    "message": "File not found",
                },
            },
        )
    )

    async with AnthropicClient(TEST_TOKEN) as client:
        with pytest.raises(NotFoundError):
            await client.send_request(anthropic.RetrieveFile(file_id="nope"))

    assert recorder.last.raw_path == "/v1/files/nope"

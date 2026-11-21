"""Shared files behave for Anthropic as for every provider.

The scenarios of ``aiollm.testing.contract.files`` run against
``AnthropicClient.files`` on the local server, with the documented results
of Anthropic's Files API; only the wire format is the provider's own, the
shared results and errors must be the same.
"""

from __future__ import annotations

from datetime import UTC, datetime

from aiollm.testing.contract.files import FilesCase, FilesContract

from aiollm_anthropic import AnthropicClient
from aiollm_anthropic.api import PROVIDER as ANTHROPIC_PROVIDER
from aiollm_anthropic.client import (
    session_factory as anthropic_factory,
)
from aiollm_anthropic.types import FileMetadata
from tests import test_messages as anthropic
from tests.test_files import documented

ANTHROPIC_FILE = documented("createFile")

ANTHROPIC = FilesCase(
    provider=ANTHROPIC_PROVIDER,
    client=AnthropicClient,
    factory=anthropic_factory,
    raw_type=FileMetadata,
    file=ANTHROPIC_FILE,
    size=102400,
    created_at=datetime(2025, 4, 15, 18, 37, 24, 100435, tzinfo=UTC),
    expires_at=datetime(2025, 5, 15, 18, 37, 24, 100435, tzinfo=UTC),
    deleted=documented("deleteFile"),
    page={"data": [ANTHROPIC_FILE], "next_page": "page_2"},
    cursor=("page", "page_2"),
    last_page={
        "data": [ANTHROPIC_FILE | {"id": "file_older"}],
        "next_page": None,
    },
    upload_parts={"expires_in_seconds": b"3600"},
    bare_upload_parts=["file"],
    options=None,
    message_path="/v1/messages",
    message=anthropic.MESSAGE,
    input_field="messages",
    stored_parts=[
        {
            "type": "document",
            "source": {"type": "file", "file_id": ANTHROPIC_FILE["id"]},
        },
        {
            "type": "image",
            "source": {"type": "file", "file_id": ANTHROPIC_FILE["id"]},
        },
    ],
    container_download=None,
)


class TestAnthropicFiles(FilesContract):
    case = ANTHROPIC

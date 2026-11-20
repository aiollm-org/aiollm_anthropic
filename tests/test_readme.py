"""The example of README.md runs as it is written."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from aiollm.testing import TEST_TOKEN, LocalAPI, Reply
from aiollm.testing.aiohttp import Recorder

from aiollm_anthropic.adapters import message_text
from aiollm_anthropic.client import session_factory
from aiollm_anthropic.types import Message
from tests.test_messages import MESSAGE

README = Path(__file__).resolve().parents[1] / "README.md"


def example(title: str) -> str:
    """The Python block of README.md with this title."""
    blocks = re.findall(
        r'^```python title="([^"]+)"\n(.*?)^```$',
        README.read_text(encoding="utf-8"),
        flags=re.MULTILINE | re.DOTALL,
    )
    found = [code for name, code in blocks if name == title]
    assert len(found) == 1, title
    return found[0]


def test_every_example_is_tested() -> None:
    titles = re.findall(
        r'^```python title="([^"]+)"', README.read_text("utf-8"), re.M
    )

    assert titles == ["quick_start.py"]


async def test_quick_start(
    recorder: Recorder,
    api: LocalAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session_factory, "api", api)
    monkeypatch.setenv("ANTHROPIC_API_KEY", TEST_TOKEN)
    recorder.reply(Reply(json=MESSAGE))
    namespace: dict[str, Any] = {}
    exec(
        compile(example("quick_start.py"), "quick_start.py", "exec"), namespace
    )  # noqa: S102

    text = await namespace["ask"]()

    assert text == message_text(Message.model_validate(MESSAGE))
    (request,) = recorder.requests
    assert request.raw_path == "/v1/messages"
    assert request.headers["Authorization"] == f"Bearer {TEST_TOKEN}"

"""Shared fixtures: the ``aiollm.testing`` plugin, set up for this suite."""

from __future__ import annotations

import pytest

pytest_plugins = ["aiollm.testing.plugin"]


@pytest.fixture
def provider_key_envs() -> tuple[str, ...]:
    """The key the Anthropic client reads, cleared in every test."""
    return ("ANTHROPIC_API_KEY",)

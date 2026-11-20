"""What every provider client does the same way, run for Anthropic.

The scenarios of ``aiollm.testing.contract.clients`` run against
``AnthropicClient``; what is Anthropic's own (its error resolver,
``default_max_tokens``) stays in ``test_client.py``.
"""

from __future__ import annotations

from aiollm.http.auth import APIKeyAuth, BearerAuth
from aiollm.providers.anthropic import (
    ANTHROPIC_API,
    AnthropicClient,
    AnthropicErrorResolver,
)
from aiollm.providers.anthropic.api import PROVIDER
from aiollm.testing.contract.clients import ClientCase, ClientContract

ANTHROPIC = ClientCase(
    client=AnthropicClient,
    provider=PROVIDER,
    api=ANTHROPIC_API,
    url=("/messages", "https://api.anthropic.com/v1/messages"),
    resolver=AnthropicErrorResolver,
    env="ANTHROPIC_API_KEY",
    auths=(BearerAuth("k"), APIKeyAuth("k")),
)


class TestAnthropicClient(ClientContract):
    case = ANTHROPIC

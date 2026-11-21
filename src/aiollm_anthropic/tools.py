"""Anthropic's built-in tools, as the shared request takes them.

``WebSearch`` and ``CodeExecution`` of the shared layer send the first
version of each tool and the options both providers have. These
functions are for the rest: a newer version of a tool, Anthropic's own
options (``blocked_domains``, ``max_uses``), and the web fetch tool.
Each returns a ``ProviderTool`` of this provider, built from the
generated request parameters of the version, so a field the version
does not have, or a value it refuses, raises ``ValueError`` here and not
at the request.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal

from pydantic import MISSING, BaseModel

from aiollm.types.requests.builtin import ProviderTool

from aiollm_anthropic.api import PROVIDER
from aiollm_anthropic.methods.params import UserLocationParam
from aiollm_anthropic.methods.params.tools import (
    CodeExecutionTool20250522Param,
    CodeExecutionTool20250825Param,
    CodeExecutionTool20260120Param,
    CodeExecutionTool20260521Param,
    WebFetchTool20250910Param,
    WebFetchTool20260209Param,
    WebFetchTool20260309Param,
    WebFetchTool20260318Param,
    WebSearchTool20250305Param,
    WebSearchTool20260209Param,
    WebSearchTool20260318Param,
)

type WebSearchVersion = Literal["20250305", "20260209", "20260318"]
"""The dated versions of the web search tool."""

type WebFetchVersion = Literal["20250910", "20260209", "20260309", "20260318"]
"""The dated versions of the web fetch tool."""

type CodeExecutionVersion = Literal[
    "20250522", "20250825", "20260120", "20260521"
]
"""The dated versions of the code execution tool."""

WEB_SEARCH: Final[dict[str, type[BaseModel]]] = {
    "20250305": WebSearchTool20250305Param,
    "20260209": WebSearchTool20260209Param,
    "20260318": WebSearchTool20260318Param,
}
"""The request parameters of each version of the web search tool."""

WEB_FETCH: Final[dict[str, type[BaseModel]]] = {
    "20250910": WebFetchTool20250910Param,
    "20260209": WebFetchTool20260209Param,
    "20260309": WebFetchTool20260309Param,
    "20260318": WebFetchTool20260318Param,
}
"""The request parameters of each version of the web fetch tool."""

CODE_EXECUTION: Final[dict[str, type[BaseModel]]] = {
    "20250522": CodeExecutionTool20250522Param,
    "20250825": CodeExecutionTool20250825Param,
    "20260120": CodeExecutionTool20260120Param,
    "20260521": CodeExecutionTool20260521Param,
}
"""The request parameters of each version of the code execution tool."""


def _tool(
    versions: Mapping[str, type[BaseModel]],
    version: str,
    fields: Mapping[str, object],
) -> ProviderTool:
    """Build a tool of this provider from the parameters of a version.

    Args:
        versions: The parameters of each version of the tool.
        version: The version asked for.
        fields: The options given.

    Returns:
        The tool, its definition the parameters as Anthropic takes them.

    Raises:
        ValueError: The version is none of the tool's, or an option is
            not one of that version or has a value the API refuses.
    """
    if version not in versions:
        message = f"no version {version!r}; use one of {', '.join(versions)}"
        raise ValueError(message)
    parameters = versions[version](**fields)
    return ProviderTool(
        provider=PROVIDER, definition=parameters.model_dump(mode="json")
    )


def _check_domains(
    allowed: list[str] | MISSING, blocked: list[str] | MISSING
) -> None:
    """Refuse both lists of domains, which Anthropic takes only apart.

    Args:
        allowed: The ``allowed_domains`` given.
        blocked: The ``blocked_domains`` given.

    Raises:
        ValueError: Both are given.
    """
    if allowed is not MISSING and blocked is not MISSING:
        message = (
            "allowed_domains and blocked_domains cannot be given together"
        )
        raise ValueError(message)


def web_search(
    *,
    version: WebSearchVersion = "20250305",
    allowed_domains: list[str] | MISSING = MISSING,
    blocked_domains: list[str] | MISSING = MISSING,
    max_uses: int | MISSING = MISSING,
    user_location: UserLocationParam | MISSING = MISSING,
) -> ProviderTool:
    """Build Anthropic's web search tool, with its own options.

    The newer versions filter the results with code execution, which
    the request must have as a tool too.

    Args:
        version: The dated version of the tool; the shared
            ``WebSearch`` sends ``20250305``.
        allowed_domains: Search only these domains.
        blocked_domains: Never search these; not together with
            ``allowed_domains``.
        max_uses: How many searches one request may make.
        user_location: Where the user is, to localize the results; the
            shared ``UserLocation`` is for the shared ``WebSearch``.

    Returns:
        The tool, for ``tools=`` of a request.

    Raises:
        ValueError: The version is none of the tool's, does not have an
            option, an option has a value the API refuses, or both
            lists of domains are given.

    Example:
        >>> web_search(blocked_domains=["example.com"], max_uses=3)
    """
    _check_domains(allowed_domains, blocked_domains)
    fields: dict[str, object] = {}
    if allowed_domains is not MISSING:
        fields["allowed_domains"] = allowed_domains
    if blocked_domains is not MISSING:
        fields["blocked_domains"] = blocked_domains
    if max_uses is not MISSING:
        fields["max_uses"] = max_uses
    if user_location is not MISSING:
        fields["user_location"] = user_location
    return _tool(WEB_SEARCH, version, fields)


def web_fetch(
    *,
    version: WebFetchVersion = "20250910",
    allowed_domains: list[str] | MISSING = MISSING,
    blocked_domains: list[str] | MISSING = MISSING,
    max_uses: int | MISSING = MISSING,
    max_content_tokens: int | MISSING = MISSING,
    citations: bool | MISSING = MISSING,
) -> ProviderTool:
    """Build Anthropic's web fetch tool: the model reads a page itself.

    Only Anthropic has it, so it has no shared counterpart.

    Args:
        version: The dated version of the tool.
        allowed_domains: Fetch only from these domains.
        blocked_domains: Never fetch from these.
        max_uses: How many fetches one request may make.
        max_content_tokens: The most tokens of a page that count.
        citations: Whether the answer cites the pages it fetched.

    Returns:
        The tool, for ``tools=`` of a request.

    Raises:
        ValueError: The version is none of the tool's, does not have an
            option, an option has a value the API refuses, or both
            lists of domains are given.

    Example:
        >>> web_fetch(max_uses=2, citations=True)
    """
    _check_domains(allowed_domains, blocked_domains)
    fields: dict[str, object] = {}
    if allowed_domains is not MISSING:
        fields["allowed_domains"] = allowed_domains
    if blocked_domains is not MISSING:
        fields["blocked_domains"] = blocked_domains
    if max_uses is not MISSING:
        fields["max_uses"] = max_uses
    if max_content_tokens is not MISSING:
        fields["max_content_tokens"] = max_content_tokens
    if citations is not MISSING:
        fields["citations"] = {"enabled": citations}
    return _tool(WEB_FETCH, version, fields)


def code_execution(
    *, version: CodeExecutionVersion = "20250825"
) -> ProviderTool:
    """Build Anthropic's code execution tool of a given version.

    Args:
        version: The dated version of the tool; the shared
            ``CodeExecution`` sends ``20250825``.

    Returns:
        The tool, for ``tools=`` of a request.

    Raises:
        ValueError: The version is none of the tool's.

    Example:
        >>> code_execution(version="20260120")
    """
    return _tool(CODE_EXECUTION, version, {})


__all__ = ["code_execution", "web_fetch", "web_search"]

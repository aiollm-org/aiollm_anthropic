# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Up to 0.14.0 the package was `aiollm.providers.anthropic`, part of aiollm: its
earlier changes are in [aiollm's changelog](https://github.com/aiollm-org/aiollm/blob/master/CHANGELOG.md).

## [Unreleased]

<!-- towncrier release notes start -->

## [0.16.0] - 2026-11-21

### Added

- Built-in tools of the shared request: `WebSearch` (with `allowed_domains` and
  `user_location`) and `CodeExecution` go in `tools=` beside the functions and
  reach Anthropic as `web_search_20250305` and `code_execution_20250825`; a
  `ProviderTool` of this provider is sent as its own definition, and a tool
  Anthropic does not run raises `RequestValidationError`. The new
  `aiollm_anthropic.tools` module builds those provider tools for the rest:
  `web_search`, `web_fetch` and `code_execution`, with the dated version and
  Anthropic's own options (`blocked_domains`, `max_uses`, `max_content_tokens`,
  `citations`). The calls the provider ran come back as `server_tool_calls`
  (`server_tool_use` blocks with their results: `COMPLETED`, `FAILED` with the
  `error_code`, `INCOMPLETE` without a result) and the sources of the text as
  `citations` (a page of a search is a `URL`, a part of a stored file a
  `FILE`), in a response and in the `CompletedEvent` of a stream, where each
  call is also a `ServerToolCallEvent` when it starts and when its result
  comes. An `assistant` turn of `to_turn()` with Anthropic's own content goes
  back as that content, so the encrypted search results reach the next request.

### Changed

- A message that stopped with `pause_turn` is now `IncompleteReason.PAUSED` (it
  was `OTHER`; `raw_incomplete_reason` is still `pause_turn`). Send `to_turn()`
  back to let Anthropic go on with the turn.
- Requires `aiollm-core` 0.16 (`>=0.16,<0.17`), the release with the shared
  built-in tool types.

### Fixed

- A stream no longer raises `APIResponseError` when the input of a built-in
  tool (`server_tool_use`) arrives in `input_json_delta` pieces: the pieces are
  joined into the block's `input`, and the call is complete in `native`,
  `server_tool_calls` and its `ServerToolCallEvent`.

## [0.15.0] - 2026-11-20

### Changed

- **Breaking:** the Anthropic provider is its own package, `aiollm-anthropic`,
  imported as `aiollm_anthropic`: `aiollm.providers.anthropic` becomes
  `aiollm_anthropic`, and each of its modules keeps its name
  (`aiollm.providers.anthropic.types` becomes `aiollm_anthropic.types`, and so
  on). `aiollm_anthropic.__version__` is its own version.
- The core this package depends on is now the `aiollm-core` distribution
  (it was `aiollm`), from PyPI: `aiollm-core>=0.15,<0.16`. The module is
  still `aiollm`.

[Unreleased]: https://github.com/aiollm-org/aiollm_anthropic/compare/v0.16.0...HEAD
[0.16.0]: https://github.com/aiollm-org/aiollm_anthropic/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/aiollm-org/aiollm_anthropic/releases/tag/v0.15.0

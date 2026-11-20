# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`aiollm-anthropic` (import `aiollm_anthropic`) is aiollm's Anthropic provider: `AnthropicClient`, Anthropic's Messages and Files APIs, and the adapters that give it aiollm's provider-neutral `client.generation` and `client.files`. It was `aiollm.providers.anthropic` up to aiollm 0.14.0; its history came along (`git log --follow`). Pre-alpha; the public surface is unstable.

The core is [aiollm-org/aiollm](https://github.com/aiollm-org/aiollm), pinned to a commit in `pyproject.toml` until it is on PyPI as `aiollm-core`. **The core's `CLAUDE.md` holds the rules this package follows too**: docstrings, logging, naming, `MISSING` and `None` in requests, the request and stream flows, errors, the shared layer. This file adds only what is this package's own. The code says what the package does; an agreed decision says what it should do; a text that lags behind the code is fixed. `CONTRIBUTING.md` is the guide for contributors; keep it in step with this file. The user documentation is the core's site (<https://aiollm-org.github.io/aiollm/>); a docstring here is user documentation too.

## Commands

The project is managed with `uv` (see `uv.lock`) and requires Python 3.13 or newer (`.python-version` pins 3.13). The version is read by hatchling from `src/aiollm_anthropic/__meta__.py`.

```bash
uv sync --all-groups                 # install runtime + dev + test + codegen deps
uv run ruff check src tests          # lint (line length 79)
uv run ruff format src tests         # format
uv run mypy src                      # strict type checking (pydantic.mypy plugin)
uv run pyright                       # strict pyright over src and tests
uv run pytest                        # tests (asyncio_mode = auto)
uv run pytest --cov                  # with coverage (branch, fail_under = 98; kept at 100 %)
uv run pytest --mypy -m mypy tests/typing   # static checks of the public API
uv run rumdl check README.md CHANGELOG.md CONTRIBUTING.md CLAUDE.md .claude
uv run python scripts/generate_types.py     # the types and request params (pinned generator and snapshot)
uv run python scripts/refresh_upstream.py   # move the SDK pin to PyPI's newest anthropic, rebuild snapshot and types
uv run --group upstream python -m tests.contracts.update   # rebuild the contract snapshot from the pinned SDK
uv run towncrier create +slug.fixed.md      # a changelog fragment in CHANGES/
uv pip install -e ../aiollm          # try a change of the core from a sibling checkout (until the next uv sync)
```

CI (`.github/workflows/tests.yml`) runs the checks, the Markdown lint and a regeneration of the types that must change no file (job `lint and types`), pytest on Ubuntu and Windows × every supported CPython (3.13, 3.14, 3.15; the `tests` job sums up the matrix for the branch ruleset), the wheel installed into a clean environment with the core it pins and imported (`package`, which also runs `check_extension_api`), `uv audit` (`audit`) and, on pull requests, `towncrier check` (`changelog`; the `no changelog` label skips it). `core-master.yml` runs the tests every Monday (and by hand) against the core's `master` instead of the pin, so a change of the core that breaks this package shows up before a core release. `upstream.yml` runs `scripts/refresh_upstream.py` every Monday at 06:00 UTC (and by hand): a newer `anthropic` on PyPI becomes one pull request on `upstream/refresh`, opened with the organization secret `UPSTREAM_PR_TOKEN` so CI runs on it; once a commit was pushed to that open pull request by hand, the branch is no longer rebuilt and newer sources only get a comment there. A changed page of the Files reference opens the issue "Anthropic Files reference changed", closed by updating the transcription and running the script with `--update-pages`. `publish.yml` publishes a tag `v<version>` to PyPI by trusted publishing; PyPI refuses the git pin, so the first release waits for `aiollm-core` on PyPI.

## Layout

- `src/aiollm_anthropic/`: `api.py` (`ANTHROPIC_API`, `PROVIDER: Final = Provider("anthropic")`, `API_VERSION`, the `anthropic-version` header), `resolver.py` (`AnthropicErrorResolver`), `client.py` (`AnthropicClient`, `session_factory`), `methods/` (hand-written `messages.py` and `files.py` over the generated `methods/params/`), `types/` (generated but `options.py`), `adapters/` (hand-written: `conversions.py`, `generation.py`, `files.py`). The package imports from the core only the names of its extension API (`aiollm/testing/extension_api.json`); `tests/test_extension_api.py` runs `check_extension_api("aiollm_anthropic")`. A name missing there is added in the core first (a core pull request), never imported around it.
- `tests/`: the provider's tests, `contracts/` (the wire-contract snapshot: `messages/` and `files/`, `manifest.json`, `update.py`, and `sources/`: the transcription of the Files API reference and its page hashes), `typing/check_public_api.py` (never run: every `# type: ignore[...]` there marks a line that must stay a type error) and `public_api.json` (every package's `__all__`; a public name added, renamed or removed is a decision, updated in the same change with a fragment). `conftest.py` loads the core's test kit (`aiollm.testing.plugin`) and names `ANTHROPIC_API_KEY` as the key to clear in every test. `test_readme.py` runs the README's example.
- `scripts/`: `generate_types.py` and `refresh_upstream.py`.

## Tests

- The shared scenarios are the core's (`aiollm.testing.contract`): `test_generation_contract.py`, `test_files_contract.py` and `test_client_contract.py` subclass `GenerationContract`, `FilesContract` and `ClientContract` with this provider's cases. What is Anthropic's own is in `test_client.py`, `test_generation.py`, `test_messages.py`, `test_files.py` and the rest.
- The local server (`aiollm.testing.aiohttp.Recorder`) answers only what a test scripted with `recorder.reply(...)`; any other request gets a 500 and fails the test at teardown.
- `test_wire_contracts.py` checks the endpoints against the snapshot (request schemas closed, so unknown fields fail), with the core's `aiollm.testing.wire`. Every model sets `__upstream_schema__`, the upstream `$def` it mirrors, and each field is checked against it, request parameters both ways. The examples (`responses.json`, `events.json`) are added by hand from Anthropic's documentation or a live run; a refresh never changes them and lists the ones the new schemas refuse under `rejected` in `manifest.json`.

## Schema sources and generated code

- The sources are Anthropic's own, never a third-party dump. Anthropic publishes no OpenAPI file, so the Messages snapshot is derived by `tests/contracts/update.py` from the official Python SDK `anthropic`, pinned exactly in the `upstream` dependency group: the request from the body `Messages.create` sends, the response and stream from `Message` and `RawMessageStreamEvent`, with pydantic's `TypeAdapter`. The SDK writes an optional field as `Optional[X] = None`, so nullability is checked against the live API reference (<https://platform.claude.com/docs/en/api>). The Files API is `tests/contracts/sources/anthropic-files.openapi.json`, a hand transcription of that reference with its URLs and retrieval date; `anthropic-files.pages.json` holds the hashes of its pages.
- `test_codegen_pin.py` checks that the SDK pin, the manifest and the script's `ANTHROPIC_VERSION` agree: a new SDK version changes the pin, the snapshot and the generated types in one change.
- The response and event types (`types/<family>.py`) and the request parameters (`methods/params/`) are generated by aiollm-codegen from the Messages snapshot with the Files transcription merged (`--merge`), and never edited by hand; a defect is fixed in aiollm-codegen and the pin moves. aiollm-codegen names an API after the snapshot's `<provider>/<endpoint>` folders, so `scripts/generate_types.py` hands it a copy under `anthropic/messages`. `types/options.py`, the method classes and the adapters stay hand-written. `methods/params/` is deeper than the core's depth rule allows, decided with the maintainer (2026-11-06).
- Ruff gives the core (`aiollm`) an import section of its own between third-party and first-party, so the generated modules keep the layout aiollm-codegen writes.
- `scripts/refresh_upstream.py` compares the pin with the newest `anthropic` on PyPI; behind, it moves the pin, rebuilds the snapshot and the types, and writes the pull request's description with the public names they add or remove against `tests/public_api.json` (updated by hand). It also compares the Files reference pages with their hashes; a changed page is closed by updating the transcription and running the script with `--update-pages`.

## Conventions

- The core's conventions apply (its `CLAUDE.md`, "Conventions"); a conflict is settled with the maintainer.
- A change of the core and of this package at once: the core's pull request first; this package's pull request pins the commit of that branch, then the commit of its merge.
- A change pull request adds a fragment to `CHANGES/` and never bumps the version or edits `CHANGELOG.md`; a release is a separate pull request (`ai-api-release`). Tags are set by the maintainer only.
- A provider type uses `Literal` for a schema constant that names the object (`type`, `object`, `role: "assistant"`) and `str` for an enumeration that grows (`status`, error codes).
- Provider-only request fields reach `client.generation` through `provider_options=`: `AnthropicGenerationOptions` (`types/options.py`), whose keys are method fields of the same type and never a field the shared request fills (`test_options.py`); `output_config` is filled by the shared `output_schema` and `effort`, and an option may fill the rest of it.

## Skills

`.claude/skills/`: `ai-api-endpoint` (a new endpoint or field, from the upstream schema to the wire contracts) and `ai-api-release` (fragments and releases). They follow this file; when a skill and this file disagree, this file wins and the skill is fixed.

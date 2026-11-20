---
name: ai-api-endpoint
description: Add or change an endpoint of aiollm-anthropic - read Anthropic's official SDK types at the pinned version and its live API reference, rebuild the snapshot, generate types and request parameters with scripts/generate_types.py (aiollm-codegen), hand-write the method classes, the wire-contract Case, adapters and options, docs and tests. Use it in the aiollm_anthropic repository whenever the owner asks for a new Anthropic endpoint or resource, a new request field on an existing method, or to follow a new SDK version or a changed Files reference page.
---

# aiollm-anthropic endpoint

Every type and method of this package mirrors Anthropic's own schema, and
tests hold it there. An endpoint is done when the snapshot, the types, the
method fields and the adapters all agree with upstream - not when a request
happens to work.

## 1. Read upstream, not memory

Read Anthropic's Python SDK at the version the `upstream` dependency group
pins (the resource method, its body and its `cast_to` type) and its live API
reference (<https://platform.claude.com/docs/en/api>). Note the request and
response objects, which fields are required, which are nullable, and the
stream events. The SDK writes an optional field as `Optional[X] = None`, so
check nullability against the API reference. If the sources disagree or lack
something, say so and ask; do not guess a schema from an example.

Decide with the owner when the endpoint touches the shared layer: a shared
type of the core holds only what every provider has, and a change there is a
core pull request first (CONTRIBUTING.md, "A change in the core and here");
the rest stays in this package's types and is reached through `raw`.

## 2. Snapshot and contract tests

- `tests/contracts/update.py`: add an `Endpoint(name, source)` with
  `source=SDKOperation(...)` naming the SDK resource method and its response
  and stream types, or an operation of the `OpenAPIEndpoints` group read from
  `tests/contracts/sources/anthropic-files.openapi.json` (the hand
  transcription of the reference, with its URLs and retrieval date; update
  `anthropic-files.pages.json` with `scripts/refresh_upstream.py
  --update-pages`). Put the documented or recorded examples in
  `responses.json` and `events.json`, then refresh with
  `uv run --group upstream python -m tests.contracts.update`. Review the
  snapshot diff and `manifest.json` (its `rejected` examples).
- `tests/test_wire_contracts.py`: add to the `Case` (methods, a request that
  sets every field the adapter and the options fill, response model, event
  union, examples). The tests then check the request against the closed
  schema, every method field, and every model field against its `$def`,
  request parameters both ways.

## 3. Types and request parameters

They are generated, never written by hand: `scripts/generate_types.py` runs
the pinned aiollm-codegen (the `codegen` dependency group; `--codegen DIR`
for a local checkout) on the Messages snapshot with the Files transcription
merged, and replaces `types/<family>.py` and `methods/params/`. A defect in
them is fixed in aiollm-codegen, then the pin in `pyproject.toml` moves to
that commit (`uv lock`) and they are regenerated; CI fails when the committed
files differ from a regeneration. A new SDK version changes the `upstream`
pin, the snapshot, `ANTHROPIC_VERSION` where it moved and the generated types
in one change (`tests/test_codegen_pin.py`). `types/options.py` stays
hand-written: its `TypedDict`s mirror the method fields
(`tests/test_options.py`).

## 4. Methods

In `src/aiollm_anthropic/methods/<endpoint>.py`, hand-written over the
generated parameter mixins: `__request_path__` and the other dunders,
`VERSION_HEADERS`, `__retryable__`, `__stream_terminal__`, error events and
parsers as the other methods do, and the limits the schema leaves out
(`MessageLimits`) as narrowed fields or validators, so the generated
parameters are never edited.

## 5. Adapters

Adapters are pure converters (no I/O) with `provider = PROVIDER`; they pass
`MISSING` through unchanged and test it with `is MISSING`. A request
Anthropic cannot take raises `RequestValidationError` before sending. A
shared field that changes gets its row in the core's `docs/unification.md`,
in the core's pull request.

## 6. Tests, docs, release

Tests in `tests/` (and the shared scenarios through the core's
`aiollm.testing.contract` cases), public names re-exported with a sorted
`__all__` and `tests/public_api.json` updated, `tests/test_extension_api.py`
green, the user guide in the core's `docs/guide/providers.md` (a core pull
request), then `ai-api-release` for the fragment. Report in Russian.

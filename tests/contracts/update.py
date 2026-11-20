"""Rebuild Anthropic's wire-contract snapshots from its schema sources.

Every endpoint has one pinned upstream source. Anthropic publishes no
OpenAPI file, so the Messages API is derived from the types of its
official Python SDK, pinned in the ``upstream`` dependency group, with
pydantic's ``TypeAdapter``. The Files API is
``sources/anthropic-files.openapi.json``, a transcription of its API
reference.

The examples (``responses.json``, ``events.json``) are kept in the
repository: documented and recorded payloads, added by hand. A refresh
never changes them: it checks each against the new schemas and lists one
that fails under ``rejected`` in ``manifest.json`` with the first error,
and ``test_wire_contracts.py`` fails until it is fixed or removed by
hand.

Usage (the ``upstream`` group pins the SDK)::

    uv run --group upstream python -m \
        tests.providers.anthropic.contracts.update
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import sys
import textwrap
import typing
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import UnionType
from typing import Any, Final, NotRequired, Required, TypedDict

from jsonschema import Draft202012Validator
from pydantic import TypeAdapter

CONTRACTS: Final = Path(__file__).parent
"""Folder of the snapshots."""

MANIFEST: Final = CONTRACTS / "manifest.json"
"""Provenance of every snapshot file."""


@dataclass(frozen=True)
class SDKOperation:
    """Where an endpoint's types are in the provider's official SDK.

    Each name is ``module:attribute``, imported from the SDK the
    ``upstream`` dependency group pins.

    Attributes:
        resource: The SDK class whose method sends the request.
        method: That method's name.
        response: The type of the JSON response.
        stream: The type of a stream event.
    """

    resource: str
    method: str
    response: str
    stream: str


@dataclass(frozen=True)
class Endpoint:
    """One endpoint whose contract is snapshotted.

    Attributes:
        name: Folder name of the snapshot.
        source: Where its types are in the SDK.
    """

    name: str
    source: SDKOperation


@dataclass(frozen=True)
class OpenAPIEndpoints:
    """A group of endpoints snapshotted from one OpenAPI file alone.

    Attributes:
        name: Folder name of the snapshot.
        operations: The ``operationId`` of every endpoint, in order.
        file: The file, relative to this folder.
    """

    name: str
    operations: tuple[str, ...]
    file: str


ENDPOINTS: Final = (
    Endpoint(
        name="messages",
        source=SDKOperation(
            resource="anthropic.resources.messages.messages:Messages",
            method="create",
            response="anthropic.types:Message",
            stream="anthropic.types:RawMessageStreamEvent",
        ),
    ),
)
"""Endpoints with a stream that the library implements."""

OPENAPI_ENDPOINTS: Final = (
    OpenAPIEndpoints(
        name="files",
        operations=(
            "createFile",
            "listFiles",
            "retrieveFile",
            "deleteFile",
            "downloadFile",
        ),
        file="sources/anthropic-files.openapi.json",
    ),
)
"""Endpoint groups taken from an OpenAPI file alone.

Anthropic publishes no OpenAPI file, so this one is a transcription of
the API reference (``sources/``, with the pages and the date in its
``info``).
"""

METHODS: Final = ("get", "post", "put", "patch", "delete")
"""HTTP methods of an OpenAPI path item."""


def type_tags(schema: Any) -> set[str]:
    """Collect every ``type`` value a schema declares.

    Args:
        schema: A JSON schema, or any node of one.

    Returns:
        The ``const`` and ``enum`` values of every ``type`` property.
    """
    tags: set[str] = set()
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            kind = properties.get("type")
            if isinstance(kind, dict):
                if isinstance(kind.get("const"), str):
                    tags.add(kind["const"])
                tags.update(
                    value
                    for value in kind.get("enum", [])
                    if isinstance(value, str)
                )
        for value in schema.values():
            tags |= type_tags(value)
    elif isinstance(schema, list):
        for value in schema:
            tags |= type_tags(value)
    return tags


def event_tags(schema: dict[str, Any]) -> set[str]:
    """Collect the ``type`` of every top-level event of a stream schema.

    Args:
        schema: A stream schema whose ``anyOf`` (or ``oneOf``, as
            pydantic writes a tagged union) refers to one definition per
            event.

    Returns:
        The ``type`` values of those definitions, without the types of
        nested values.
    """
    tags: set[str] = set()
    for option in schema.get("anyOf") or schema["oneOf"]:
        name = option["$ref"].removeprefix("#/$defs/")
        kind = schema["$defs"][name]["properties"]["type"]
        tags |= type_tags({"properties": {"type": kind}})
    return tags


def first_error(validator: Draft202012Validator, value: Any) -> str | None:
    """Return the first schema error of a value.

    Args:
        validator: Validator of the upstream schema.
        value: The value to check.

    Returns:
        The error message cut to 200 characters, or ``None`` if the
        value is valid.
    """
    error = next(iter(validator.iter_errors(value)), None)
    return None if error is None else error.message[:200]


def read_json(path: Path) -> Any:
    """Read a JSON file.

    Args:
        path: The file.

    Returns:
        The decoded value.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    """Write a JSON file the way the snapshots are stored.

    Args:
        path: The file; its folder is created.
        value: The value to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=1, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


COMPONENTS: Final = "#/components/schemas/"
"""Prefix of a reference to an OpenAPI component."""

NOT_SCHEMA: Final = frozenset({"example", "examples", "deprecated"})
"""OpenAPI keys left out of a schema: documentation, not constraints."""


def component_closure(
    components: dict[str, Any],
    root: str,
) -> set[str]:
    """Collect a component and every component it references.

    Args:
        components: The OpenAPI file's ``components.schemas``.
        root: Name of the first component.

    Returns:
        The names of all of them.
    """
    found: set[str] = set()
    pending = [root]

    while pending:
        name = pending.pop()
        if name in found:
            continue
        found.add(name)
        pending.extend(references(components[name]))

    return found


def references(node: Any) -> Iterator[str]:
    """Name every component a schema node references.

    Args:
        node: A schema or any node of one.

    Yields:
        The component name of each ``$ref``.
    """
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(COMPONENTS):
            yield ref.removeprefix(COMPONENTS)
        for value in node.values():
            yield from references(value)
    elif isinstance(node, list):
        for value in node:
            yield from references(value)


def json_schema(node: Any, components: dict[str, Any]) -> Any:
    """Turn an OpenAPI schema node into plain JSON Schema.

    References point into ``$defs``; documentation keys and ``x-``
    extensions are left out; a ``oneOf`` becomes an ``anyOf``; an
    ``allOf`` of objects becomes one object
    with all their properties and required fields, so a schema can be
    closed (``additionalProperties: false``) without each part refusing
    the others' fields.

    Args:
        node: A schema node.
        components: The OpenAPI file's ``components.schemas``.

    Returns:
        The converted node.
    """
    if isinstance(node, list):
        return [json_schema(value, components) for value in node]
    if not isinstance(node, dict):
        return node

    converted: dict[str, Any] = {}
    for key, value in node.items():
        if key in NOT_SCHEMA or key.startswith("x-"):
            continue
        if key == "$ref" and isinstance(value, str):
            converted[key] = "#/$defs/" + value.removeprefix(COMPONENTS)
        elif key == "oneOf":
            # OpenAPI's oneOf names alternatives; as JSON Schema it would
            # also refuse a value two of them accept (a message without
            # its optional type is both an EasyInputMessage and an
            # InputMessage), which the API takes.
            converted["anyOf"] = json_schema(value, components)
        else:
            converted[key] = json_schema(value, components)

    if "allOf" in converted:
        merged = merge_objects(node["allOf"], components)
        if merged is not None:
            del converted["allOf"]
            converted.setdefault("type", "object")
            converted["properties"] = {
                **merged["properties"],
                **converted.get("properties", {}),
            }
            required = merged["required"] + converted.get("required", [])
            if required:
                converted["required"] = list(dict.fromkeys(required))

    return converted


def merge_objects(
    parts: list[Any],
    components: dict[str, Any],
) -> dict[str, Any] | None:
    """Merge the object schemas of an ``allOf`` into one.

    Args:
        parts: The ``allOf`` items, as the OpenAPI file has them.
        components: The OpenAPI file's ``components.schemas``.

    Returns:
        The converted properties and required fields of all parts, or
        ``None`` when a part is not a plain object schema (a union, a
        scalar), which stays an ``allOf``.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    for part in parts:
        node = part
        while isinstance(node, dict) and "$ref" in node:
            node = components[str(node["$ref"]).removeprefix(COMPONENTS)]
        if not isinstance(node, dict) or any(
            key in node for key in ("anyOf", "oneOf", "enum", "const")
        ):
            return None
        if "allOf" in node:
            inner = merge_objects(node["allOf"], components)
            if inner is None:
                return None
            properties.update(inner["properties"])
            required.extend(inner["required"])
        if node.get("type", "object") != "object":
            return None
        properties.update(json_schema(node.get("properties", {}), components))
        required.extend(node.get("required", []))

    return {"properties": properties, "required": required}


def sdk_attribute(name: str) -> Any:
    """Import an attribute of the provider's SDK.

    Args:
        name: ``module:attribute``.

    Returns:
        The attribute.
    """
    module, attribute = name.split(":")
    return getattr(importlib.import_module(module), attribute)


def without_omit(annotation: Any) -> Any:
    """Drop the SDK's "not given" markers from a parameter's type.

    Args:
        annotation: The parameter's type, such as ``int | Omit``.

    Returns:
        The type the wire takes, such as ``int``.
    """
    if typing.get_origin(annotation) not in (typing.Union, UnionType):
        return annotation
    values = tuple(
        value
        for value in typing.get_args(annotation)
        if getattr(value, "__name__", "") not in ("Omit", "NotGiven")
    )
    return values[0] if len(values) == 1 else typing.Union[values]  # noqa: UP007


def wire_body(operation: SDKOperation) -> Any:
    """Build the request body type of an SDK method.

    The method passes its body to ``self._post`` as a dictionary of its
    parameters; each key becomes a field of a ``TypedDict`` with the
    parameter's type, required where the parameter has no default.

    Args:
        operation: Where the endpoint is in the SDK.

    Returns:
        The ``TypedDict``, named ``<Class><Method>WireBody``.

    Raises:
        SystemExit: The method's body is not such a dictionary.
    """
    resource = sdk_attribute(operation.resource)
    function = inspect.unwrap(getattr(resource, operation.method))
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    body = next(
        (
            keyword.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_post"
            for keyword in node.keywords
            if keyword.arg == "body"
        ),
        None,
    )
    if not isinstance(body, ast.Dict):
        message = f"{operation.resource}.{operation.method}: no body dict"
        raise SystemExit(message)
    hints = typing.get_type_hints(function, include_extras=True)
    parameters = inspect.signature(function).parameters
    fields: dict[str, Any] = {}
    for key, value in zip(body.keys, body.values, strict=True):
        if not (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and isinstance(value, ast.Name)
        ):
            message = f"{operation.resource}.{operation.method}: wire field"
            raise SystemExit(message)
        hint = without_omit(hints[value.id])
        required = parameters[value.id].default is inspect.Parameter.empty
        fields[key.value] = Required[hint] if required else NotRequired[hint]
    name = resource.__name__ + operation.method.title() + "WireBody"
    return TypedDict(name, fields)  # type: ignore[operator]


def sdk_schemas(operation: SDKOperation) -> dict[str, Any]:
    """Derive an endpoint's schemas from the provider's SDK.

    Args:
        operation: Where the endpoint is in the SDK.

    Returns:
        The request, response and stream schemas, by name.
    """
    request = TypeAdapter(wire_body(operation)).json_schema(mode="validation")
    return {
        "request": request,
        "response": TypeAdapter(
            sdk_attribute(operation.response)
        ).json_schema(),
        "stream": TypeAdapter(sdk_attribute(operation.stream)).json_schema(),
    }


def sdk_source(operation: SDKOperation) -> dict[str, Any]:
    """Describe the SDK an endpoint's schemas come from.

    Args:
        operation: Where the endpoint is in the SDK.

    Returns:
        The package, its version, the API version it sends and the
        types the schemas are derived from.
    """
    package = operation.resource.partition(".")[0]
    constants = importlib.import_module(f"{package}._constants")
    return {
        "sdk": {
            "package": package,
            "version": importlib.import_module(package).__version__,
            "api_version": getattr(constants, "ANTHROPIC_API_VERSION", None),
        },
        "request": {
            "type": f"{operation.resource}.{operation.method}",
            "transformation": "pydantic.TypeAdapter.json_schema(validation)",
        },
        "response": {"type": operation.response},
        "stream": {"type": operation.stream},
    }


def snapshot(endpoint: Endpoint) -> dict[str, Any]:
    """Write the snapshot of one endpoint.

    The schemas are rebuilt from the endpoint's source; the examples
    already in the folder are checked against them and left as they are.

    Args:
        endpoint: The endpoint.

    Returns:
        The endpoint's manifest entry.
    """
    schemas = sdk_schemas(endpoint.source)
    sources = sdk_source(endpoint.source)
    target = CONTRACTS / endpoint.name
    for name, schema in schemas.items():
        write_json(target / f"{name}.schema.json", schema)

    validators = {
        "responses": Draft202012Validator(schemas["response"]),
        "events": Draft202012Validator(schemas["stream"]),
    }
    counts: dict[str, int] = {}
    rejected: list[dict[str, Any]] = []
    for group, validator in validators.items():
        stored = read_json(target / f"{group}.json")
        for example in stored:
            error = first_error(validator, example["body"])
            if error is not None:
                rejected.append(
                    {"id": example["id"], "group": group, "error": error}
                )
        counts[group] = len(stored)

    return {**sources, **counts, "rejected": rejected}


def class_name(operation_id: str) -> str:
    """Name an operation's parameters the way aiollm-codegen does.

    Args:
        operation_id: Such as ``listFiles``.

    Returns:
        Such as ``ListFilesParams``.
    """
    return operation_id[:1].upper() + operation_id[1:] + "Params"


def operations_of(
    spec: dict[str, Any], names: tuple[str, ...]
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Find operations of an OpenAPI file by their ids.

    Args:
        spec: The decoded file.
        names: The ``operationId`` values.

    Returns:
        Each operation and its path item, by id, in the given order;
        the operation also carries its ``x-path`` and ``x-method``.

    Raises:
        SystemExit: An operation is not in the file.
    """
    found: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for path, item in spec["paths"].items():
        for method in METHODS:
            operation = item.get(method)
            if operation and operation.get("operationId") in names:
                where = {"x-path": path, "x-method": method.upper()}
                found[operation["operationId"]] = (
                    {**operation, **where},
                    item,
                )
    if missing := [name for name in names if name not in found]:
        message = f"operations not in the file: {missing}"
        raise SystemExit(message)
    return {name: found[name] for name in names}


def operation_parameters(
    operation: dict[str, Any], item: dict[str, Any]
) -> list[dict[str, Any]]:
    """List an operation's parameters, its path item's included.

    An operation's parameter replaces the path item's one of the same
    ``name`` and ``in``, as OpenAPI defines.

    Args:
        operation: The operation.
        item: Its path item.

    Returns:
        The parameters, path item's first.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for parameter in [
        *item.get("parameters", []),
        *operation.get("parameters", []),
    ]:
        merged[parameter["name"], parameter["in"]] = parameter
    return list(merged.values())


def parameters_schema(
    spec: dict[str, Any],
    operation: dict[str, Any],
    item: dict[str, Any],
) -> tuple[dict[str, Any], set[str]]:
    """Describe an operation's request fields as one object schema.

    Path and query parameters are properties under their wire names; a
    request body's object properties join them, as they do in the
    method aiollm-codegen makes of the operation.

    Args:
        spec: The decoded OpenAPI file.
        operation: The operation.
        item: Its path item, whose parameters it inherits.

    Returns:
        The object schema, and the components its body refers to.
    """
    components: dict[str, Any] = spec["components"]["schemas"]
    properties: dict[str, Any] = {}
    required: list[str] = []
    roots: set[str] = set()

    for parameter in operation_parameters(operation, item):
        if parameter.get("in") not in ("path", "query"):
            continue
        name = parameter["name"]
        properties[name] = json_schema(parameter["schema"], components)
        if parameter.get("required"):
            required.append(name)

    body = operation.get("requestBody", {}).get("content", {})
    for media in body.values():
        reference = media["schema"]["$ref"]
        root = reference.removeprefix(COMPONENTS)
        roots.add(root)
        component = components[root]
        properties.update(
            json_schema(component.get("properties", {}), components)
        )
        required.extend(component.get("required", []))

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = list(dict.fromkeys(required))
    return schema, roots


def operation_shape(
    operation: dict[str, Any], item: dict[str, Any]
) -> dict[str, Any]:
    """Describe how an operation is sent and what it returns.

    Args:
        operation: The operation, with its ``x-path`` and ``x-method``.
        item: Its path item, whose parameters it inherits.

    Returns:
        Its HTTP method, path template, path and query parameters, the
        media types of its body and of its ``200`` response, and the
        component the JSON response refers to.
    """
    parameters = operation_parameters(operation, item)
    body = operation.get("requestBody", {}).get("content", {})
    result = operation["responses"]["200"].get("content", {})
    roots = [
        media["schema"]["$ref"].removeprefix(COMPONENTS)
        for media in json_responses(operation)
    ]
    return {
        "method": operation["x-method"],
        "path": operation["x-path"],
        "path_parameters": [
            p["name"] for p in parameters if p.get("in") == "path"
        ],
        "query": sorted(
            p["name"] for p in parameters if p.get("in") == "query"
        ),
        "body": sorted(body),
        "response": sorted(result),
        "schema": roots[0] if roots else None,
    }


def json_responses(operation: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield the JSON media of an operation's ``200`` response.

    Args:
        operation: The operation.

    Yields:
        The ``application/json`` media object, if there is one.
    """
    content = operation["responses"]["200"].get("content", {})
    if "application/json" in content:
        yield content["application/json"]


def response_example(operation: dict[str, Any]) -> Any:
    """Read the documented response of an operation.

    Args:
        operation: The operation.

    Returns:
        The JSON media's ``example``; ``None`` without one.
    """
    for media in json_responses(operation):
        if "example" in media:
            return media["example"]
    return None


def snapshot_openapi(group: OpenAPIEndpoints) -> dict[str, Any]:
    """Write the snapshot of an endpoint group from its OpenAPI file.

    ``request.schema.json`` holds one object per operation, named like
    the method's parameters (``ListFilesParams``), and the components a
    body refers to; ``response.schema.json`` is the union of the JSON
    results with every component they reference; ``operations.json``
    says how each operation is sent and what it returns
    (``operation_shape``); ``responses.json`` keeps the documented
    results the schema accepts.

    Args:
        group: The endpoint group.

    Returns:
        The group's manifest entry.
    """
    path = CONTRACTS / group.file
    spec: dict[str, Any] = read_json(path)
    components: dict[str, Any] = spec["components"]["schemas"]
    operations = operations_of(spec, group.operations)

    requests: dict[str, Any] = {}
    request_roots: set[str] = set()
    results: list[str] = []
    for name, (operation, item) in operations.items():
        requests[class_name(name)], roots = parameters_schema(
            spec, operation, item
        )
        request_roots |= roots
        for media in json_responses(operation):
            root = media["schema"]["$ref"].removeprefix(COMPONENTS)
            if root not in results:
                results.append(root)

    referenced = set().union(
        *(component_closure(components, root) for root in request_roots)
    )
    requests.update(
        {
            name: json_schema(components[name], components)
            for name in sorted(referenced)
        }
    )
    request = {"$defs": dict(sorted(requests.items()))}
    response = {
        "anyOf": [{"$ref": f"#/$defs/{root}"} for root in results],
        "$defs": {
            name: json_schema(components[name], components)
            for name in sorted(
                set().union(
                    *(component_closure(components, root) for root in results)
                )
            )
        },
    }
    target = CONTRACTS / group.name
    write_json(target / "request.schema.json", request)
    write_json(target / "response.schema.json", response)
    write_json(
        target / "operations.json",
        {
            name: operation_shape(operation, item)
            for name, (operation, item) in operations.items()
        },
    )

    validator = Draft202012Validator(response)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for name, (operation, _) in operations.items():
        body = response_example(operation)
        if body is None:
            continue
        error = first_error(validator, body)
        if error is not None:
            rejected.append({"id": name, "group": "responses", "error": error})
            continue
        kept.append({"id": name, "source": "openapi", "body": body})
    write_json(target / "responses.json", kept)

    info = spec.get("info", {})
    return {
        "operations": list(group.operations),
        "openapi": {
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "openapi": spec.get("openapi"),
            "version": info.get("version"),
            **(
                {
                    "sources": info["x-sources"],
                    "retrieved": info["x-retrieved"],
                }
                if "x-sources" in info
                else {}
            ),
        },
        "responses": len(kept),
        "rejected": rejected,
    }


def main() -> None:
    """Rebuild the snapshots and the manifest."""
    endpoints = {endpoint.name: snapshot(endpoint) for endpoint in ENDPOINTS}
    for group in OPENAPI_ENDPOINTS:
        endpoints[group.name] = snapshot_openapi(group)
    write_json(MANIFEST, {"endpoints": endpoints})
    for name, entry in endpoints.items():
        sys.stdout.write(
            f"{name}: {entry['responses']} responses, "
            f"{entry.get('events', 0)} events, "
            f"{len(entry['rejected'])} rejected\n"
        )


if __name__ == "__main__":
    main()

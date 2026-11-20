"""Follow Anthropic's pinned schema sources and refresh what they make.

Run from the aiollm checkout (``.github/workflows/upstream.yml`` runs it
every week):

    uv run python scripts/anthropic/refresh_upstream.py --out DIR

It compares each pin with its source:

- the SDK: the newest ``anthropic`` release on PyPI, against the pin of
  the ``upstream`` dependency group;
- the Files API: the Markdown of the API reference pages its
  transcription follows, against the hashes in
  ``tests/providers/anthropic/contracts/sources/anthropic-files.pages.json``.

A pin that is behind is moved (the pin, ``uv.lock`` and
``ANTHROPIC_VERSION`` from the new SDK), then the contract snapshot is
rebuilt and the types regenerated, so the working tree holds the whole
change for a pull request; ``DIR/pull-request.md`` describes it, with
the public names the new types add or remove against
``tests/providers/anthropic/public_api.json``, which a person updates. A
changed page is not followed automatically, since the transcription is
written by hand: it is listed in ``DIR/issue.md`` for someone to compare
with the transcription, who then runs this script with
``--update-pages``. Under GitHub Actions the outputs ``pins`` and
``pages`` say whether each file was written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

ROOT: Final = Path(__file__).resolve().parents[2]
SCRIPT: Final = ROOT / "scripts" / "anthropic" / "generate_types.py"
PYPROJECT: Final = ROOT / "pyproject.toml"
TESTS: Final = ROOT / "tests" / "providers" / "anthropic"
SOURCES: Final = TESTS / "contracts" / "sources"
FILES_SPEC: Final = SOURCES / "anthropic-files.openapi.json"
PAGES: Final = SOURCES / "anthropic-files.pages.json"
PUBLIC_API: Final = TESTS / "public_api.json"

EXPORTS: Final = (
    "import importlib, json, sys; "
    "print(json.dumps({n: importlib.import_module(n).__all__ "
    "for n in sys.argv[1:]}))"
)
"""Prints the ``__all__`` of the packages named on the command line."""

ANTHROPIC_PIN: Final = re.compile(r'"anthropic==([^"]+)"')
"""The pin of the ``upstream`` dependency group."""

ANTHROPIC_VERSION: Final = re.compile(
    r'(ANTHROPIC_VERSION: Final = ")([^"]+)"'
)
"""The API version the Anthropic types are generated for."""


@dataclass(frozen=True)
class Change:
    """A source that moved.

    Attributes:
        source: What moved, as the report names it.
        old: The pinned value.
        new: The upstream value.
        link: Where to read the difference.
    """

    source: str
    old: str
    new: str
    link: str


def fetch(url: str) -> bytes:
    """Download a URL.

    ``GH_TOKEN`` is sent to the GitHub API when set, for its rate limit.

    Args:
        url: The URL.

    Returns:
        The body.
    """
    request = urllib.request.Request(  # noqa: S310
        url, headers={"User-Agent": "aiollm-upstream-check"}
    )
    token = os.environ.get("GH_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        body: bytes = response.read()
    return body


def fetch_json(url: str) -> Any:
    """Download and decode a JSON document.

    Args:
        url: The URL.

    Returns:
        The decoded value.
    """
    return json.loads(fetch(url))


def run(*command: str, **env: str) -> str:
    """Run a command in the checkout.

    Args:
        *command: The command and its arguments.
        **env: Variables added to the environment.

    Returns:
        Its standard output.
    """
    result = subprocess.run(  # noqa: S603
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **env},
    )
    return result.stdout


def follow_anthropic() -> Change | None:
    """Move the Anthropic SDK pin to its newest release.

    Returns:
        The change, or ``None`` when the pin is current.
    """
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    match = ANTHROPIC_PIN.search(pyproject)
    assert match is not None
    old = match.group(1)
    new: str = fetch_json("https://pypi.org/pypi/anthropic/json")["info"][
        "version"
    ]
    if new == old:
        return None
    pyproject = ANTHROPIC_PIN.sub(f'"anthropic=={new}"', pyproject)
    PYPROJECT.write_text(pyproject, encoding="utf-8", newline="\n")
    run("uv", "lock")
    api_version = run(
        "uv",
        "run",
        "--group",
        "upstream",
        "python",
        "-c",
        "import anthropic._constants as c; print(c.ANTHROPIC_API_VERSION)",
    ).strip()
    script = SCRIPT.read_text(encoding="utf-8")
    script = ANTHROPIC_VERSION.sub(rf'\g<1>{api_version}"', script)
    SCRIPT.write_text(script, encoding="utf-8", newline="\n")
    return Change(
        source="Anthropic SDK (`anthropic` on PyPI)",
        old=old,
        new=f"{new} (API version {api_version})",
        link=(
            "https://github.com/anthropics/anthropic-sdk-python/compare/"
            f"v{old}...v{new}"
        ),
    )


@dataclass(frozen=True)
class Names:
    """The public names of a package that differ from the snapshot.

    Attributes:
        package: The package.
        added: Names it exports that the snapshot lacks.
        removed: Names of the snapshot it no longer exports.
    """

    package: str
    added: list[str]
    removed: list[str]


def public_names() -> list[Names]:
    """Compare the exported names with ``public_api.json``.

    The packages are imported in a new interpreter, so the names read
    are those of the types just regenerated.

    Returns:
        One entry per package whose names differ.
    """
    snapshot: dict[str, list[str]] = json.loads(
        PUBLIC_API.read_text(encoding="utf-8")
    )
    current: dict[str, list[str]] = json.loads(
        run("uv", "run", "python", "-c", EXPORTS, *snapshot)
    )
    return [
        Names(
            package=package,
            added=sorted(set(current[package]) - set(names)),
            removed=sorted(set(names) - set(current[package])),
        )
        for package, names in snapshot.items()
        if set(current[package]) != set(names)
    ]


def names_section(names: list[Names]) -> str:
    """Describe the changed public names for the pull request.

    Args:
        names: The packages whose names differ.

    Returns:
        A Markdown section.
    """
    if not names:
        return "No public name is added or removed.\n"
    lines = [
        "The public names differ from",
        "`tests/providers/anthropic/public_api.json`, so",
        "`test_the_public_api_matches_its_snapshot` fails until it is",
        "updated; a removed or renamed name is a breaking change:",
        "",
    ]
    for entry in names:
        lines.append(f"- `{entry.package}`")
        lines.extend(f"  - added `{name}`" for name in entry.added)
        lines.extend(f"  - removed `{name}`" for name in entry.removed)
    return "\n".join(lines) + "\n"


def page_hashes() -> dict[str, str]:
    """Hash the Markdown of every page the Files transcription follows.

    Returns:
        The SHA-256 of each page, by its URL.
    """
    spec = json.loads(FILES_SPEC.read_text(encoding="utf-8"))
    return {
        url: hashlib.sha256(fetch(f"{url}.md")).hexdigest()
        for url in spec["info"]["x-sources"]
    }


def changed_pages(hashes: dict[str, str]) -> list[Change]:
    """Compare the pages with the hashes kept beside the transcription.

    Args:
        hashes: The current hashes, by URL.

    Returns:
        One change per page whose Markdown differs or is new.
    """
    kept: dict[str, str] = json.loads(PAGES.read_text(encoding="utf-8"))
    return [
        Change(
            source="Anthropic Files reference page",
            old=kept.get(url, "none")[:12],
            new=digest[:12],
            link=url,
        )
        for url, digest in hashes.items()
        if kept.get(url) != digest
    ]


def table(changes: list[Change]) -> str:
    """Write changes as a Markdown table.

    Args:
        changes: The changes.

    Returns:
        The table.
    """
    rows = [
        "| Source | Pinned | Upstream | Difference |",
        "|---|---|---|---|",
        *(
            f"| {c.source} | {c.old} | {c.new} | [link]({c.link}) |"
            for c in changes
        ),
    ]
    return "\n".join(rows) + "\n"


PULL_REQUEST: Final = """\
The weekly check found a new release of Anthropic's SDK. This pull
request moves the pin, rebuilds the contract snapshot
(`tests/providers/anthropic/contracts/`) and regenerates Anthropic's
types.

{table}
{names}
Before merging, read the diff of the snapshot and of
`src/aiollm/providers/anthropic`:

- a field, type or event that was removed or narrowed is a breaking
  change: add a `**Breaking:**` fragment to `CHANGES/`;
- a new field or event is an `added` fragment, if users can reach it;
- an example listed under `rejected` in `manifest.json` fails the tests
  until it is fixed or removed by hand.

Opened by `.github/workflows/upstream.yml`
(`scripts/anthropic/refresh_upstream.py`).
"""

ISSUE: Final = """\
The weekly check found changes in the API reference pages that
`tests/providers/anthropic/contracts/sources/anthropic-files.openapi.json`
transcribes. Compare each page with the transcription, update it where
the API changed, then run
`uv run python scripts/anthropic/refresh_upstream.py --update-pages` and
commit
the new hashes.

{table}"""


def output(name: str, value: str) -> None:
    """Set an output of the GitHub Actions step, when there is one.

    Args:
        name: The output's name.
        value: Its value.
    """
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with Path(path).open("a", encoding="utf-8") as file:
            file.write(f"{name}={value}\n")


def write_pages(hashes: dict[str, str]) -> None:
    """Keep the given page hashes beside the transcription.

    Args:
        hashes: The hashes, by URL.
    """
    text = json.dumps(hashes, indent=1, sort_keys=True) + "\n"
    PAGES.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    """Check every source and refresh what moved."""
    parser = argparse.ArgumentParser(
        description="Follow Anthropic's pinned schema sources."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(tempfile.gettempdir()) / "aiollm-upstream-anthropic",
        help="folder for pull-request.md and issue.md",
    )
    parser.add_argument(
        "--update-pages",
        action="store_true",
        help="only store the current hashes of the Files reference pages",
    )
    args = parser.parse_args()

    hashes = page_hashes()
    if args.update_pages:
        write_pages(hashes)
        return

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    change = follow_anthropic()
    pins = [] if change is None else [change]
    if pins:
        run(
            "uv",
            "run",
            "--group",
            "upstream",
            "python",
            "-m",
            "tests.providers.anthropic.contracts.update",
        )
        run("uv", "run", "python", str(SCRIPT))
        body = PULL_REQUEST.format(
            table=table(pins), names=names_section(public_names())
        )
        (out / "pull-request.md").write_text(body, encoding="utf-8")

    pages = changed_pages(hashes)
    if pages:
        body = ISSUE.format(table=table(pages))
        (out / "issue.md").write_text(body, encoding="utf-8")

    output("pins", "true" if pins else "false")
    output("pages", "true" if pages else "false")
    for change in [*pins, *pages]:
        sys.stdout.write(f"{change.source}: {change.old} -> {change.new}\n")


if __name__ == "__main__":
    main()

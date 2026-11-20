"""Regenerate Anthropic's types and request parameters with aiollm-codegen.

Run from the aiollm checkout:

    uv run python scripts/anthropic/generate_types.py [--codegen DIR]

Both inputs are pinned, so the same commit always gives the same files
(the CI job ``lint and types`` regenerates them and fails on any
difference). The generator is the ``codegen`` dependency group, a commit
of aiollm-codegen named in ``pyproject.toml`` and ``uv.lock``;
``--codegen DIR`` runs a local checkout instead, to try a change of the
generator. The types come from the contract snapshot in
``tests/providers/anthropic/contracts/messages``, derived from the
pinned ``anthropic`` SDK, and, for the Files API, from
``tests/providers/anthropic/contracts/sources/anthropic-files.openapi.json``,
a transcription of Anthropic's API reference. ``ANTHROPIC_VERSION`` is
the API version of that SDK (a test checks).

The script replaces ``types/<family>.py`` and ``methods/params/`` with
the generated modules, keeps the hand-written ``types/options.py`` and
the method classes in ``methods/``, and writes ``types/__init__.py``
from the generated one with the options added. Nothing else is touched.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
PACKAGE: Final = ROOT / "src" / "aiollm" / "providers" / "anthropic"
MODULE: Final = "aiollm.providers.anthropic"
CONTRACTS: Final = ROOT / "tests" / "providers" / "anthropic" / "contracts"
ANTHROPIC_VERSION: Final = "2026-10-21"

ANTHROPIC_FILES: Final = CONTRACTS / "sources" / "anthropic-files.openapi.json"
"""Anthropic's Files API, transcribed from its API reference."""

HAND_WRITTEN_TYPES = {"options.py"}
"""Modules of ``types/`` the script never replaces."""


def codegen(codegen_dir: Path | None, *args: str, output: Path) -> None:
    """Run aiollm-codegen: the pinned one, or a local checkout."""
    if codegen_dir is None:
        command = ["uv", "run", "--group", "codegen", "aiollm-codegen"]
        cwd = ROOT
    else:
        command = ["uv", "run", "aiollm-codegen"]
        cwd = codegen_dir
    command += [*args, "-o", str(output)]
    subprocess.run(command, cwd=cwd, check=True)  # noqa: S603


def exported(module: Path) -> tuple[list[str], list[str]]:
    """Read the import lines and ``__all__`` of a generated ``__init__``."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    lines = module.read_text(encoding="utf-8").splitlines()
    imports: list[str] = []
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module != "__future__":
            imports.append("\n".join(lines[node.lineno - 1 : node.end_lineno]))
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            names = list(ast.literal_eval(node.value))
    return imports, names


def options_names() -> list[str]:
    """Public names of the hand-written ``types/options.py``."""
    tree = ast.parse((PACKAGE / "types" / "options.py").read_text("utf-8"))
    return [
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    ]


def install(generated: Path) -> None:
    """Replace the generated modules of the package."""
    types_dir = PACKAGE / "types"
    params_dir = PACKAGE / "methods" / "params"

    for module in types_dir.glob("*.py"):
        if module.name not in HAND_WRITTEN_TYPES:
            module.unlink()
    shutil.rmtree(params_dir, ignore_errors=True)
    params_dir.mkdir(parents=True)

    for module in (generated / "types").glob("*.py"):
        if module.name != "__init__.py":
            shutil.copy2(module, types_dir / module.name)
    for module in (generated / "methods" / "params").glob("*.py"):
        shutil.copy2(module, params_dir / module.name)

    imports, names = exported(generated / "types" / "__init__.py")
    options = options_names()
    body = [
        '"""Anthropic\'s own types and the options of a generation call.',
        "",
        "Every module but ``options`` is written by aiollm-codegen through",
        "``scripts/anthropic/generate_types.py``; do not edit them by hand.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        *imports,
        f"from {MODULE}.types.options import (",
        *[f"    {option}," for option in sorted(options)],
        ")",
        "",
        "__all__ = [",
        *[f'    "{item}",' for item in sorted({*names, *options})],
        "]",
        "",
    ]
    (types_dir / "__init__.py").write_text(
        "\n".join(body), encoding="utf-8", newline="\n"
    )


def main() -> None:
    """Generate the types and install them."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--codegen",
        type=Path,
        help="run this checkout of aiollm-codegen, not the pinned one",
    )
    args = parser.parse_args()
    codegen_dir: Path | None = args.codegen and args.codegen.resolve()

    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        # aiollm-codegen names the API after the snapshot's folders,
        # <provider>/<endpoint> ("Anthropic Messages API").
        contract = out / "contracts" / "anthropic" / "messages"
        shutil.copytree(CONTRACTS / "messages", contract)
        codegen(
            codegen_dir,
            str(contract),
            "--package",
            "--contract",
            "/messages",
            "--merge",
            str(ANTHROPIC_FILES),
            "--header",
            f"anthropic-version={ANTHROPIC_VERSION}",
            output=out / "anthropic",
        )
        install(out / "anthropic")

    ruff = ["uv", "run", "ruff"]
    for command in (
        [*ruff, "check", "--select", "I", "--fix", "-q"]
        + [str(PACKAGE / "types" / "__init__.py")],
        [*ruff, "format", "-q", str(PACKAGE)],
    ):
        subprocess.run(command, cwd=ROOT, check=True)  # noqa: S603
    print("Anthropic types regenerated", file=sys.stderr)  # noqa: T201


if __name__ == "__main__":
    main()

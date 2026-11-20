"""The generator's inputs are the files of the contract snapshot."""

from __future__ import annotations

import json
import runpy
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent


def test_the_pinned_sdk_is_the_snapshot_one() -> None:
    # Anthropic's schemas are derived from its SDK, so the snapshot, the
    # pin of the upstream group and the API version the types are made
    # for change together.
    script: dict[str, Any] = runpy.run_path(
        str(ROOT / "scripts" / "anthropic" / "generate_types.py")
    )
    manifest = json.loads(
        (HERE / "contracts" / "manifest.json").read_text("utf-8")
    )
    project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    sdk = manifest["endpoints"]["messages"]["sdk"]

    pin = f"{sdk['package']}=={sdk['version']}"
    assert pin in project["dependency-groups"]["upstream"]
    assert script["ANTHROPIC_VERSION"] == sdk["api_version"]

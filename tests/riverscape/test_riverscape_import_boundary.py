from __future__ import annotations

import ast
from pathlib import Path

RIVERSCAPE_PACKAGE = Path(__file__).resolve().parents[2] / "hydrofragments" / "riverscape"


def test_riverscape_package_never_imports_spatial() -> None:
    offenders: list[str] = []
    for path in sorted(RIVERSCAPE_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("hydrofragments.spatial"):
                offenders.append(f"{path.name}: from {node.module}")
            if isinstance(node, ast.Import):
                offenders.extend(
                    f"{path.name}: import {alias.name}"
                    for alias in node.names
                    if alias.name.startswith("hydrofragments.spatial")
                )
    assert offenders == []

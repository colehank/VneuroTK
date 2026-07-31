from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("docs_script", ROOT / "scripts" / "docs.py")
assert SPEC and SPEC.loader
docs_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = docs_script
SPEC.loader.exec_module(docs_script)

DocsError = docs_script.DocsError


def _nav_targets(items: list) -> list[str]:
    targets: list[str] = []
    for item in items:
        for value in item.values():
            if isinstance(value, str):
                targets.append(value)
            else:
                targets.extend(_nav_targets(value))
    return targets


def _write_notebook(path: Path) -> bytes:
    content = json.dumps(
        {
            "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# Example\n"]}],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return content


def _write_site(root: Path, *, nav_href: str = "../example/") -> tuple[Path, Path]:
    notebook = root / "docs" / "example_ipynb" / "example.ipynb"
    source = _write_notebook(notebook)
    output = root / "site" / "example_ipynb"
    page = output / "example" / "index.html"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        f'<html><body><a class="md-nav__link" href="{nav_href}">Example</a>'
        '<a href="../example.ipynb">Download the original notebook</a></body></html>',
        encoding="utf-8",
    )
    (output / "example.ipynb").write_bytes(source)
    return notebook, page


def test_notebook_navigation_targets_generated_markdown() -> None:
    config = tomllib.loads((ROOT / "zensical.toml").read_text(encoding="utf-8"))
    targets = _nav_targets(config["project"]["nav"])
    notebook_dir = ROOT / "docs" / "example_ipynb"
    expected = {f"example_ipynb/{path.stem}.md" for path in notebook_dir.glob("*.ipynb")}
    actual = {target for target in targets if target.startswith("example_ipynb/")}

    assert actual == expected
    assert not any(target.endswith(".ipynb") for target in targets)


def test_validate_site_accepts_rendered_pages_and_preserved_sources(tmp_path: Path) -> None:
    notebook, _ = _write_site(tmp_path)

    docs_script.validate_site(tmp_path, [notebook])


def test_validate_site_rejects_raw_notebook_navigation(tmp_path: Path) -> None:
    notebook, page = _write_site(tmp_path, nav_href="../example.ipynb")

    with pytest.raises(DocsError, match="raw notebook"):
        docs_script.validate_site(tmp_path, [notebook])

    page.write_text(
        '<html><head><link rel="next" href="../example.ipynb"></head>'
        '<body><a href="../example.ipynb">Download the original notebook</a></body></html>',
        encoding="utf-8",
    )
    with pytest.raises(DocsError, match="raw notebook"):
        docs_script.validate_site(tmp_path, [notebook])


def test_validate_site_rejects_missing_or_changed_notebook(tmp_path: Path) -> None:
    notebook, page = _write_site(tmp_path)
    page.unlink()
    with pytest.raises(DocsError, match="rendered page"):
        docs_script.validate_site(tmp_path, [notebook])

    _, _ = _write_site(tmp_path)
    (tmp_path / "site" / "example_ipynb" / "example.ipynb").write_text("changed", encoding="utf-8")
    with pytest.raises(DocsError, match="does not match"):
        docs_script.validate_site(tmp_path, [notebook])


def test_generated_artifacts_are_ignored_but_not_notebooks() -> None:
    patterns = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "docs/example_ipynb/*.md" in patterns
    assert "docs/example_ipynb/*_files/" in patterns
    assert not any(pattern.endswith("*.ipynb") for pattern in patterns)


def test_convert_never_executes_notebooks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notebook = tmp_path / "docs" / "example_ipynb" / "example.ipynb"
    _write_notebook(notebook)
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        notebook.with_suffix(".md").write_text("# Example\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(docs_script.subprocess, "run", fake_run)
    docs_script.convert_notebooks(tmp_path, [notebook])

    assert len(calls) == 1
    assert "--execute" not in calls[0]
    assert "--ExecutePreprocessor.enabled=False" in calls[0]
    priorities = [arg for arg in calls[0] if arg.startswith("--NbConvertBase.display_data_priority=")]
    assert priorities == [
        "--NbConvertBase.display_data_priority=image/png",
        "--NbConvertBase.display_data_priority=image/jpeg",
        "--NbConvertBase.display_data_priority=image/svg+xml",
        "--NbConvertBase.display_data_priority=text/plain",
    ]
    assert calls[0][:4] == [sys.executable, "-m", "nbconvert", "--to"]


def test_build_is_strict_and_cleans_after_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notebook = tmp_path / "docs" / "example_ipynb" / "example.ipynb"
    _write_notebook(notebook)
    zensical_args: list[str] = []

    def fake_convert(root, notebooks):
        notebook.with_suffix(".md").write_text("generated", encoding="utf-8")
        (notebook.parent / "example_files").mkdir()

    def fake_zensical(root, args):
        zensical_args.extend(args)
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(docs_script, "convert_notebooks", fake_convert)
    monkeypatch.setattr(docs_script, "_run_zensical", fake_zensical)

    with pytest.raises(subprocess.CalledProcessError):
        docs_script.build(tmp_path)

    assert zensical_args == ["build", "--clean", "--strict"]
    assert not notebook.with_suffix(".md").exists()
    assert not (notebook.parent / "example_files").exists()

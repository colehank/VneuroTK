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
DOCS_DEPS_AVAILABLE = all(
    importlib.util.find_spec(module) is not None for module in ("bleach", "nbconvert", "nbformat")
)
requires_docs_deps = pytest.mark.skipif(not DOCS_DEPS_AVAILABLE, reason="requires the docs dependency group")


def _nav_targets(items: list) -> list[str]:
    targets: list[str] = []
    for item in items:
        for value in item.values():
            if isinstance(value, str):
                targets.append(value)
            else:
                targets.extend(_nav_targets(value))
    return targets


def _write_template(root: Path) -> None:
    source = ROOT / "docs" / "templates" / "notebook-markdown.md.j2"
    target = root / "docs" / "templates" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _write_notebook(path: Path, *, outputs: list[dict] | None = None) -> bytes:
    cells: list[dict] = [{"id": "markdown-cell", "cell_type": "markdown", "metadata": {}, "source": ["# Example\n"]}]
    if outputs is not None:
        cells.append(
            {
                "id": "code-cell",
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": outputs,
                "source": ["display(value)\n"],
            }
        )
    content = json.dumps(
        {
            "cells": cells,
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


@requires_docs_deps
def test_rich_output_uses_placeholder_and_not_plain_fallback(tmp_path: Path) -> None:
    notebook = tmp_path / "docs" / "example_ipynb" / "example.ipynb"
    rich = (
        '<style>.dataframe{position:fixed}</style><details open onclick="bad()"><summary>Full output</summary>'
        '<table class="dataframe md-nav"><tr><td>last complete row</td></tr></table>'
        '<pre><span style="color:#ff0000">complete tree</span></pre><script>bad()</script></details>'
    )
    _write_notebook(
        notebook,
        outputs=[
            {
                "data": {"text/html": rich, "text/plain": "truncated fallback"},
                "execution_count": 1,
                "metadata": {},
                "output_type": "execute_result",
            }
        ],
    )
    _write_template(tmp_path)

    manifest = docs_script.convert_notebooks(tmp_path, [notebook])
    markdown = notebook.with_suffix(".md").read_text(encoding="utf-8")

    assert len(manifest) == 1
    assert manifest[0].placeholder in markdown
    assert rich not in markdown
    assert "truncated fallback" not in markdown


@requires_docs_deps
def test_sanitize_rich_output_preserves_structure_and_removes_active_content() -> None:
    html = (
        '<style>.dataframe{position:fixed}</style><details open onclick="bad()"><summary>Full output</summary>'
        '<table class="dataframe md-nav"><tr><td>last complete row</td></tr></table>'
        '<pre><span style="color:#ff0000;position:fixed">complete tree</span></pre>'
        '<script>bad()</script><iframe src="https://evil.invalid"></iframe></details>'
    )

    clean = docs_script.sanitize_rich_html(html)

    assert all(tag in clean for tag in ("<details", "<summary", "<table", "<pre", "<span"))
    assert "last complete row" in clean
    assert "complete tree" in clean
    assert "dataframe" in clean
    assert "md-nav" not in clean
    assert "script" not in clean
    assert "iframe" not in clean
    assert "onclick" not in clean
    assert "position" not in clean
    assert "<style" not in clean


@requires_docs_deps
def test_inject_rich_outputs_replaces_each_placeholder_once(tmp_path: Path) -> None:
    notebook = tmp_path / "docs" / "example_ipynb" / "example.ipynb"
    _write_notebook(notebook)
    page = tmp_path / "site" / "example_ipynb" / "example" / "index.html"
    page.parent.mkdir(parents=True)
    entry = docs_script.RichOutput(
        "example", 1, 0, "<!-- VNEUROTK-RICH:test -->", "<table><tr><td>full</td></tr></table>"
    )
    page.write_text(
        f'<html><body><article class="md-content__inner md-typeset">{entry.placeholder}</article></body></html>',
        encoding="utf-8",
    )

    docs_script.inject_rich_outputs(tmp_path, [entry])
    result = page.read_text(encoding="utf-8")

    assert entry.placeholder not in result
    assert result.count('class="notebook-output notebook-output--html"') == 1
    assert "<table><tbody><tr><td>full</td></tr></tbody></table>" in result

    page.write_text(f"<html><body>{entry.placeholder}{entry.placeholder}</body></html>", encoding="utf-8")
    with pytest.raises(DocsError, match="exactly once"):
        docs_script.inject_rich_outputs(tmp_path, [entry])


@requires_docs_deps
def test_rich_output_falls_back_when_image_or_safe_html_is_unavailable(tmp_path: Path) -> None:
    notebook = tmp_path / "docs" / "example_ipynb" / "example.ipynb"
    _write_notebook(
        notebook,
        outputs=[
            {
                "data": {
                    "image/png": "aW1hZ2U=",
                    "text/html": "<strong>html alternative</strong>",
                    "text/plain": "image fallback",
                },
                "metadata": {},
                "output_type": "display_data",
            },
            {
                "data": {"text/html": "<script>only active</script>", "text/plain": "safe fallback"},
                "metadata": {},
                "output_type": "display_data",
            },
        ],
    )
    _write_template(tmp_path)

    prepared, manifest = docs_script._prepare_notebook(notebook)

    assert manifest == []
    assert "text/html" in prepared.cells[1].outputs[0].data
    assert "text/html" not in prepared.cells[1].outputs[1].data


@requires_docs_deps
def test_image_resources_are_namespaced_per_notebook(tmp_path: Path) -> None:
    template = tmp_path / "docs" / "templates" / "notebook-markdown.md.j2"
    template.parent.mkdir(parents=True)
    template.write_text(
        (ROOT / "docs" / "templates" / "notebook-markdown.md.j2").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    notebooks = []
    for stem in ("first", "second"):
        notebook = tmp_path / "docs" / "example_ipynb" / f"{stem}.ipynb"
        _write_notebook(
            notebook,
            outputs=[
                {
                    "data": {"image/png": "aW1hZ2U=", "text/html": "<strong>alternative</strong>"},
                    "metadata": {},
                    "output_type": "display_data",
                }
            ],
        )
        notebooks.append(notebook)

    manifest = docs_script.convert_notebooks(tmp_path, notebooks)

    assert manifest == []
    for stem in ("first", "second"):
        markdown = (tmp_path / "docs" / "example_ipynb" / f"{stem}.md").read_text(encoding="utf-8")
        assert f"{stem}_files/" in markdown
        resources = list((tmp_path / "docs" / "example_ipynb" / f"{stem}_files").glob("*.png"))
        assert len(resources) == 1
        assert resources[0].read_bytes() == b"image"


@requires_docs_deps
def test_class_filter_does_not_modify_preformatted_text() -> None:
    clean = docs_script.sanitize_rich_html("<pre>print('class=\"remove-me\"')</pre>")

    assert 'class="remove-me"' in clean


def test_notebook_theme_preserves_full_scrollable_output() -> None:
    css = (ROOT / "docs" / "stylesheets" / "extra.css").read_text(encoding="utf-8")

    assert ".notebook-output" in css
    assert "overflow-x: auto" in css
    assert "width: max-content" in css
    assert '[data-md-color-scheme="slate"] .notebook-ansi-' in css
    notebook_css = css[css.index("/* Notebook outputs") :]
    assert "max-height" not in notebook_css
    assert "text-overflow" not in notebook_css
    assert "line-clamp" not in notebook_css


def test_generated_artifacts_are_ignored_but_not_notebooks() -> None:
    patterns = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert "docs/example_ipynb/*.md" in patterns
    assert "docs/example_ipynb/*_files/" in patterns
    assert not any(pattern.endswith("*.ipynb") for pattern in patterns)


@requires_docs_deps
def test_convert_never_executes_notebooks(tmp_path: Path) -> None:
    notebook = tmp_path / "docs" / "example_ipynb" / "example.ipynb"
    _write_notebook(notebook)
    _write_template(tmp_path)

    manifest = docs_script.convert_notebooks(tmp_path, [notebook])
    markdown = notebook.with_suffix(".md").read_text(encoding="utf-8")

    assert manifest == []
    assert "# Example" in markdown
    assert not (notebook.parent / "example_files").exists()


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

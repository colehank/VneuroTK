from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("docs_script", ROOT / "scripts/docs.py")
assert SPEC and SPEC.loader
DOCS_SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DOCS_SCRIPT
SPEC.loader.exec_module(DOCS_SCRIPT)
CONTRACT = json.loads((ROOT / "tests/data/docs_compatibility.json").read_text(encoding="utf-8"))
SITE = Path(os.environ["VNEUROTK_DOCS_SITE"]) if "VNEUROTK_DOCS_SITE" in os.environ else None
USAGE_STEMS = ("path", "data", "viz", "vision_models", "vision_alone", "vision_union")
EXAMPLE_STEMS = ("data", "path", "viz", "vision", "neurovision")


def _load_conf():
    path = ROOT / "docs/conf.py"
    spec = importlib.util.spec_from_file_location("vneurotk_docs_conf", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sphinx_configuration_contract() -> None:
    conf = _load_conf()

    assert conf.html_theme == "pydata_sphinx_theme"
    assert conf.nb_execution_mode == "off"
    assert conf.html_baseurl == "https://colehank.github.io/VneuroTK/"
    assert {"myst_nb", "numpydoc", "sphinx.ext.autodoc", "sphinx.ext.autosummary"} <= set(conf.extensions)


def test_external_toc_contains_notebooks_and_navigation_order() -> None:
    import yaml

    toc = yaml.safe_load((ROOT / "docs/_toc.yml").read_text(encoding="utf-8"))
    sections = toc["parts"][0]["chapters"]
    titles = [entry["title"] for entry in sections]

    assert titles == [
        "Installation",
        "File formats",
        "Usage",
        "Examples",
        "API Reference",
        "Project",
        "Changelog",
    ]
    assert sections[2]["file"] == "usage"
    assert sections[3]["file"] == "examples"
    assert sections[5]["file"] == "project"

    usage_docs = [entry["file"] for entry in sections[2]["sections"]]
    assert usage_docs == [f"usage/{stem}" for stem in USAGE_STEMS]
    assert all((ROOT / f"docs/usage/{stem}.ipynb").is_file() for stem in USAGE_STEMS)
    assert all(not (ROOT / f"docs/usage/{stem}.md").exists() for stem in USAGE_STEMS)

    serialized = json.dumps(toc)
    for stem in EXAMPLE_STEMS:
        assert f"example_ipynb/{stem}" in serialized


def test_landing_pages_are_consistent_and_markdown_has_no_python_walkthroughs() -> None:
    home = (ROOT / "docs/index.md").read_text(encoding="utf-8")
    examples = (ROOT / "docs/examples.md").read_text(encoding="utf-8")
    project = (ROOT / "docs/project.md").read_text(encoding="utf-8")

    assert "Quickstart" not in home
    assert "```python" not in home and "```py" not in home
    for target in ("installation", "usage", "examples", "api", "project"):
        assert f":link: {target}" in home
    for stem in EXAMPLE_STEMS:
        assert f":link: example_ipynb/{stem}" in examples
    assert "`viz`" in examples
    assert ":link: contribute" in project

    markdown = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "docs").rglob("*.md"))
    assert "```python" not in markdown
    assert "```py\n" not in markdown


def test_api_target_contract_is_complete() -> None:
    targets = set(CONTRACT["api_targets"])
    assert len(targets) == 46
    api_text = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "docs/api").glob("*.md"))

    assert all(target in api_text for target in targets)
    assert ":::" not in api_text


def test_redirect_contract_matches_sphinx_configuration() -> None:
    conf = _load_conf()
    expected = {f"{source}/index.html": target for source, target in conf.redirects.items()}

    assert CONTRACT["redirects"] == expected


def test_notebook_sources_are_clean_and_consistent() -> None:
    notebooks = list((ROOT / "docs").rglob("*.ipynb"))
    assert len(notebooks) == len(USAGE_STEMS) + len(EXAMPLE_STEMS)

    forbidden = ("np.ndarray subclass", "/nfs/t", "/var/tmp/", "Pepare VneuroTK")
    transient_output = re.compile(r"\x1b|\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}|:[A-Za-z_][A-Za-z0-9_]*:\d+")
    png_notebooks = {ROOT / "docs/usage/viz.ipynb", ROOT / "docs/example_ipynb/viz.ipynb"}
    for notebook in notebooks:
        node = json.loads(notebook.read_text(encoding="utf-8"))
        assert "widgets" not in node.get("metadata", {})
        headings: list[str] = []
        markdown_sources: list[str] = []
        has_png = False
        for cell in node["cells"]:
            raw_source = cell["source"]
            source = "".join(raw_source) if isinstance(raw_source, list) else raw_source
            if cell["cell_type"] == "markdown":
                headings.extend(line for line in source.splitlines() if line.startswith("# "))
                markdown_sources.append(source)
            elif cell["cell_type"] == "code":
                ast.parse(source)
            for output in cell.get("outputs", []):
                assert output.get("output_type") != "error"
                data = output.get("data", {})
                assert "application/vnd.jupyter.widget-view+json" not in data
                has_png |= "image/png" in data
                stream = "".join(output.get("text", []))
                assert not transient_output.search(stream), notebook
        assert len(headings) == 1, notebook
        assert has_png == (notebook in png_notebooks), notebook
        assert f'href="../{notebook.stem}.ipynb"' in "\n".join(markdown_sources), notebook
        serialized = notebook.read_text(encoding="utf-8")
        assert not any(value in serialized for value in forbidden), notebook


def test_local_link_validation_rejects_missing_targets(tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text('<a href="missing.ipynb">download</a>', encoding="utf-8")

    with pytest.raises(DOCS_SCRIPT.DocsError, match="broken local documentation link"):
        DOCS_SCRIPT.validate_local_links(site)


def test_notebook_source_hook_sanitizes_html_without_mutating_source(tmp_path: Path) -> None:
    original = json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "id": "output",
                    "metadata": {},
                    "outputs": [
                        {
                            "data": {
                                "text/html": (
                                    '<table class="dataframe bad"><tr><td>safe</td></tr></table><script>bad()</script>'
                                ),
                                "text/plain": "fallback",
                            },
                            "metadata": {},
                            "output_type": "display_data",
                        }
                    ],
                    "source": ["display(value)"],
                }
            ],
            "metadata": {"widgets": {"state": {}}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
    )
    source = [original]

    DOCS_SCRIPT.sanitize_notebook_source(None, "example", source)

    assert source[0] != original
    node = json.loads(source[0])
    assert "widgets" not in node["metadata"]
    html = node["cells"][0]["outputs"][0]["data"]["text/html"]
    assert "safe" in html and "dataframe" in html
    assert "script" not in html and 'bad"' not in html


def test_notebook_source_hook_ignores_markdown_and_prefers_images() -> None:
    markdown = ["# unchanged"]
    DOCS_SCRIPT.sanitize_notebook_source(None, "index", markdown)
    assert markdown == ["# unchanged"]

    source = [
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": 1,
                        "id": "output",
                        "metadata": {},
                        "source": [],
                        "outputs": [
                            {
                                "data": {
                                    "image/png": "aW1hZ2U=",
                                    "text/html": "<script>alternative</script>",
                                },
                                "metadata": {},
                                "output_type": "display_data",
                            }
                        ],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
    ]
    DOCS_SCRIPT.sanitize_notebook_source(None, "image", source)
    assert "text/html" in json.loads(source[0])["cells"][0]["outputs"][0]["data"]


def test_sphinx_dependencies_replace_zensical_stack() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    docs = "\n".join(project["dependency-groups"]["docs"]).lower()

    assert all(name in docs for name in ("sphinx", "pydata-sphinx-theme", "myst-nb", "numpydoc"))
    assert all(name not in docs for name in ("zensical", "mkdocstrings", "nbconvert"))


@pytest.mark.skipif(SITE is None, reason="requires VNEUROTK_DOCS_SITE")
def test_built_site_preserves_routes_and_notebook_downloads() -> None:
    from sphinx.util.inventory import InventoryFile

    assert SITE is not None
    for route in CONTRACT["routes"]:
        assert (SITE / route).is_file(), route
    for relative, target in CONTRACT["redirects"].items():
        redirect = (SITE / relative).read_text(encoding="utf-8")
        assert f"url={target}" in redirect
    for relative in CONTRACT["notebook_downloads"]:
        built = SITE / relative
        source = ROOT / "docs" / relative
        assert built.read_bytes() == source.read_bytes()
    inventory_path = SITE / "objects.inv"
    assert inventory_path.is_file()
    inventory = InventoryFile.loads(inventory_path.read_bytes(), uri="")
    object_names = {name for domain in inventory.data.values() for name in domain}
    assert set(CONTRACT["api_targets"]) <= object_names
    assert (SITE / "sitemap.xml").is_file()
    assert (SITE / ".nojekyll").is_file()
    assert (SITE / "404.html").is_file()

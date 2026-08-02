from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = Path("docs")
SITE_DIR = Path("site")
CONTRACT_PATH = Path("tests/data/docs_compatibility.json")
ALLOWED_TAGS = {
    "div",
    "details",
    "summary",
    "strong",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
    "pre",
    "span",
}
ANSI_COLORS = {
    "#000080": "blue",
    "#0000ff": "bright-blue",
    "#008000": "green",
    "#008080": "cyan",
    "#7f7f7f": "gray",
    "#800000": "red",
    "#800080": "magenta",
    "#808000": "yellow",
    "#00ff00": "bright-green",
    "#ff0000": "bright-red",
    "#ffff00": "bright-yellow",
    "#ffffff": "bright-white",
}
ALLOWED_CLASSES = {
    "vtk-info",
    "vtk-na",
    "dataframe",
    "notebook-ansi-bold",
    *(f"notebook-ansi-{name}" for name in ANSI_COLORS.values()),
}
IMAGE_MIMES = {"image/png", "image/jpeg", "image/svg+xml"}


class DocsError(RuntimeError):
    """Raised when documentation preparation or validation fails."""


def discover_notebooks(root: Path = ROOT) -> list[Path]:
    notebooks = sorted((root / DOCS_DIR).rglob("*.ipynb"))
    if not notebooks:
        raise DocsError(f"no notebooks found in {root / DOCS_DIR}")
    return notebooks


def _sanitize_attributes(tag: str, name: str, value: str) -> bool:
    if name == "open" and tag == "details":
        return True
    return name in {"class", "style"}


def _filter_clean_classes(fragment: str) -> str:
    pattern = re.compile(r'(<[a-zA-Z][^<>]*?)\sclass="([^"]*)"')

    def replace(match: re.Match[str]) -> str:
        classes = [class_name for class_name in match.group(2).split() if class_name in ALLOWED_CLASSES]
        class_attribute = f' class="{" ".join(classes)}"' if classes else ""
        return f"{match.group(1)}{class_attribute}"

    return pattern.sub(replace, fragment)


def _normalize_rich_styles(fragment: str) -> str:
    style_pattern = re.compile(r'<span\s+style="([^"]*)"([^>]*)>', re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        style = match.group(1).lower()
        classes: list[str] = []
        color = re.search(r"(?:^|;)\s*color:\s*(#[0-9a-f]{6})", style)
        if color and color.group(1) in ANSI_COLORS:
            classes.append(f"notebook-ansi-{ANSI_COLORS[color.group(1)]}")
        if re.search(r"(?:^|;)\s*font-weight:\s*bold", style):
            classes.append("notebook-ansi-bold")
        class_attribute = f' class="{" ".join(classes)}"' if classes else ""
        return f"<span{class_attribute}{match.group(2)}>"

    return style_pattern.sub(replace, fragment)


def sanitize_rich_html(fragment: str) -> str:
    import bleach
    from bleach.css_sanitizer import CSSSanitizer

    fragment = _normalize_rich_styles(fragment)
    fragment = re.sub(r"<style\b[^>]*>.*?</style\s*>", "", fragment, flags=re.IGNORECASE | re.DOTALL)
    fragment = re.sub(r"<(script|iframe)\b[^>]*>.*?</\1\s*>", "", fragment, flags=re.IGNORECASE | re.DOTALL)
    cleaned = bleach.clean(
        fragment,
        tags=ALLOWED_TAGS,
        attributes=_sanitize_attributes,
        css_sanitizer=CSSSanitizer(allowed_css_properties=["text-align"]),
        strip=True,
        strip_comments=True,
    )
    return _filter_clean_classes(cleaned)


def sanitize_notebook_source(app: Any, docname: str, source: list[str]) -> None:
    """Sanitize notebook output in Sphinx's in-memory source document."""
    try:
        notebook = json.loads(source[0])
    except (json.JSONDecodeError, TypeError):
        return
    if not isinstance(notebook, dict) or "cells" not in notebook or "nbformat" not in notebook:
        return

    notebook.get("metadata", {}).pop("widgets", None)
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            data.pop("application/vnd.jupyter.widget-view+json", None)
            fragment = data.get("text/html")
            if fragment is None or IMAGE_MIMES.intersection(data):
                continue
            raw = "".join(fragment) if isinstance(fragment, list) else fragment
            clean = sanitize_rich_html(raw)
            if re.sub(r"<[^>]+>", "", clean).strip():
                data["text/html"] = clean
            else:
                data.pop("text/html", None)
    source[0] = json.dumps(notebook, ensure_ascii=False)


def _run_sphinx(root: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "dirhtml",
            "-W",
            "--keep-going",
            "docs",
            "site",
        ],
        cwd=root,
        check=True,
    )


def finalize_site(root: Path = ROOT) -> None:
    site = root / SITE_DIR
    for notebook in discover_notebooks(root):
        destination = site / notebook.relative_to(root / DOCS_DIR)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(notebook, destination)

    nested_404 = site / "404" / "index.html"
    root_404 = site / "404.html"
    if nested_404.is_file():
        shutil.copyfile(nested_404, root_404)
    if not root_404.is_file():
        raise DocsError(f"missing generated not-found page: {nested_404}")


def validate_site(root: Path = ROOT) -> None:
    site = root / SITE_DIR
    contract = json.loads((root / CONTRACT_PATH).read_text(encoding="utf-8"))
    for route in contract["routes"]:
        if not (site / route).is_file():
            raise DocsError(f"missing documentation route: {route}")
    for relative, target in contract.get("redirects", {}).items():
        redirect = site / relative
        if f"url={target}" not in redirect.read_text(encoding="utf-8"):
            raise DocsError(f"documentation redirect does not target {target}: {relative}")
    for relative in contract["notebook_downloads"]:
        built = site / relative
        source = root / "docs" / relative
        if not built.is_file():
            raise DocsError(f"missing source notebook download: {relative}")
        if built.read_bytes() != source.read_bytes():
            raise DocsError(f"source notebook download does not match: {relative}")
    for artifact in ("objects.inv", "sitemap.xml", ".nojekyll", "404.html"):
        if not (site / artifact).is_file():
            raise DocsError(f"missing documentation artifact: {artifact}")


def build(root: Path = ROOT) -> None:
    site = root / SITE_DIR
    if site.exists():
        shutil.rmtree(site)
    _run_sphinx(root)
    finalize_site(root)
    validate_site(root)


def serve(root: Path = ROOT, *, dev_addr: str = "0.0.0.0:8000") -> None:
    host, separator, port = dev_addr.rpartition(":")
    if not separator or not port.isdigit():
        raise DocsError(f"invalid development address: {dev_addr}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx_autobuild",
            "-b",
            "dirhtml",
            "-W",
            "--keep-going",
            "--host",
            host or "0.0.0.0",
            "--port",
            port,
            "--watch",
            "src",
            "--post-build",
            f'"{sys.executable}" scripts/docs.py finalize',
            "docs",
            "site",
        ],
        cwd=root,
        check=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build VneuroTK documentation with Sphinx and MyST-NB.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="Build strictly and validate the documentation site.")
    subparsers.add_parser("finalize", help="Finalize and validate an existing Sphinx site.")
    serve_parser = subparsers.add_parser("serve", help="Build and serve the documentation with live reload.")
    serve_parser.add_argument("--dev-addr", default="0.0.0.0:8000")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            build()
        elif args.command == "finalize":
            finalize_site()
            validate_site()
        else:
            serve(dev_addr=args.dev_addr)
    except (DocsError, subprocess.CalledProcessError) as exc:
        print(f"docs error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

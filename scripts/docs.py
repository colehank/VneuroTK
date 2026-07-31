from __future__ import annotations

import argparse
import copy
import hashlib
import html
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = Path("docs/example_ipynb")
SITE_NOTEBOOK_DIR = Path("site/example_ipynb")
RICH_MARKER = "VNEUROTK-RICH"
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
CSS_SANITIZER: Any = None


@dataclass(frozen=True)
class RichOutput:
    notebook_stem: str
    cell_index: int
    output_index: int
    placeholder: str
    html: str

    @property
    def output_id(self) -> str:
        return f"{self.notebook_stem}:{self.cell_index}:{self.output_index}"


class DocsError(RuntimeError):
    """Raised when documentation preparation or validation fails."""


class _SiteParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.raw_notebook_links: list[str] = []
        self.download_links: list[str] = []
        self.image_sources: list[str] = []

    @staticmethod
    def _is_notebook(href: str) -> bool:
        return urlparse(href).path.lower().endswith(".ipynb")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        href = attributes.get("href")
        if href is not None and self._is_notebook(href):
            if tag == "a":
                self.download_links.append(href)
                if "md-nav__link" in (attributes.get("class") or "").split():
                    self.raw_notebook_links.append(href)
            if tag == "link" and (attributes.get("rel") or "").lower() in {"prev", "next"}:
                self.raw_notebook_links.append(href)
        src = attributes.get("src")
        if tag == "img" and src is not None:
            self.image_sources.append(src)


def discover_notebooks(root: Path = ROOT) -> list[Path]:
    notebooks = sorted((root / NOTEBOOK_DIR).glob("*.ipynb"))
    if not notebooks:
        raise DocsError(f"no notebooks found in {root / NOTEBOOK_DIR}")
    return notebooks


def _sanitize_attributes(tag: str, name: str, value: str) -> bool:
    if name == "open" and tag == "details":
        return True
    if name == "class":
        return True
    return name == "style"


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


def _rich_placeholder(notebook: Path, cell_index: int, output_index: int, fragment: str) -> str:
    digest = hashlib.sha256(f"{notebook.name}:{cell_index}:{output_index}:".encode() + fragment.encode()).hexdigest()[
        :16
    ]
    return f"<!-- {RICH_MARKER}:{notebook.stem}:{cell_index}:{output_index}:{digest} -->"


def _prepare_notebook(notebook: Path) -> tuple[Any, list[RichOutput]]:
    import nbformat

    node = nbformat.read(notebook, as_version=4)
    prepared = copy.deepcopy(node)
    manifest: list[RichOutput] = []
    for cell_index, cell in enumerate(prepared.cells):
        for output_index, output in enumerate(cell.get("outputs", [])):
            data = output.get("data", {})
            fragment = data.get("text/html")
            if fragment is None:
                continue
            fragment = "".join(fragment) if isinstance(fragment, list) else fragment
            if any(mime in data for mime in ("image/png", "image/jpeg", "image/svg+xml")):
                continue
            clean = sanitize_rich_html(fragment)
            if not re.sub(r"<[^>]+>", "", clean).strip():
                data.pop("text/html", None)
                continue
            placeholder = _rich_placeholder(notebook, cell_index, output_index, fragment)
            data["text/html"] = placeholder
            manifest.append(RichOutput(notebook.stem, cell_index, output_index, placeholder, fragment))
    return prepared, manifest


def clean_generated(root: Path = ROOT, notebooks: Sequence[Path] | None = None) -> None:
    notebooks = discover_notebooks(root) if notebooks is None else notebooks
    for notebook in notebooks:
        notebook.with_suffix(".md").unlink(missing_ok=True)
        resources = notebook.parent / f"{notebook.stem}_files"
        if resources.is_dir():
            shutil.rmtree(resources)


def convert_notebooks(root: Path, notebooks: Sequence[Path]) -> list[RichOutput]:
    from nbconvert import MarkdownExporter
    from nbconvert.writers import FilesWriter

    notebook_dir = root / NOTEBOOK_DIR
    template_dir = root / "docs/templates"
    manifest: list[RichOutput] = []
    for notebook in notebooks:
        prepared, rich_outputs = _prepare_notebook(notebook)
        manifest.extend(rich_outputs)
        exporter = MarkdownExporter(
            template_file="notebook-markdown.md.j2",
            extra_template_paths=[str(template_dir)],
            config={
                "ExecutePreprocessor": {"enabled": False},
                "NbConvertBase": {
                    "display_data_priority": [
                        "image/png",
                        "image/jpeg",
                        "image/svg+xml",
                        "text/html",
                        "text/plain",
                    ]
                },
            },
        )
        body, resources = exporter.from_notebook_node(
            prepared,
            resources={
                "metadata": {"name": notebook.stem},
                "unique_key": notebook.stem,
                "output_files_dir": f"{notebook.stem}_files",
            },
        )
        writer = FilesWriter(build_directory=str(notebook_dir))
        writer.write(body, resources, notebook_name=notebook.stem)
        markdown = notebook.with_suffix(".md")
        notice = (
            f"<!-- Generated from {notebook.name} by scripts/docs.py; do not edit. -->\n\n"
            f"[Download the original notebook]({notebook.name})\n\n"
        )
        markdown.write_text(notice + markdown.read_text(encoding="utf-8"), encoding="utf-8")
    return manifest


def inject_rich_outputs(root: Path, manifest: Sequence[RichOutput]) -> None:
    by_notebook: dict[str, list[RichOutput]] = {}
    for entry in manifest:
        by_notebook.setdefault(entry.notebook_stem, []).append(entry)
    for notebook_stem, entries in by_notebook.items():
        page = root / SITE_NOTEBOOK_DIR / notebook_stem / "index.html"
        document = page.read_text(encoding="utf-8")
        for entry in entries:
            if document.count(entry.placeholder) != 1:
                raise DocsError(f"rich output placeholder must occur exactly once: {entry.output_id}")
            clean = sanitize_rich_html(entry.html)
            wrapper = (
                f'<div class="notebook-output notebook-output--html" '
                f'data-notebook-output="{html.escape(entry.output_id)}" tabindex="0" '
                f'aria-label="Notebook output">{clean}</div>'
            )
            document = document.replace(entry.placeholder, wrapper)
        unresolved = re.search(rf"<!--\s*{RICH_MARKER}:[^>]+-->", document)
        if unresolved:
            raise DocsError(f"unresolved rich output placeholder in {page}")
        page.write_text(document, encoding="utf-8")


def validate_site(root: Path, notebooks: Sequence[Path]) -> None:
    site_notebook_dir = root / SITE_NOTEBOOK_DIR
    for notebook in notebooks:
        page = site_notebook_dir / notebook.stem / "index.html"
        if not page.is_file():
            raise DocsError(f"missing rendered page for {notebook.name}: {page}")
        html = page.read_text(encoding="utf-8")
        if "<html" not in html.lower():
            raise DocsError(f"rendered page is not HTML: {page}")

        copied = site_notebook_dir / notebook.name
        if not copied.is_file():
            raise DocsError(f"missing source notebook download: {copied}")
        if copied.read_bytes() != notebook.read_bytes():
            raise DocsError(f"source notebook download does not match {notebook}")

        parser = _SiteParser()
        parser.feed(html)
        expected_download = f"../{notebook.name}"
        if expected_download not in parser.download_links:
            raise DocsError(f"missing source notebook link in {page}: {expected_download}")
        for src in parser.image_sources:
            parsed = urlparse(src)
            if parsed.scheme or parsed.netloc or parsed.path.startswith("/") or parsed.path.startswith("data:"):
                continue
            asset = (page.parent / parsed.path).resolve()
            if not asset.is_file():
                raise DocsError(f"missing local image referenced by {page}: {src}")

    for page in (root / "site").rglob("*.html"):
        parser = _SiteParser()
        parser.feed(page.read_text(encoding="utf-8"))
        if parser.raw_notebook_links:
            links = ", ".join(parser.raw_notebook_links)
            raise DocsError(f"raw notebook navigation in {page}: {links}")


def _run_zensical(root: Path, args: Sequence[str]) -> None:
    subprocess.run([sys.executable, "-m", "zensical", *args], cwd=root, check=True)


def build(root: Path = ROOT) -> None:
    notebooks = discover_notebooks(root)
    clean_generated(root, notebooks)
    try:
        manifest = convert_notebooks(root, notebooks)
        _run_zensical(root, ["build", "--clean", "--strict"])
        inject_rich_outputs(root, manifest)
        validate_site(root, notebooks)
    finally:
        clean_generated(root, notebooks)


def serve(root: Path = ROOT, *, dev_addr: str = "0.0.0.0:8000") -> None:
    build(root)
    host, separator, port = dev_addr.rpartition(":")
    if not separator or not port.isdigit():
        raise DocsError(f"invalid development address: {dev_addr}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "http.server",
            port,
            "--bind",
            host or "0.0.0.0",
            "--directory",
            str(root / "site"),
        ],
        cwd=root,
        check=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build VneuroTK documentation with rendered notebooks.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="Convert notebooks, build strictly, and validate the site.")
    serve_parser = subparsers.add_parser("serve", help="Convert notebooks and serve the documentation.")
    serve_parser.add_argument("--dev-addr", default="0.0.0.0:8000")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            build()
        else:
            serve(dev_addr=args.dev_addr)
    except (DocsError, subprocess.CalledProcessError) as exc:
        print(f"docs error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

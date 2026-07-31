from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = Path("docs/example_ipynb")
SITE_NOTEBOOK_DIR = Path("site/example_ipynb")


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


def clean_generated(root: Path = ROOT, notebooks: Sequence[Path] | None = None) -> None:
    notebooks = discover_notebooks(root) if notebooks is None else notebooks
    for notebook in notebooks:
        notebook.with_suffix(".md").unlink(missing_ok=True)
        resources = notebook.parent / f"{notebook.stem}_files"
        if resources.is_dir():
            shutil.rmtree(resources)


def convert_notebooks(root: Path, notebooks: Sequence[Path]) -> None:
    notebook_dir = root / NOTEBOOK_DIR
    for notebook in notebooks:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "nbconvert",
                "--to",
                "markdown",
                "--template-file",
                str(root / "docs/templates/notebook-markdown.md.j2"),
                "--ExecutePreprocessor.enabled=False",
                "--NbConvertBase.display_data_priority=image/png",
                "--NbConvertBase.display_data_priority=image/jpeg",
                "--NbConvertBase.display_data_priority=image/svg+xml",
                "--NbConvertBase.display_data_priority=text/plain",
                "--output",
                notebook.stem,
                "--output-dir",
                str(notebook_dir),
                str(notebook),
            ],
            cwd=root,
            check=True,
        )
        markdown = notebook.with_suffix(".md")
        body = markdown.read_text(encoding="utf-8")
        notice = (
            f"<!-- Generated from {notebook.name} by scripts/docs.py; do not edit. -->\n\n"
            f"[Download the original notebook]({notebook.name})\n\n"
        )
        markdown.write_text(notice + body, encoding="utf-8")


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
        convert_notebooks(root, notebooks)
        _run_zensical(root, ["build", "--clean", "--strict"])
        validate_site(root, notebooks)
    finally:
        clean_generated(root, notebooks)


def serve(root: Path = ROOT, *, dev_addr: str = "0.0.0.0:8000") -> None:
    notebooks = discover_notebooks(root)
    clean_generated(root, notebooks)
    try:
        convert_notebooks(root, notebooks)
        _run_zensical(root, ["serve", "--dev-addr", dev_addr])
    finally:
        clean_generated(root, notebooks)


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

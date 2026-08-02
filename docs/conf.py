from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

project = "VneuroTK"
author = "VneuroTK contributors"
copyright = "2026, VneuroTK contributors"

extensions = [
    "myst_nb",
    "numpydoc",
    "notfound.extension",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_external_toc",
    "sphinx_reredirects",
    "sphinx_sitemap",
]

source_suffix = {
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}
external_toc_path = "_toc.yml"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
myst_fence_as_directive = ["eval-rst"]
nb_execution_mode = "off"
nb_merge_streams = True
nb_mime_priority_overrides = [
    ("html", "image/svg+xml", 10),
    ("html", "image/png", 20),
    ("html", "image/jpeg", 30),
    ("html", "text/html", 40),
    ("html", "text/markdown", 50),
    ("html", "text/latex", 60),
    ("html", "text/plain", 70),
]

numpydoc_show_class_members = False
autosummary_generate = True
autodoc_typehints = "description"
autodoc_mock_imports = ["matplotlib", "torch", "transformers"]

html_theme = "pydata_sphinx_theme"
html_title = project
html_baseurl = "https://colehank.github.io/VneuroTK/"
html_logo = "assets/logo.svg"
html_favicon = "assets/logo.svg"
html_static_path = ["stylesheets"]
html_css_files = ["extra.css"]
html_theme_options = {
    "github_url": "https://github.com/colehank/vneurotk",
    "logo": {"text": project},
    "show_toc_level": 2,
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "colehank",
    "github_repo": "vneurotk",
    "github_version": "main",
    "doc_path": "docs",
}

sitemap_url_scheme = "{link}"
notfound_urls_prefix = "/VneuroTK/"
notfound_exclude_urls = True
redirects = {
    "usage/vision-models": "../vision_models/",
    "file-formats/hdf5-recordings": "../../format/hdf5/",
    "example/nod-neurovision": "../../example_ipynb/neurovision/",
}


def setup(app):
    from scripts.docs import sanitize_notebook_source

    app.connect("source-read", sanitize_notebook_source)

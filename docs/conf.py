"""Sphinx configuration for the JeevesAgent docs site.

Build locally with::

    pip install -e ".[docs]"
    sphinx-build -b html docs docs/_build/html

The build is autoapi-driven — every public symbol with a Google-
or NumPy-style docstring lands in the API reference automatically.
Hand-written guides live next to the auto-generated reference under
``docs/guides/`` and ``docs/migrations/``.

ReadTheDocs picks this up via ``.readthedocs.yaml`` at the repo
root (no further setup needed; the rebuild is triggered on push).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable for autodoc / autoapi without
# requiring an editable install — useful for ReadTheDocs CI.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# -- Project metadata -------------------------------------------------

project = "JeevesAgent"
author = "JeevesAgent contributors"
copyright = "2026, JeevesAgent contributors"

# Pull the version from the installed package so docs and code never
# drift. Falls back to "0.0.0" only when the package isn't importable
# (e.g. building docs in a fresh checkout before ``pip install``).
try:
    from jeevesagent import __version__ as _pkg_version
    release = _pkg_version
    version = ".".join(_pkg_version.split(".")[:2])
except ImportError:
    release = "0.0.0"
    version = "0.0"


# -- Extensions -------------------------------------------------------

extensions = [
    # Markdown support — lets us mount the existing docs/*.md files
    # alongside any new .rst content without rewriting them.
    "myst_parser",
    # Auto-generate API reference from package source. autoapi runs at
    # build time, walks ``jeevesagent/``, and emits one page per
    # module with all public symbols documented from their inline
    # docstrings.
    "autoapi.extension",
    # Cross-reference standard library types (``int``, ``Path``,
    # ``BaseModel``) into the right targets.
    "sphinx.ext.intersphinx",
    # Render type hints in function signatures and parameter lists.
    "sphinx.ext.napoleon",
    # Pull in admonitions / code copy buttons / nicer rendering.
    "sphinx.ext.viewcode",
]


# -- General settings -------------------------------------------------

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]


# -- Theme ------------------------------------------------------------

html_theme = "furo"
html_title = f"JeevesAgent {release}"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/anthropics/jeevesagent",
    "source_branch": "main",
    "source_directory": "docs/",
}


# -- autoapi ----------------------------------------------------------

autoapi_type = "python"
autoapi_dirs = [str(_REPO_ROOT / "jeevesagent")]
autoapi_root = "api"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_keep_files = False
autoapi_member_order = "groupwise"
autoapi_python_class_content = "both"  # combine class + __init__ docs

# Skip private modules from the API tree — they're exposed only for
# tests / internal wiring, not for consumers to import.
autoapi_ignore = [
    "*/tests/*",
    "*/_internal/*",
    "*/__pycache__/*",
]


# -- Napoleon (Google / NumPy docstrings) -----------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True


# -- intersphinx (link to stdlib + Pydantic + Anyio) ------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest", None),
    "anyio": ("https://anyio.readthedocs.io/en/stable/", None),
}


# -- MyST parser (Markdown) -------------------------------------------

myst_enable_extensions = [
    "colon_fence",     # ::: admonitions
    "deflist",         # definition lists
    "linkify",         # auto-link bare URLs
    "tasklist",        # GitHub-style task lists
]
myst_heading_anchors = 3

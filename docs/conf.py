"""Sphinx configuration for jaxlibxc documentation."""

import sys
from pathlib import Path

# Add project root to sys.path so autodoc can import jaxlibxc
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# -- Project information -----------------------------------------------------

project = "jaxlibxc"
copyright = "2024, Connor Blake"
author = "Connor Blake"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# MyST-Parser settings
myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
]

# Source file suffixes
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

# Autodoc settings
autodoc_mock_imports = [
    "jax",
    "jaxlib",
]
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}
html_static_path = []

# Suppress warnings about missing references from mocked imports
nitpicky = False

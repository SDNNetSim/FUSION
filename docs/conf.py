# Configuration file for the Sphinx documentation builder.
import os
import sys
from datetime import datetime

# Add project root to path for autodoc
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "FUSION"
copyright = f"2022-{datetime.now().year}, FUSION Development Team"
author = "Ryan McCann, Arash Rezaee, Vinod M. Vokkarane"
release = "6.0.0"
version = "6.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    # "sphinx.ext.intersphinx",  # Disabled to speed up builds (not currently used)
    "sphinx.ext.todo",
    "myst_parser",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "__pycache__"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "en"

# -- Extension configuration -------------------------------------------------

# Napoleon (Google/NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Autodoc
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"  # Cleaner class signatures

# Autosummary
autosummary_generate = True

# Intersphinx - disabled to speed up builds (not currently used)
# To re-enable, uncomment the mapping and add "sphinx.ext.intersphinx" to extensions
# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3", None),
#     "numpy": ("https://numpy.org/doc/stable/", None),
#     "networkx": ("https://networkx.org/documentation/stable/", None),
#     "pandas": ("https://pandas.pydata.org/docs/", None),
# }

# MyST parser
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]
myst_heading_anchors = 3
# Suppress warnings for:
# - myst.header: MyST heading anchor warnings
# - autodoc.duplicate_object: Dataclass fields documented multiple times
# - ref.python: Ambiguous cross-references (multiple classes with same name)
# - app.add_object: Duplicate object descriptions (same module documented twice)
suppress_warnings = [
    "myst.header",
    "autodoc.duplicate_object",
    "ref.python",
    "app.add_object",
]

# Avoid duplicate object warnings from dataclass fields
# by not re-documenting inherited members
autodoc_inherit_docstrings = False

# Todo
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = "FUSION"

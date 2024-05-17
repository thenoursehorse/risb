# ruff: noqa: D100
from __future__ import annotations

import importlib.metadata

project = "risb"
copyright = "2016-2023 H. L. Nourse and B. J. Powell, 2016-2022 R. H. McKenzie"
author = "H. L. Nourse"
try:
    version = release = importlib.metadata.version("risb")
except:  # noqa: E722
    version = release = "0.0.0"

extensions = [
    "myst_parser",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # Type hinting just does not work nicely with such abstract objects
    # "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    # "autodoc2",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
]

# autodoc2_packages = [
#    "../src/risb",
# ]
#
# autodoc2_hidden_objects = [
#    "private",
# ]

# Don't show typehints/annotations
autodoc_typehints = "none"

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_theme = "furo"
html_title = "risb " + version
html_theme_options = {
    "source_repository": "https://github.com/thenoursehorse/risb/",
    "source_branch": "3.2.x",
    "source_directory": "docs/",
}

myst_enable_extensions = [
    "colon_fence",
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
    "linkify",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    # I don't think TRIQS has intersphinx set up?
    # "triqs": ('https://triqs.github.io/triqs/latest/documentation/', None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]

always_document_param_types = True

myst_substitutions = {
    "RISB": r"{abbr}`RISB (rotationally invariant slave-bosons)`",
    "DFT": r"{abbr}`DFT (density functional theory)`",
    "DMFT": r"{abbr}`DMFT (dynamical mean-field theory)`",
    "TRIQS": r"[TRIQS](https://triqs.github.io/)",
    "DIIS": r"{abbr}`DIIS (direct inversion in the iterative subspace)`",
    "DMRG": r"{abbr}`DMRG (density matrix renormalization group)`",
}

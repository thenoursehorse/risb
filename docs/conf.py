# ruff: noqa: D100, D103
from __future__ import annotations

import importlib.metadata

from sphinx.ext.intersphinx import missing_reference

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

# Parse numpydoc types (e.g. "int, optional") instead of linking every word
napoleon_preprocess_types = True

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

# Generate anchors for headings so pages can link to e.g. page.md#heading
myst_heading_anchors = 3

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
    "triqs": ("https://triqs.github.io/triqs/latest/", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
    # fockit has no hosted docs to link to
    ("py:class", "fockit.algebra.OpSum"),
    ("py:class", "fockit.hilbert.Basis"),
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


# Names in docstring types that are plain words, or that numpy and TRIQS
# document under another path or object type: (role, target in an inventory)
_external_targets = {
    "callable": ("class", "collections.abc.Callable"),
    "mapping": ("class", "collections.abc.Mapping"),
    "ndarray": ("class", "numpy.ndarray"),
    "sequence": ("class", "collections.abc.Sequence"),
    "numpy._typing._array_like.NDArray": ("data", "numpy.typing.NDArray"),
    "numpy.float64": ("attr", "numpy.float64"),
    "triqs.gfs.BlockGf": ("class", "triqs.gfs.block_gf.BlockGf"),
    "triqs.gfs.MeshImFreq": ("class", "triqs.mesh.meshes.MeshImFreq"),
    "triqs.gfs.MeshProduct": ("class", "triqs.mesh.mesh_product.MeshProduct"),
    "triqs.gfs.MeshReFreq": ("class", "triqs.mesh.meshes.MeshReFreq"),
    "triqs.operators.Operator": ("class", "triqs.operators.operators.Operator"),
}


def _resolve_external_reference(app, env, node, contnode):
    if node.get("refdomain") != "py" or node["reftarget"] not in _external_targets:
        return None
    node["reftype"], node["reftarget"] = _external_targets[node["reftarget"]]
    return missing_reference(app, env, node, contnode)


def setup(app):
    app.connect("missing-reference", _resolve_external_reference)

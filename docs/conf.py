# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import importlib.metadata


# -- Project information -----------------------------------------------------

project = "FFTL"
release = importlib.metadata.version("fftl")
copyright = "2022-2025, Nicolas Tessore"
author = "Nicolas Tessore"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


# -- numpydoc extension -----------------------------------------------------

numpydoc_use_plots = True
numpydoc_show_class_members = False


# -- matplotlib extension ----------------------------------------------------

plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False

plot_font_size = 11

plot_rcparams = {
    "font.size": plot_font_size,
    "axes.titlesize": plot_font_size,
    "axes.labelsize": plot_font_size,
    "xtick.labelsize": plot_font_size,
    "ytick.labelsize": plot_font_size,
    "legend.fontsize": plot_font_size,
    "legend.frameon": False,
    "figure.figsize": (6, 4),
    "text.usetex": False,
    "savefig.bbox": "tight",
}

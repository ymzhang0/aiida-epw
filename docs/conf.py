"""Configuration file for the Sphinx documentation builder."""
# docs/conf.py

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# This adds the project's root directory (which is one level up from 'docs/')
# to Python's path. This is crucial for autodoc to find your source code.
sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "aiida-epw"
copyright = "2025, Yiming Zhang, Marnik Bercx"
author = "Yiming Zhang, Marnik Bercx"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",  # For Markdown file support
    "sphinx.ext.autodoc",  # The core engine for generating API docs from docstrings
    "sphinx.ext.napoleon",  # To understand NumPy/Google style docstrings
    "sphinx.ext.viewcode",  # To add links from the docs to the source code
    "sphinx.ext.intersphinx",  # To link to other projects' documentation (e.g., AiiDA's)
]

# This allows you to write math in your markdown files
myst_enable_extensions = ["dollarmath", "amsmath"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
# html_theme = 'sphinx_rtd_theme' # The classic Read the Docs theme is a great choice
html_theme = "sphinx_book_theme"  # The classic Read the Docs theme is a great choice

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

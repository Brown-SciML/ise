# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # Adjust if needed

project = 'ise'
copyright = '2025, Peter Van Katwyk'
author = 'Peter Van Katwyk'
release = '1.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',  # Links to source code in docs
    'sphinx.ext.autosummary',  # Generates API summary tables,
    'sphinx_rtd_theme',  # Read the Docs theme
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "/")
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

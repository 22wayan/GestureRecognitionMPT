# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GestureRecognitionMPT'
copyright = '2026, Jannis Bollien, Dennis Müller'
author = 'Jannis Bollien, Dennis Müller'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
autoclass_content = "both"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon"
]

templates_path = ['_templates']
exclude_patterns = []

language = 'de'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

import os
import sys

# Projekt-Wurzel in den Python-Pfad haengen, damit Sphinx (autodoc) unsere
# Module unter GestureRecognition/ importieren und ihre Docstrings lesen kann.
sys.path.insert(0, os.path.abspath('../..'))
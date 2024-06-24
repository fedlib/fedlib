import os.path as osp
import sys

import blades_sphinx_theme

import fedlib

version = fedlib.__version__

project = "blades"
copyright = "2023, Shenghui Li"
author = "Shenghui Li"
release = "0.2"

sys.path.append(osp.join(osp.dirname(blades_sphinx_theme.__file__), "extension"))

templates_path = ["_templates"]
add_module_names = False
autodoc_member_order = "bysource"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "blades_sphinx",
    "sphinx.ext.doctest",
    # "sphinx_gallery.gen_gallery",
    # "sphinx_gallery.load_style",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "blades_sphinx_theme"
# html_theme = "pytorch_sphinx_theme"
html_static_path = ["_static"]

# html_logo = "_static/blades_logo.png"
# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.make
html_favicon = "_static/favicon.ico"


intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None),
    "torch": ("https://pytorch.org/docs/master", None),
}


def rst_jinja_render(app, _, source):
    if hasattr(app.builder, "templates"):
        rst_context = {"fedlib": fedlib}
        source[0] = app.builder.templates.render_string(source[0], rst_context)


def setup(app):
    app.connect("source-read", rst_jinja_render)
    app.add_js_file("js/version_alert.js")

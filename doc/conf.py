# Configuration file for the Sphinx documentation builder.

import os
import textwrap
import furiosa

import sys
sys.path.append('../build/src/module')

# -- Project information -----------------------------------------------------

project = 'pocky'
copyright = '2024, Ellen M. Price'
author = '@emprice'

# The full version, including alpha/beta/rc tags
release = '1.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['furiosa', 'breathe', 'exhale', 'sphinx.ext.todo',
              'sphinx.ext.autodoc', 'sphinx.ext.napoleon']

root_dir = os.environ['ROOT_DIR']

breathe_projects = { 'pocky' : './doxygen/xml' }
breathe_default_project = 'pocky'
breathe_domain_by_extension = { 'h' : 'c', 'cl' : 'c' }
breathe_implementation_filename_extensions = ['c']
breathe_show_define_initializer = True
breathe_show_enumvalue_initializer = True
breathe_separate_member_pages = False

stdin = textwrap.dedent(f'''\
OUTPUT_LANGUAGE         = English
EXTENSION_MAPPING       = h=C cl=C
EXCLUDE_PATTERNS       += *.txt *.c *.py
INPUT                   = {root_dir}/src
RECURSIVE               = YES
QUIET                   = YES
PREDEFINED             += DOXYGEN_SHOULD_SKIP_THIS
ENABLE_PREPROCESSING    = YES
OPTIMIZE_OUTPUT_FOR_C   = YES
''')

exhale_args = { 'containmentFolder' : './exhale',
    'rootFileName' : 'lowlevel.rst', 'rootFileTitle' : 'Low-level C API',
    'createTreeView' : True, 'exhaleExecutesDoxygen' : True,
    'contentsDirectives' : False, 'doxygenStripFromPath' : root_dir,
    'lexerMapping' : { r'.*\.h' : 'c', r'.*\.cl' : 'c' },
    'exhaleDoxygenStdin' : stdin }

smartquotes = True
primary_domain = None
todo_include_todos = True
highlight_language = 'default'

html_title = 'pocky'

bibtex_bibfiles = []
bibtex_default_style = 'unsrt'
bibtex_reference_style = 'label'

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

html_theme = 'furiosa'
pygments_style = 'nordlight'
pygments_dark_style = 'norddark'

html_static_path = ['_static']

html_theme_options = {
    'light_css_variables': {
        'font-stack': '\'Open Sans\', sans-serif'
    },
    'dark_css_variables': {
        'font-stack': '\'Open Sans\', sans-serif'
    }
}
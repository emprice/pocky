import os
import numpy as np
from setuptools import Extension, setup

source_dir = 'src/pocky/ext'
include_dir = 'src/pocky/ext/include'

source_files = ['pocky.c', 'pocky_bufpair.c', 'pocky_context.c',
                'pocky_functions.c', 'pocky_helpers.c', 'pocky_utils.c']
source_files = [os.path.join(source_dir, fname) for fname in source_files]

header_files = ['pocky.h', 'pocky_bufpair.h', 'pocky_context.h',
                'pocky_functions.h', 'pocky_helpers.h', 'pocky_utils.h']
header_files = [os.path.join(include_dir, fname) for fname in header_files]

ext_modules = [
    Extension(name='pocky.ext', sources=source_files, libraries=['OpenCL'],
        language='c', include_dirs=[include_dir, np.get_include()],
        depends=header_files)
]
setup(ext_modules=ext_modules)

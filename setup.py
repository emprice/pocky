import os
import numpy as np
from setuptools import Extension, setup

source_dir = 'src/pocky/ext'
include_dir = 'src/pocky/ext/include'

source_files = ['pocky.c', 'bufpair.c', 'context.c',
                'functions.c', 'helpers.c', 'utils.c']
source_files = [os.path.join(source_dir, fname) for fname in source_files]

header_files = ['pocky.h', 'bufpair.h', 'context.h',
                'functions.h', 'helpers.h', 'utils.h']
header_files = [os.path.join(include_dir, fname) for fname in header_files]

ext_modules = [
    Extension(name='pocky.ext', sources=source_files, libraries=['OpenCL'],
        define_macros=[('CL_TARGET_OPENCL_VERSION', '300')], language='c',
        include_dirs=[include_dir, np.get_include()], depends=header_files)
]
setup(ext_modules=ext_modules)

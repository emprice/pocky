from .ext import *

def get_include():
    import os
    import pocky.ext as ext
    return os.path.join(os.path.dirname(ext.__file__), 'ext/include')

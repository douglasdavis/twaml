import re

__version__ = "0.9.2"
version = __version__
version_info = tuple(re.split(r"[-\.]", __version__))

del re

#!/usr/bin/env python

import re

__version__ = '0.0.2b'
version = __version__
version_info = tuple(re.split(r'[-\.]', __version__))

del re

"""
Functions for reading and writing Jeol JDF files, Bruker
JCAMP-DX parameter (acqus) files, and Bruker pulse program (pulseprogram)
files.

"""

import locale
import io

__developer_info__ = """
Jeol data format information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
:

"""


from functools import reduce
import operator
import os
from warnings import warn

import numpy as np

from . import fileiobase
from ..process import proc_base

"""
Processing functions specific for Bruker
Includes functions that mimic TopSpin and NMRPipe implementations 
"""

from __future__ import print_function

import numpy as np

# nmrglue modeules
from ..fileio import bruker, fileiobase
from . import proc_base as p
from . import proc_bl
from . import proc_lp


#---Apodizaton functions

def sp(dic, data, off=0.0, end=1.0, pow=1.0, c=1.0, start=1, size='default',
        inv=False, one=False, hdr):

    if dic['type'] == 'bruker':
       return bruker_proc.sp(dic, data, off, end, pow, c, start, size, inv, one)

    elif dic['type'] == 'nmrpipe':
        return pipe_proc.sp(dic, data, off, end, pow, c, start, size, inv, one)


    start = bruker.get_grpdly(dic, data)

    if start == 0 and size == 'default':
        data = p.sp(data, off, end, pow, inv=inv)
    else:   # only part of the data window is apodized
        if size == 'default':
            stop = data.shape[-1]
        else:
            stop = start + size
        data[..., start:stop] = p.sp(data[..., start:stop], off, end, pow,
                                     inv=inv)
        if one is False:
            data[..., :start] = 0.0
            data[..., stop:] = 0.0

    # first point scaling
    if inv:
        data[..., 0] = data[..., 0] / c
    else:
        data[..., 0] = data[..., 0] * c

    return dic, data
      

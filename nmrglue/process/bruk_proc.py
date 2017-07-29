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

def apod(dic, data, qName=None, q1=1.0, q2=1.0, q3=1.0, c=1.0, start=1,
        size='default', inv=False, one=False):

    if qName == "EM":
        return em(dic, data, q1, c, start, size, inv, one, hdr)
    elif qName == "GM":
        return gm(dic, data, q1, q2, q3, c, start, size, inv, one, hdr)
    elif qName == "GMB":
        return gmb(dic, data, q1, q2, c, start, size, inv, one, hdr)
    elif qName == "JMOD":
        return jmod(dic, data, q1, q2, q3, False, False, c, start, size, inv,
                    one, hdr)
    elif qName == "SP":
        return sp(dic, data, q1, q2, q3, c, start, size, inv, one, hdr)
    elif qName == "SINE":
        return sp(dic, data, q1, end=1.0, pow=1.0, c, start, size, inv, one, hdr)
    elif qName == "QSINE":
        return sp(dic, data, q1, end=1.0, pow=2.0, c, start, size, inv, one, hdr)
    elif qName == "TM" or qName == "TRAP":
        return tm(dic, data, q1, q2, c, start, size, inv, one, hdr)
    elif qName == "TRI":
        return tri(dic, data, q1, q2, q3, c, start, size, inv, one, hdr)
    elif qName == "SINC":
        return sinc(dic, data, q1, q2, q3, c, start, size, inv, one, hdr)
    elif qName == "TRAF":
        return tri(dic, data, q1, q2, q3, c, start, size, inv, one, hdr)
    elif qName == "TRAFS":
        return sinc(dic, data, q1, q2, q3, c, start, size, inv, one, hdr)

    else:
        raise ValueError("qName must be SP, SINE, QSINE, EM, GM, GMB, TM, TRAP
                TRI, SINC, TRAF, TRAFS or JMOD")

def sp(dic, data, off=0.0, end=1.0, pow=1.0, c=1.0, start=1, size='default',
        inv=False, one=False, hdr=False):

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
      

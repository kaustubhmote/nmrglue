# -*- coding: utf-8 -*-
"""
Covariance Processing from raw bruker data

Need to implemet:
1. unit conversion objects for getting the ppm scale
2. 2D ft processing
3. 


"""

import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm



dic, data = ng.bruker.read('.')
data = ng.bruker.remove_digital_filter(dic, data)

data = ng.proc_base.zf(data, pad=8192)
data = ng.proc_base.gm(data, g1=0.0034, g2=0.0045)
data = ng.proc_base.fft(data)
data = ng.proc_base.ps(data, p0=0, p1=0)
data = ng.proc_base.rev(data)
data = ng.proc_base.di(data)
data = data.T
covdata = np.cov(data)


cmap = matplotlib.cm.bone
cont_start = 120*np.mean(covdata)
cont_num = 20
cont_factor = 1.3
cl = contour_start * contour_factor ** np.arange(contour_num)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.contour(covdata, cl, cmap=cmap)
fig.savefig('figure_nmrglue.png')


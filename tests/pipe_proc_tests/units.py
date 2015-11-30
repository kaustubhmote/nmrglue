#! /usr/bin/env python
""" Create files for unit_conversion unit test """

import nmrglue.fileio.pipe as pipe
import nmrglue.process.pipe_proc as p

# points
d, a = pipe.read("1D_time.fid")
d, a = p.ext(d, a, x1=30, xn=340.6)
pipe.write("units1.glue", d, a, overwrite=True)

# percent
d, a = pipe.read("1D_time.fid")
xc = p.make_uc(d, a)
d, a = p.ext(d, a, x1=xc("11%"), xn=xc("60.2%"))
pipe.write("units2.glue", d, a, overwrite=True)

d, a = pipe.read("1D_time.fid")
xc = p.make_uc(d, a)
d, a = p.ext(d, a, x1=xc("20.5%"), xn=xc("78.9%"))
pipe.write("units3.glue", d, a, overwrite=True)

# Hz
d, a = pipe.read("1D_freq_real.dat")
xc = p.make_uc(d, a)
d, a = p.ext(d, a, x1=xc("24000 Hz"), xn=xc("12000 Hz"))
pipe.write("units4.glue", d, a, overwrite=True)

d, a = pipe.read("1D_freq_real.dat")
xc = p.make_uc(d, a)
d, a = p.ext(d, a, x1=xc("15693.7 Hz"), xn=xc("-1634.1 Hz"))
pipe.write("units5.glue", d, a, overwrite=True)

# PPM
d, a = pipe.read("1D_freq_real.dat")
xc = p.make_uc(d, a)
d, a = p.ext(d, a, x1=xc("120.2ppm"), xn=xc("-12.1ppm"))
pipe.write("units6.glue", d, a, overwrite=True)

d, a = pipe.read("1D_freq_real.dat")
xc = p.make_uc(d, a)
d, a = p.ext(d, a, x1=xc("67.23 ppm"), xn=xc("11.2 ppm"))
pipe.write("units7.glue", d, a, overwrite=True)

# 2D points
d, a = pipe.read("freq_real.ft2")
x1 = 193
xn = 1812
d, a = p.ext(d, a, x1=x1, xn=xn)
pipe.write("units8.glue", d, a, overwrite=True)

d, a = pipe.read("freq_real.ft2")
x1 = 679
xn = 1456
y1 = 45
yn = 80
d, a = p.ext(d, a, x1=x1, xn=xn, y1=y1, yn=yn)
pipe.write("units9.glue", d, a, overwrite=True)

d, a = pipe.read("time_complex.fid")
x1 = 189
xn = 798
d, a = p.ext(d, a, x1=x1, xn=xn)
pipe.write("units10.glue", d, a, overwrite=True)

d, a = pipe.read("time_complex.fid")
x1 = 87
xn = 991
y1 = 182
yn = 310
d, a = p.ext(d, a, x1=x1, xn=xn, y1=y1, yn=yn)
pipe.write("units11.glue", d, a, overwrite=True)

# 2D percent
d, a = pipe.read("freq_real.ft2")
xc = p.make_uc(d, a)
d, a = p.ext(d, a, x1=xc("22.4 %"), xn=xc("67.8 %"))
pipe.write("units12.glue", d, a, overwrite=True)

d, a = pipe.read("freq_real.ft2")
xc = p.make_uc(d, a)
yc = p.make_uc(d, a, dim=0)
d, a = p.ext(d, a, x1=xc("12.6%"), xn=xc("20%"), y1=yc("89%"), yn=yc("99%"))
pipe.write("units13.glue", d, a, overwrite=True)

# 2D HZ
d, a = pipe.read("freq_real.ft2")
xc = p.make_uc(d, a)
d, a = p.ext(d, a, x1=xc("13203 hz"), xn=xc("-1560 hz"))
pipe.write("units14.glue", d, a, overwrite=True)

d, a = pipe.read("freq_real.ft2")
xc = p.make_uc(d, a)
yc = p.make_uc(d, a, dim=0)
y1 = yc("1333 hz")
yn = yc("-1234 HZ")
d, a = p.ext(d, a, x1=xc("10239 Hz"), xn=xc("-19341 hZ"), y1=y1, yn=yn)
pipe.write("units15.glue", d, a, overwrite=True)

# 2D PPM
d, a = pipe.read("freq_real.ft2")
xc = p.make_uc(d, a)
d, a = p.ext(d, a, x1=xc("145.2 ppm"), xn=xc("-11.2 ppm"))
pipe.write("units16.glue", d, a, overwrite=True)

d, a = pipe.read("freq_real.ft2")
xc = p.make_uc(d, a)
y1 = yc("23.0 ppm")
yn = yc("-14.2 ppm")
d, a = p.ext(d, a, x1=xc("12.3 ppm"), xn=xc("-101.2 ppm"), y1=y1, yn=yn)
pipe.write("units17.glue", d, a, overwrite=True)

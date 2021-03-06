#! /usr/bin/env python

import nmrglue as ng

# check against a file processed with NMRPipe
dic1, data1 = ng.pipe.read("nmrpipe_2d_tppi/test.ft2")
dic2, data2 = ng.pipe.read("2d_pipe_tppi.ft2")
r1, r2 = ng.misc.pair_similar(dic1, data1, dic2, data2, verb=True)

print "==================="
print "Summary:"

if r1:
    print "Data arrays are similar."
else:
    print "Data arrays differ as listed above."

if r2:
    print "Spectral parameters are similar."
else:
    print "Spectral parameters differ as listed above."

#!/bin/csh

nmrPipe -in ./time_complex.fid                \
| nmrPipe  -fn NULL  \
-ov -out null1.dat

nmrPipe -in ./time_complex.fid                \
| nmrPipe  -fn NULL -di \
-ov -out null2.dat

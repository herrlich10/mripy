#!/bin/tcsh

set input = "$1"
set output = "$2"
set slice = "$3"

3dcalc -a "$input" -expr "$slice" -prefix tmp.mask -overwrite
3dAutobox -input tmp.mask+orig -prefix tmp.master -overwrite
3dresample -master tmp.master+orig -input "$input" -prefix "$output" -overwrite
rm tmp.*

#!/usr/bin/env bash
TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")

cd src
echo "Compiling resample2d kernels by gcc..."
rm Resample2d_kernel.o
rm -r ../_ext

gcc -c -o Resample2d_kernel.o Resample2d_kernel.c -fPIC -I ${TORCH}/lib/include/TH -Wstrict-prototypes

cd ../
python2 build.py

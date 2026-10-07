#!/bin/bash
# Build liblanesim with the project clang (LLVM FMA contraction, the closest to XLA's float results) into
# native/build-clang. Portable x86-64-v3 (AVX2/FMA): the same .so runs on the login node and the desktop.
set -e
T=${LANESIM_TOOLCHAIN:-/mnt/nfs/projects/ahriuwu-native-sim/.toolchain}
cd "$(dirname "$0")"
make -s -j"${JOBS:-6}" BUILD=build-clang CXX="$T/bin/clang++" CXXFLAGS="-O3 -march=x86-64-v3 -std=c++17 -fPIC -fopenmp \
  -I$T/include --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13 -ffp-contract=fast -Wall -Wno-unused-parameter \
  -L$T/lib -Wl,-rpath,$T/lib"

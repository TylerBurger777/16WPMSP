#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CXX=${CXX:-g++}
F="-O3 -march=znver4 -mavx512f -mavx512dq -mavx512bw -mavx512vl -std=c++17 -pthread -fopenmp -Iinclude"

[ -f external/pdqsort/pdqsort.h ]                   && F="$F -Iexternal/pdqsort"
[ -f external/ips4o/include/ips4o.hpp ]             && F="$F -Iexternal/ips4o/include"
[ -f external/ska_sort/ska_sort.hpp ]               && F="$F -Iexternal/ska_sort"
[ -f external/x86-simd-sort/src/x86simdsort-static-incl.h ] \
    && F="$F -Iexternal/x86-simd-sort/src -DX86SIMD_STATIC=1"

# IPS4o needs TBB
LDFLAGS=""
if [ -f /usr/include/tbb/concurrent_queue.h ] || [ -f /usr/include/oneapi/tbb/concurrent_queue.h ]; then
    LDFLAGS="$LDFLAGS -ltbb -latomic"
fi

# vqsort only if libhwy was built (skip by default)
[ -f external/highway/hwy/contrib/sort/vqsort.h ] && F="$F -Iexternal/highway"
if [ -f external/highway/build/libhwy_contrib.a ]; then
    LDFLAGS="$LDFLAGS -Lexternal/highway/build -lhwy_contrib -lhwy"
fi

echo ">> bench_16wpmsp"
$CXX $F -o bench_16wpmsp bench/bench_16wpmsp.cpp $LDFLAGS
echo ">> stress_16wpmsp"
$CXX $F -o stress_16wpmsp tests/stress_16wpmsp.cpp $LDFLAGS
echo Done.

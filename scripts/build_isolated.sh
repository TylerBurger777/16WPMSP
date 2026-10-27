#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CXX=${CXX:-g++}
BASE="-O3 -march=native -std=c++17 -pthread -Iinclude"
[ -f external/ips4o/include/ips4o.hpp ]             && BASE="$BASE -Iexternal/ips4o/include"
[ -f external/highway/hwy/contrib/sort/vqsort.h ]   && BASE="$BASE -Iexternal/highway"

LD="-ltbb -latomic -fopenmp"
if [ -f external/highway/build/libhwy_contrib.a ]; then
    LD="$LD -Lexternal/highway/build -lhwy_contrib -lhwy"
fi

for alg in 16wpmsp ips4o vqsort tbb; do
    case $alg in
        16wpmsp) DEF="-DALGO_16WPMSP" ;;
        ips4o)   DEF="-DALGO_IPS4O" ;;
        vqsort)  DEF="-DALGO_VQSORT" ;;
        tbb)     DEF="-DALGO_TBB" ;;
    esac
    echo ">> bench_iso_$alg"
    $CXX $BASE $DEF -o bench_iso_$alg bench/bench_algo.cpp $LD
done
echo "Done."

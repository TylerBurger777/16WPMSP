#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

ARGS="$@"
if [ -z "$ARGS" ]; then
    ARGS="--pin 100000 250000 500000 1000000 2000000 4000000 8000000 16000000 32000000 64000000"
fi

# IPS4o needs active OpenMP waiting, otherwise its thread pool parks between
# calls and the scheduler sometimes wakes threads too late (30-50 ms outliers).
OMP_ENV="OMP_WAIT_POLICY=active OMP_PROC_BIND=spread OMP_PLACES=cores OMP_DYNAMIC=FALSE"

for alg in 16wpmsp ips4o vqsort tbb; do
    echo "=== bench_iso_$alg ==="
    if [ "$alg" = "ips4o" ]; then
        eval env $OMP_ENV ./bench_iso_$alg $ARGS
    else
        ./bench_iso_$alg $ARGS
    fi
    echo
done

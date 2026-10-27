#!/bin/bash
SIZES="100000 200000 400000 800000 1600000 3200000 6400000 12800000 25600000 51200000 64000000"
THS="4000 8000 16000 24000 32000 48000 64000 96000"

echo "N,TH,Median_ms" > sweep_results.csv

for N in $SIZES; do
    for TH in $THS; do
        # Запускаем, фильтруем строку с результатом и берем медиану (3-я колонка)
        MEDIAN=$(WPMSP_FORCE_TH=$TH ./bench_final --pin $N 2>/dev/null | grep "^$N" | awk '{print $3}')
        if [ -z "$MEDIAN" ]; then
            MEDIAN="FAIL"
        fi
        echo "$N,$TH,$MEDIAN" >> sweep_results.csv
        echo "Done: N=$N, TH=$TH, Median=$MEDIAN ms"
    done
done
echo "=== СВИП ЗАВЕРШЕН ==="

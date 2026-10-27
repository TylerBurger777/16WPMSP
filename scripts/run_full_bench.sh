#!/bin/bash
cd "$(dirname "$0")/.."

# Флаги для компиляции (Zen 5)
CXXFLAGS="-O3 -march=native -std=c++20 -pthread -Iinclude -Iexternal/ips4o/include -Iexternal/highway \
    -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vbmi -mavx512vbmi2"
LDFLAGS="-ltbb -latomic -fopenmp"
HWY_LDFLAGS="-Lexternal/highway/build -lhwy_contrib -lhwy"

echo "=== Компиляция бенчмарков ==="
g++ $CXXFLAGS -o bench_iso_16wpmsp bench/bench_algo.cpp -DALGO_16WPMSP $LDFLAGS
g++ $CXXFLAGS -o bench_iso_ips4o bench/bench_algo.cpp -DALGO_IPS4O $LDFLAGS
g++ $CXXFLAGS -o bench_iso_tbb bench/bench_algo.cpp -DALGO_TBB $LDFLAGS
g++ $CXXFLAGS -o bench_iso_vqsort bench/bench_algo.cpp -DALGO_VQSORT $HWY_LDFLAGS $LDFLAGS

# Размеры для тестирования
SIZES="1000000 4000000 16000000 64000000"

echo ""
echo "=== Полный бенчмарк (Ryzen 7 9700X, Zen 5) ==="
printf "%-10s | %-12s | %-12s | %-12s | %-12s\n" "N" "16WPMSP" "IPS4o" "vqsort" "TBB"
printf "%-10s-|-%-12s-|-%-12s-|-%-12s-|-%-12s\n" "----------" "------------" "------------" "------------" "------------"

for N in $SIZES; do
    # Запускаем и берем медиану (3-я колонка)
    T_16=$(./bench_iso_16wpmsp --pin $N | grep "^$N" | awk '{print $3}')
    T_IPS=$(OMP_WAIT_POLICY=active ./bench_iso_ips4o --pin $N | grep "^$N" | awk '{print $3}')
    T_VQ=$(./bench_iso_vqsort --pin $N | grep "^$N" | awk '{print $3}')
    T_TBB=$(./bench_iso_tbb --pin $N | grep "^$N" | awk '{print $3}')

    # Форматируем вывод
    printf "%-10s | %-12s | %-12s | %-12s | %-12s\n" "$N" "$T_16" "$T_IPS" "$T_VQ" "$T_TBB"
done

echo ""
echo "=== Готово! ==="

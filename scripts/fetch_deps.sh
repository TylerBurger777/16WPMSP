#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external
c() { if [ ! -d "external/$1" ]; then echo ">> clone $1"; git clone --depth 1 "$2" "external/$1"; fi; }
c pdqsort        https://github.com/orlp/pdqsort.git
c ips4o          https://github.com/ips4o/ips4o.git
c highway        https://github.com/google/highway.git
c ska_sort       https://github.com/skarupke/ska_sort.git
c x86-simd-sort  https://github.com/intel/x86-simd-sort.git
echo Done.

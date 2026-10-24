#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
./stress_16wpmsp && echo && ./bench_16wpmsp

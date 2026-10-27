# quill-fastsort-simd

Single-threaded radix sort backend for [quill-sort](https://pypi.org/project/quill-sort/).
Portable **C++17** build, compiling natively on **macOS (arm64/x86_64),
Linux (x86_64/aarch64) and Windows**.

Provides the entry points `quill._backends.SimdCompanionBackend` (backend name
`simd_companion`) probes for:

```
sort_i64  sort_u64  sort_i32
sort_u32  sort_f64  sort_f32
```

Each sorts a contiguous, writable numpy buffer of the matching dtype **in place,
ascending**. The kernel is `quillcore::serial_sort` — a single-threaded 8-bit LSD
radix sort. It is the SIMD-friendly companion with a conservative `min_n` in
quill, so small arrays fall back to the dispatcher's default path.

* **Never-lose:** results are identical to `np.sort` on NaN-free numeric data
  (verified exhaustively against `np.sort` across dtypes, sizes and adversarial
  distributions). NaN is stripped by the Quill dispatcher before any kernel runs.

## Build

```bash
pip install .            # needs only a C++17 compiler
```

The sort kernels are header-only (`src/quill_core.hpp`, a synced copy of
`companions/_core/quill_core.hpp` — run `companions/sync_core.sh` after editing
the canonical core). No numpy headers or third-party libraries are required.

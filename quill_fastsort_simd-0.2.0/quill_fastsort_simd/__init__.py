"""quill-fastsort-simd — single-threaded radix sort backend for quill-sort.

Provides the ``sort_{i64,u64,i32,u32,f64,f32}`` entry points that
``quill._backends.SimdCompanionBackend`` (backend name ``simd_companion``)
probes for. Each sorts a contiguous, writable numpy buffer of the matching
dtype in place, ascending.

Portable C++17 build (macOS / Linux / Windows). The kernel is a single-threaded
radix sort — a SIMD-friendly companion with a conservative ``min_n`` in quill —
with the same never-lose contract as the other Quill companions.
"""
from ._impl import (  # noqa: F401
    sort_i64,
    sort_u64,
    sort_i32,
    sort_u32,
    sort_f64,
    sort_f32,
)

__all__ = [
    "sort_i64", "sort_u64", "sort_i32",
    "sort_u32", "sort_f64", "sort_f32",
]
__version__ = "0.2.0"

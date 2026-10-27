"""Portable build for quill-fastsort-simd.

Builds the ``quill_fastsort_simd._impl`` C++ extension on macOS, Linux and
Windows. The sort kernels are header-only (``src/quill_core.hpp``, a synced copy
of ``companions/_core/quill_core.hpp``), so the only build requirement is a
C++17 compiler — no numpy headers, no third-party libraries.
"""
from __future__ import annotations

import sys
from setuptools import setup, Extension

if sys.platform == "win32":
    extra_compile_args = ["/std:c++17", "/O2", "/EHsc"]
    extra_link_args: list[str] = []
else:
    # -pthread for std::thread; -O3 for the radix/scatter hot loops.
    extra_compile_args = ["-std=c++17", "-O3", "-pthread"]
    extra_link_args = ["-pthread"]

ext = Extension(
    "quill_fastsort_simd._impl",
    sources=["src/_impl.cpp"],
    include_dirs=["src"],
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(ext_modules=[ext])

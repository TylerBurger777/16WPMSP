// ===========================================================================
//
// SIMDAlloc.H --
// allocation code
//
// This source code file is part of the following software:
//
//    - the low-level C++ template SIMD library
//    - the SIMD implementation of the MinWarping and the 2D-Warping methods
//      for local visual homing.
//
// The software is provided based on the accompanying license agreement in the
// file LICENSE.md.
// The software is provided "as is" without any warranty by the licensor and
// without any liability of the licensor, and the software may not be
// distributed by the licensee; see the license agreement for details.
//
// (C) Ralf Möller
//     Computer Engineering
//     Faculty of Technology
//     Bielefeld University
//     www.ti.uni-bielefeld.de
//
// ===========================================================================

// 30. Aug 22 (Jonas Keller): added simd_aligned_malloc and simd_aligned_free

// 02. Mar 23 (Jonas Keller): added doxygen documentation

/**
 * @file SIMDAlloc.H
 * @brief Aligned memory allocation and deallocation.
 *
 * This file contains functions for aligned memory allocation and deallocation.
 *
 * This file is standalone, i.e. it can also be used independently of T-SIMD.
 *
 * @author Ralf Möller
 * @author Jonas Keller
 */

#pragma once
#ifndef SIMDALLOC_H_
#define SIMDALLOC_H_

#include <cstddef>
#include <utility>

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

/**
 * @ingroup group_aligned_alloc
 * @brief Aligned memory allocation.
 *
 * This function allocates a block of memory of size bytes, aligned to the
 * specified alignment.
 *
 * The allocated memory must be freed with simd_aligned_free().
 *
 * @param alignment alignment of the memory block in bytes
 * @param size size of the memory block in bytes
 * @return pointer to the allocated memory block
 */
inline void *simd_aligned_malloc(size_t alignment, size_t size)
{
#ifdef _WIN32
  return _aligned_malloc(size, alignment);
#else
  void *ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size) != 0) { return nullptr; }
  return ptr;
#endif
}

/**
 * @ingroup group_aligned_alloc
 * @brief Aligned memory deallocation.
 *
 * This function frees a block of memory that was allocated with
 * simd_aligned_malloc().
 *
 * @param ptr pointer to the memory block to be freed
 */
inline void simd_aligned_free(void *ptr)
{
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

// 05. Sep 23 (Jonas Keller): added simd_aligned_allocator

/**
 * @ingroup group_aligned_alloc
 * @brief Aligned allocator.
 *
 * This class is an allocator that allocates aligned memory blocks.
 *
 * This class is meant to be used with std::vector for types that require
 * aligned memory blocks.
 *
 * @tparam T type of the elements in the memory block
 * @tparam ALIGN alignment of the memory block in bytes
 */
template <typename T, size_t ALIGN>
class simd_aligned_allocator
{
  // exclude from doxygen (until endcond)
  /// @cond
public:
  using value_type      = T;
  using pointer         = T *;
  using const_pointer   = const T *;
  using reference       = T &;
  using const_reference = const T &;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <typename U>
  struct rebind
  {
    using other = simd_aligned_allocator<U, ALIGN>;
  };

  simd_aligned_allocator() noexcept {}
  simd_aligned_allocator(const simd_aligned_allocator &) noexcept {}
  template <typename U>
  simd_aligned_allocator(const simd_aligned_allocator<U, ALIGN> &) noexcept
  {}
  ~simd_aligned_allocator() noexcept {}

  pointer address(reference x) const noexcept { return std::addressof(x); }
  const_pointer address(const_reference x) const noexcept
  {
    return std::addressof(x);
  }

  pointer allocate(size_type n, const void * = 0)
  {
    return static_cast<pointer>(simd_aligned_malloc(ALIGN, n * sizeof(T)));
  }
  void deallocate(pointer p, size_type) { simd_aligned_free(p); }

  size_type max_size() const noexcept
  {
    return (size_type(-1) - size_type(ALIGN)) / sizeof(T);
  }
  template <typename U, typename... Args>
  void construct(U *p, Args &&...args)
  {
    ::new (static_cast<void *>(p)) U(std::forward<Args>(args)...);
  }
  void destroy(pointer p) { p->~T(); }

  bool operator==(const simd_aligned_allocator &) const { return true; }
  bool operator!=(const simd_aligned_allocator &) const { return false; }
  /// @endcond
};

#endif // SIMDALLOC_H_

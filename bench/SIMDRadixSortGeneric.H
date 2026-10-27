// ===========================================================================
//
// SIMDRadixSortGeneric.H --
// generic implementation of bitwise MSB radix sort for AVX-512
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

// NOTES:
//
// - This implementation is stand-alone and does not depend on files
//   from the T-SIMD low-level C++ template library. I decided to do
//   it stand-alone since the required instructions are only available
//   on AVX-512, but not on earlier vector extensions. Moreover, some
//   element types required here (such as uint64_t) are not supported
//   by the T-SIMD library.
//
// - I prefer const-ref arguments instead of value arguments in order to
//   avoid implicit type casts.

#pragma once
#ifndef SIMD_RADIX_SORT_GENERIC_H_
#define SIMD_RADIX_SORT_GENERIC_H_

#include <cstdint>
#include <cstdio>
#include <x86intrin.h>

// std::swap
#include <algorithm>
// static_assert, is_floating_point, is_signed
#include <type_traits>

// for insertion sort
#include <cstdlib>
#include <cstring>
#include <sys/types.h>

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
#define SIMD_RADIX_HAS_AVX512
#endif

namespace radix {

// =========================================================================
// definitions
// =========================================================================

// g++, clang++; replace for other compiler
#define INLINE inline __attribute__((always_inline))

// we use signed to avoid problems with index arithmetics
using SortIndex = int64_t;

// =========================================================================
// unsigned int type with template parameter BYTES
// =========================================================================

// emulate 128bit type
struct uint128_t
{
  uint64_t half[2];

  uint128_t() = default;

  uint128_t(int x)
  {
    half[1] = 0;
    half[0] = x;
  }

  uint128_t(const uint128_t &v)
  {
    half[0] = v.half[0];
    half[1] = v.half[1];
  }

  uint128_t &operator=(const uint128_t &v)
  {
    half[0] = v.half[0];
    half[1] = v.half[1];
    return *this;
  }

  bool operator==(const uint128_t &v) const
  {
    return (half[0] == v.half[0]) && (half[1] == v.half[1]);
  }

  bool operator!=(const uint128_t &v) const { return !(*this == v); }
};

uint128_t operator&(const uint128_t &a, const uint128_t &b)
{
  uint128_t v;
  v.half[0] = a.half[0] & b.half[0];
  v.half[1] = a.half[1] & b.half[1];
  return v;
}

// UInt: T is unsigned int type of given size, T2 of double size
template <int BYTES>
struct UInt;
template <>
struct UInt<8>
{
  using T  = uint64_t;
  using T2 = uint128_t;
};
template <>
struct UInt<4>
{
  using T  = uint32_t;
  using T2 = uint64_t;
};
template <>
struct UInt<2>
{
  using T  = uint16_t;
  using T2 = uint32_t;
};
template <>
struct UInt<1>
{
  using T  = uint8_t;
  using T2 = uint16_t;
};

// =========================================================================
// set single bit from bit no.
// =========================================================================

// all uint types from stdint.h
template <typename T>
static INLINE void setBitNo(T &v, int bitNo)
{
  v = (T(1) << bitNo);
}

static INLINE void setBitNo(uint128_t &v, int bitNo)
{
  if (bitNo < 64) {
    v.half[0] = uint64_t(1) << bitNo;
    v.half[1] = 0;
  } else {
    v.half[0] = 0;
    v.half[1] = uint64_t(1) << (bitNo - 64);
  }
}

// =========================================================================
// information on bit range and type
// =========================================================================

template <typename KEYTYPE>
struct BitRange
{
  static constexpr int msb = sizeof(KEYTYPE) * 8 - 1;
  static constexpr int lsb = 0;
};

template <typename KEYTYPE, bool WithPayload>
struct KeyPayloadInfo;

// we have a key that fills the entire element, no payload
template <typename KEYTYPE>
struct KeyPayloadInfo<KEYTYPE, false>
{
  using UIntKeyType     = typename UInt<sizeof(KEYTYPE)>::T;
  using UIntElementType = typename UInt<sizeof(KEYTYPE)>::T;
};

// the element comprises a key and a payload
// assumes that key and payload have the same size
// TODO: solution for small keys and long payloads???
template <typename KEYTYPE>
struct KeyPayloadInfo<KEYTYPE, true>
{
  using UIntKeyType     = typename UInt<sizeof(KEYTYPE)>::T;
  using UIntPayloadType = typename UInt<sizeof(KEYTYPE)>::T;
  using UIntElementType = typename UInt<sizeof(KEYTYPE)>::T2;
};

// =========================================================================
// access to data
// =========================================================================

// the only legal way of type punning is memcpy(), see
// https://gist.github.com/shafik/848ae25ee209f698763cffee272a58f8

// TODO: move these functions to application? only getKey is used here

// with payload: set payload component (high part of element)
template <typename KEYTYPE>
static INLINE void setPayload(
  typename KeyPayloadInfo<KEYTYPE, true>::UIntElementType &element,
  const typename KeyPayloadInfo<KEYTYPE, true>::UIntPayloadType &payload)
{
  memcpy((void *) (((uint8_t *) &element) + sizeof(KEYTYPE)), (void *) &payload,
         sizeof(payload));
}

// with payload: get payload component (high part of element)
template <typename KEYTYPE>
static INLINE void getPayload(
  const typename KeyPayloadInfo<KEYTYPE, true>::UIntElementType &element,
  typename KeyPayloadInfo<KEYTYPE, true>::UIntPayloadType &payload)
{
  memcpy((void *) &payload, (void *) (((uint8_t *) &element) + sizeof(KEYTYPE)),
         sizeof(payload));
}

// with our without payload (free ELEMENTTYPE)
// assumes that the low portion of data type T is the key
// asserts that ELEMENTTYPE is not smaller than KEYTYPE
template <typename KEYTYPE, typename ELEMENTTYPE>
static INLINE void setKey(const KEYTYPE &key, ELEMENTTYPE &element)
{
  static_assert(sizeof(KEYTYPE) <= sizeof(ELEMENTTYPE),
                "key type is larger than element type");
  memcpy((void *) &element, (void *) &key, sizeof(KEYTYPE));
}

// with or without payload (free ELEMENTTYPE)
// assumes that the low portion of data type T is the key
// asserts that ELEMENTTYPE is not smaller than KEYTYPE
template <typename KEYTYPE, typename ELEMENTTYPE>
static INLINE KEYTYPE getKey(const ELEMENTTYPE &element)
{
  static_assert(sizeof(KEYTYPE) <= sizeof(ELEMENTTYPE),
                "key type is larger than element type");
  KEYTYPE key;
  memcpy((void *) &key, (void *) &element, sizeof(KEYTYPE));
  return key;
}

// =========================================================================
// generic AVX-512 SIMD code
// =========================================================================

#ifdef SIMD_RADIX_HAS_AVX512

// TODO: some parts done via preprocessor macros could also be
// TODO: implemented using templates

// elegant idea for constructor, operator=, and conversion operator
// in SIMDVector and BitMask taken from Agner Fog's C++ Vector
// Class Library http://www.agner.org/optimize/#vectorclass (VCL
// License: GNU General Public License Version 3,
// http://www.gnu.org/licenses/gpl-3.0.en.html)

// -------------------------------------------------------------------------
// SIMDVector
// -------------------------------------------------------------------------

template <typename T>
struct SIMDVector
{
  using Type = T;
  __m512i zmm;
  SIMDVector() = default;
  SIMDVector(const __m512i &x) { zmm = x; }
  SIMDVector &operator=(const __m512i &x)
  {
    zmm = x;
    return *this;
  }
  operator __m512i() const { return zmm; }
};

// -------------------------------------------------------------------------
// BitMask
// -------------------------------------------------------------------------

template <typename T>
struct BitMask;

#define BITMASK(TYPE, MASKTYPE)                                                \
  template <>                                                                  \
  struct BitMask<TYPE>                                                         \
  {                                                                            \
    using Type     = TYPE;                                                     \
    using MaskType = MASKTYPE;                                                 \
    MASKTYPE k;                                                                \
    BitMask() {}                                                               \
    BitMask(const MASKTYPE &x)                                                 \
    {                                                                          \
      k = x;                                                                   \
    }                                                                          \
    BitMask &operator=(const MASKTYPE &x)                                      \
    {                                                                          \
      k = x;                                                                   \
      return *this;                                                            \
    }                                                                          \
    operator MASKTYPE() const                                                  \
    {                                                                          \
      return k;                                                                \
    }                                                                          \
  };

BITMASK(uint128_t, __mmask8) // emulated
BITMASK(uint64_t, __mmask8)
BITMASK(uint32_t, __mmask16)
BITMASK(uint16_t, __mmask32)
BITMASK(uint8_t, __mmask64)

// -------------------------------------------------------------------------
// bitMaskNot
// -------------------------------------------------------------------------

#define BITMASK_NOT(TYPE, NOTFCT)                                              \
  static INLINE BitMask<TYPE> bitMaskNot(const BitMask<TYPE> &bm)              \
  {                                                                            \
    return NOTFCT(bm);                                                         \
  }

BITMASK_NOT(uint128_t, _knot_mask8) // DQ, emulated
BITMASK_NOT(uint64_t, _knot_mask8)  // DQ
BITMASK_NOT(uint32_t, _knot_mask16) // F
BITMASK_NOT(uint16_t, _knot_mask32) // BW
BITMASK_NOT(uint8_t, _knot_mask64)  // BW

// -------------------------------------------------------------------------
// bitMaskPopCnt
// -------------------------------------------------------------------------

// was easier without macro (would require 3 arguments)

static INLINE SortIndex bitMaskPopCnt(const BitMask<uint128_t> &bm)
{
  return _popcnt32(_cvtmask8_u32(bm)) >> 1;
} // DQ, POPCNT, emulated

static INLINE SortIndex bitMaskPopCnt(const BitMask<uint64_t> &bm)
{
  return _popcnt32(_cvtmask8_u32(bm));
} // DQ, POPCNT

static INLINE SortIndex bitMaskPopCnt(const BitMask<uint32_t> &bm)
{
  return _popcnt32(_cvtmask16_u32(bm));
} // F, POPCNT

static INLINE SortIndex bitMaskPopCnt(const BitMask<uint16_t> &bm)
{
  return _popcnt32(_cvtmask32_u32(bm));
} // BW, POPCNT

static INLINE SortIndex bitMaskPopCnt(const BitMask<uint8_t> &bm)
{
  return _popcnt64(_cvtmask64_u64(bm));
} // BW, POPCNT

// -------------------------------------------------------------------------
// test_mask
// -------------------------------------------------------------------------

#define TEST_MASK(TYPE, TESTFCT)                                               \
  static INLINE BitMask<TYPE> test_mask(const SIMDVector<TYPE> &a,             \
                                        const SIMDVector<TYPE> &b)             \
  {                                                                            \
    return TESTFCT(a, b);                                                      \
  }

TEST_MASK(uint64_t, _mm512_test_epi64_mask) // F
TEST_MASK(uint32_t, _mm512_test_epi32_mask) // F
TEST_MASK(uint16_t, _mm512_test_epi16_mask) // BW
TEST_MASK(uint8_t, _mm512_test_epi8_mask)   // BW

// emulation
static INLINE BitMask<uint128_t> test_mask(const SIMDVector<uint128_t> &a,
                                           const SIMDVector<uint128_t> &b)
{
  // here we can't avoid to also test the payloads (higher parts)
  __mmask8 k = _mm512_test_epi64_mask(a, b); // F
  // so we manipulate the mask: leave only key maskbits (01010101B = 0x55)
  // mask from set1 produces zero maskbits for payload, so no
  // "and" necessary?
  // k = _kand_mask8(k, _cvtu32_mask8(0x55)); // DQ, DQ
  // duplicate key maskbits to payload maskbits
  // 0A0B0C0D
  // A0B0C0D0
  // AABBCCDD
  // we could also add the mask 3 times? (01 + 01 + 01 = 11)
  return _kor_mask8(k, _kshiftli_mask8(k, 1)); // DQ, DQ
}

// -------------------------------------------------------------------------
// loadu
// -------------------------------------------------------------------------

// for all integer types
template <typename T>
static INLINE SIMDVector<T> loadu(const T *const p)
{
  return _mm512_loadu_si512((void *) p); // F
}

// -------------------------------------------------------------------------
// mask_compressstoreu
// -------------------------------------------------------------------------

#define MASK_COMPRESSSTOREU(TYPE, COMPRESSFCT)                                 \
  static INLINE void mask_compressstoreu(                                      \
    const TYPE *const p, const BitMask<TYPE> &bm, const SIMDVector<TYPE> &v)   \
  {                                                                            \
    COMPRESSFCT((void *) p, bm, v);                                            \
  }

MASK_COMPRESSSTOREU(uint128_t, _mm512_mask_compressstoreu_epi64) // F, emul.
MASK_COMPRESSSTOREU(uint64_t, _mm512_mask_compressstoreu_epi64)  // F
MASK_COMPRESSSTOREU(uint32_t, _mm512_mask_compressstoreu_epi32)  // F
#ifdef __AVX512VBMI2__
MASK_COMPRESSSTOREU(uint16_t, _mm512_mask_compressstoreu_epi16) // VBMI2
MASK_COMPRESSSTOREU(uint8_t, _mm512_mask_compressstoreu_epi8)   // VBMI2
#endif

// -------------------------------------------------------------------------
// set1
// -------------------------------------------------------------------------

#define SET1(TYPE, SET1FCT)                                                    \
  static INLINE SIMDVector<TYPE> set1(const TYPE &a)                           \
  {                                                                            \
    return SET1FCT(a);                                                         \
  }

SET1(uint64_t, _mm512_set1_epi64) // F
SET1(uint32_t, _mm512_set1_epi32) // F
SET1(uint16_t, _mm512_set1_epi16) // F
SET1(uint8_t, _mm512_set1_epi8)   // F

// emulation
static INLINE SIMDVector<uint128_t> set1(const uint128_t &a)
{
  // H                           L
  // a0 a0 | a0 a0 | a0 a0 | a0 a0 (set1)
  // a1 a1 | a1 a1 | a1 a1 | a1 a1 (set1)
  //    --      --      --      --
  // a1 a0 | a1 a0 | a1 a0 | a1 a0 (unpack_lo)
  return _mm512_unpacklo_epi64(_mm512_set1_epi64(a.half[0]),
                               _mm512_set1_epi64(a.half[1])); // F, F, F
}

#endif // SIMD_RADIX_HAS_AVX512

// =========================================================================
// sequential radix sort
// =========================================================================

// -------------------------------------------------------------------------
// TestCondition
// -------------------------------------------------------------------------

// support for upward and downward sorting
// isZero function needs to be wrapped in a struct since partial
// specialization of template functions is not supported in C++

template <int UP>
struct TestCondition;

template <>
struct TestCondition<1>
{
  template <typename T>
  static INLINE bool isZero(const T &cond)
  {
    return (cond == T(0));
  }
};

template <>
struct TestCondition<0>
{
  template <typename T>
  static INLINE bool isZero(const T &cond)
  {
    return (cond != T(0));
  }
};

// -------------------------------------------------------------------------
// SeqRadixBitSorter
// -------------------------------------------------------------------------

// bit sorter for sequential radix sort

// UP = 1: move 0s to left part, 1s to right part
// UP = 0: move 1s to left part, 0s to right part

template <int UP, typename T>
struct SeqRadixBitSorter
{
  static INLINE SortIndex bitSorter(T *d, int bitNo, SortIndex left,
                                    SortIndex right)
  {
    SortIndex l = left, r = right;
    T bitMask;
    setBitNo(bitMask, bitNo);
    while (true) {
      // advance left index
      while ((l <= r) && TestCondition<UP>::isZero(d[l] & bitMask)) l++;
      // advance right index
      while ((l <= r) && !TestCondition<UP>::isZero(d[r] & bitMask)) r--;
      // cross-over of indices -> end
      if (l > r) break;
      // swap (key and payload)
      std::swap(d[l], d[r]);
    }
    // at this point l = r + 1 (crossed over rl)
    return l;
  }
};

// -------------------------------------------------------------------------
// SeqRadixBitSorter2
// -------------------------------------------------------------------------

// experimental, swap replaced, not faster

template <int UP, typename T>
struct SeqRadixBitSorter2
{
  static INLINE SortIndex bitSorter(T *d, int bitNo, SortIndex left,
                                    SortIndex right)
  {
    SortIndex l = left, r = right;
    T bitMask, dl, dr;
    setBitNo(bitMask, bitNo);
    while (true) {
      // advance left index
      while ((l <= r) && TestCondition<UP>::isZero((dl = d[l]) & bitMask)) l++;
      // advance right index
      while ((l <= r) && !TestCondition<UP>::isZero((dr = d[r]) & bitMask)) r--;
      // cross-over of indices -> end
      if (l > r) break;
      // swap (key and payload) without std::swap
      d[l] = dr;
      d[r] = dl;
    }
    // at this point l = r + 1 (crossed over rl)
    return l;
  }
};

// -------------------------------------------------------------------------
// SeqRadixBitSorterRightLimit
// -------------------------------------------------------------------------

// right index only goes down to minRight;
// can be used if part from left to minRight-1 is already sorted
// prevents going through entire range (not necessary);
// used in SIMD implementation

// TODO: if we would have the "sequential part" in the middle
// TODO: between two "SIMD parts", we could use SeqRadixBitSorter instead?

template <int UP, typename T>
struct SeqRadixBitSorterRightLimit
{
  static INLINE SortIndex bitSorter(T *d, int bitNo, SortIndex left,
                                    SortIndex minRight, SortIndex right)
  {
    SortIndex l = left, r = right;
    T bitMask;
    setBitNo(bitMask, bitNo);
    while (true) {
      // advance left index
      while ((l <= r) && TestCondition<UP>::isZero(d[l] & bitMask)) l++;
      // advance right index (but not below minRight)
      // (NOTE: l can go above minRight if sorted part is homogenous)
      while ((l <= r) && (minRight <= r) &&
             !TestCondition<UP>::isZero(d[r] & bitMask))
        r--;
      // cross-over of indices or right limit reached -> end
      // (NOTE: l and r can cross over withouth right limit being reached)
      if ((l > r) || (minRight > r)) break;
      // swap (key and payload)
      std::swap(d[l], d[r]);
    }
    // at this point l = r + 1 (crossed over rl)
    return l;
  }
};

// =========================================================================
// baseline radix sort: no sorting at all
// =========================================================================

template <int UP, typename T>
struct BaselineRadixBitSorter
{
  static INLINE SortIndex bitSorter(T * /* d */, int, SortIndex left,
                                    SortIndex right)
  {
    // std::swap(d[left], d[right]);
    return (left + right) / 2;
  }
};

// =========================================================================
// SIMD radix sort
// =========================================================================

#ifdef SIMD_RADIX_HAS_AVX512

// -------------------------------------------------------------------------
// SIMD bit sorter based on compressstoreu
// -------------------------------------------------------------------------

template <int UP, typename T>
struct SimdRadixBitSorterCompress
{
  static constexpr SortIndex numElems = 64 / sizeof(T);
  // afterRightBlockIndex:
  // compute index immediately to the right of the last full SIMD block
  //
  // examples: for 8-block (numElems = 8)
  // no full SIMD 8-block:
  // left = 5, right = 6
  // (5 + (((6 + 1) - 5) & ~0x07))
  // (5 + (2 & ~0x07))
  // 5
  // no full SIMD 8-block:
  // left = 5, right = 9
  // (5 + (((9 + 1) - 5) & ~0x07))
  // (5 + (5 & ~0x07))
  // 5
  // just 1 SIMD 8-block:
  // left = 5, right = 12
  // (5 + (((12 + 1) - 5) & ~0x07))
  // (5 + (8 & ~0x07))
  // 13
  //
  static INLINE SortIndex afterRightBlockIndex(SortIndex left, SortIndex right)
  {
    return (left + (((right + 1) - left) & ~(numElems - 1)));
  }

  // testAndCount:
  // test relevant bits, produce two masks and two bit counts of
  // the bits for which compressstoreu will write data
  //
  // examples:
  // e.g. UP=1, vectors shown sorted for clarity
  // underlined: already read into vector store, can be overwritten
  //
  // 0000 0111 | 0011 1111 | 0001 1111 | 0000 0001
  // ---------   ^pos[0]                 ^pos[1]
  // vector store (0000 0111)
  // left side can be stored, right side needs a load before
  //
  // 0000 0xxx | 0011 1111 | 0001 1111 | xxxx x111
  //             ^pos[0]     ^pos[1]     ---------
  // vector store (0000 0001)
  // right side can be stored, left side needs a load before
  //
  // 0000 0000 | 0000 xxxx | 0001 1111 | xxxx 1111
  //             ---------   ^pos[0,1]
  // vector store (0011 1111)
  // left side can be stored, right side needs a load before
  //
  // 0000 0000 | 0000 00xx | xxxx xx11 | 1111 1111
  // vector store (0001 1111)
  // end of loop
  //
  // postprocessing: insert element from vector store (2x compress)
  // 0000 0000 | 0000 0000 | 0111 1111 | 1111 1111
  //
  // can both sides need a load?
  // no, there are always 8 free (x) elements, only one side needs reload
  // but it can be that the free elements on both sides are exactly
  // sufficient to store the new bits (in this case we enforce a load, see
  // below)
  //
  // assignment of sortBits and popcnt to both sides ([0] left, [1] right):
  // e.g. mask bits: 0100 1001
  // UP=1                       UP=0
  // sortBits[1] = 0100 1001    sortBits[0] = 0100 1001
  // sortBits[0] = 1011 0110    sortBits[1] = 1011 0110
  // popcnt[1] = 3              popcnt[0] = 3
  // popcnt[0] = 5              popcnt[1] = 5
  // compressed w. sortBits[1]  compressed w. sortBits[0]
  // 3 1-bits stored at side 1  3 1-bits stored at side 0
  // compressed w. sortBits[0]  compressed w. sortBits[1]
  // 5 0-bits stored at side 0  5 0-bits stored at side 1
  //
  static INLINE void testAndCount(const SIMDVector<T> &bitMaskVec,
                                  const SIMDVector<T> &keyPayload,
                                  BitMask<T> sortBits[2], SortIndex popcnt[2])
  {
    sortBits[UP]     = test_mask(keyPayload, bitMaskVec);
    sortBits[1 - UP] = bitMaskNot(sortBits[UP]);
    popcnt[UP]       = bitMaskPopCnt(sortBits[UP]);
    popcnt[1 - UP]   = numElems - popcnt[UP];
  }

  static INLINE SortIndex bitSorter(T *d, int bitNo, SortIndex left,
                                    SortIndex right)
  {
    T bitMask;
    setBitNo(bitMask, bitNo);
    SIMDVector<T> bitMaskVec = set1(bitMask);
    // vector store and currently processed element (key and payload)
    SIMDVector<T> vectorStore, keyPayload;
    // read and write positions, popcnt, start of sequential part (both sides)
    SortIndex readPos[2], writePos[2], popcnt[2], posSeq;
    // relevant bits (both sides)
    BitMask<T> sortBits[2];
    // 0: load from left side, 1: load from right side
    int sideToLoad;
    // read positions:
    // readPos[0]: position at SIMD block to be read next
    // readPos[1]: position after SIMD block to be read next
    // posSeq is the start of the sequential part
    // write positions:
    // writePos[0]: next element to write
    // writePos[1]: last element written
    readPos[0] = writePos[0] = left;
    readPos[1] = writePos[1] = posSeq = afterRightBlockIndex(left, right);
    // initial state before preload
    // # = sequential part
    //           ###
    //           r0
    //           w0
    //           r1
    //           w1
    //
    // |10101111|###
    //  r0       r1
    //  w0       w1
    //
    // |10101111|01010111|###
    //  r0                r1
    //  w0                w1
    //
    // at least one SIMD vector loadable?
    // even if loop is not entered, we have a preloaded vectorStore
    if (readPos[0] < readPos[1])
      // preload from right side to vectorStore
      vectorStore = loadu(d + readPos[1] - numElems);
    // position needs to be changed even if no parallel processing
    // takes place, otherwise the purely sequential case would be
    // different from the other cases with respect to comparison of
    // pos[0] and pos[1]
    readPos[1] -= numElems;
    // initial state after preload:
    // x = preloaded, free for storing
    //           ###
    //           r0
    //           w0
    //  r1
    //           w1
    //
    // |xxxxxxxx|###
    //  r0
    //  w0
    //  r1       w1
    //
    // |10101100|xxxxxxxx|###
    //  r0       r1       w1
    //  w0
    //
    // |10101100|00101111|01010101|xxxxxxxx|###
    //  r0                         r1       w1
    //  w0
    //
    // loop while there's a SIMD block which has not yet been loaded
    while (readPos[0] < readPos[1]) {
      // copy element from vectorStore (vectorStore is now "free" for load)
      keyPayload = vectorStore;
      // test bits and count
      testAndCount(bitMaskVec, keyPayload, sortBits, popcnt);
      // find out on which side additional free space is needed to
      // store the sorted (compressed) data
      // x: area was read but not yet overwritten
      // 0,1: unread data, already stored data (for UP=1)
      //
      // not needed, now in sideToLoad
      // bool needsLoad[2];
      //
      // left side:
      // |000xxxxx|10001011|      |000xxxx|10001011|01001011|
      //     w     r                  w    r
      //     000                      0000 00
      // |000000xx|10001011|      |0000000|00xxxxxx|01001011|
      //        w  r                         w      r
      // not needed, see below:
      // needsLoad[0] = ((writePos[0] + popcnt[0]) > readPos[0]);
      //
      // right side:
      // |11100011|10100011|xxxxx111|  |11100011|10100011|xxxxx111|
      //                    r    w                        r    w
      //                     1111                      11 11111
      // |11100011|10000011|x1111111|  |11100011|xxxxxx11|11111111|
      //                    rw                   r     w
      // just single variable sideToLoad sufficient:
      // needsLoad[1] = ((writePos[1] - popcnt[1]) < readPos[1]);
      sideToLoad = ((writePos[1] - popcnt[1]) < readPos[1]);
      // e.g. numElems = 8:
      // we preloaded 8, we write 8 and load one side again: always 8 free
      // we want to write 8, one side has enough space, other reloaded
      // (reload is never needed on both sides);
      // e.g.: free: left 3, right 5; to store: left 5, right 3
      // -> store left, reload right, store right
      //
      // but: we have to force one side to reload:
      // e.g.: free: left 3, right 5, to store: left 3, right 5,
      // enough space on both sides, but one side has to reload regardless
      // otherwise we can't store in the next step
      //
      // nl[0] nl[1] sideToLoad (2 versions)
      // 0     0     1       0   we are free to chose which side to load
      // 0     1     1       1   right side needs to be loaded
      // 1     0     0       0   left side needs to be loaded
      // 1     1     *       *   should never happen
      // Karnaugh plans (2 versions):
      // nl[0]\nl[1]   0 1     0 1
      //             0 1 1   0 0 1
      //             1 0 *   1 0 *
      // sideToLoad: 1-nl[0] nl[1]  second version is used
      //
      // store bits to both sides, preload if necessary
      // only one side is preloaded
      //
      // left side:
      if (/*needsLoad[0]*/ !sideToLoad) {
        vectorStore = loadu(d + readPos[0]);
        readPos[0] += numElems;
      }
      mask_compressstoreu(d + writePos[0], sortBits[0], keyPayload);
      writePos[0] += popcnt[0];
      // right side
      if (/*needsLoad[1]*/ sideToLoad) {
        readPos[1] -= numElems;
        vectorStore = loadu(d + readPos[1]);
      }
      writePos[1] -= popcnt[1];
      mask_compressstoreu(d + writePos[1], sortBits[1], keyPayload);
    }
    // example: vector with 4 elements
    //
    // r0                                      r1
    // 1 0 1 1 | 0 1 0 1 | 1 1 1 0 | 0 0 0 1 | 1 0 0
    // w0                                      w1
    //
    // r0                            r1
    // 1 0 1 1 | 0 1 0 1 | 1 1 1 0 | x x x x | 1 0 0        vs: 0 0 0 1
    // w0                                      w1
    //
    //           r0                  r1
    // 0 0 0 x | 0 1 0 1 | 1 1 1 0 | x x x 1 | 1 0 0        vs: 1 0 1 1
    //       w0                            w1
    //
    //                     r0        r1
    // 0 0 0 0 | x x x x | 1 1 1 0 | 1 1 1 1 | 1 0 0        vs: 0 1 0 1
    //           w0                  w1
    //
    //                     r1
    //                     r0
    // 0 0 0 0 | 0 0 x x | x x 1 1 | 1 1 1 1 | 1 0 0        vs: 1 1 1 0
    //               w0        w1
    //
    // postamble:
    //                     r1
    //                     r0
    // 0 0 0 0 | 0 0 0 1 | 1 1 1 1 | 1 1 1 1 | 1 0 0        vs: 1 1 1 0
    //                 w0
    //                 w1
    //
    // do we have one unprocessed vector in vectorStore?
    if (readPos[0] == readPos[1]) {
      // test bits and count
      testAndCount(bitMaskVec, vectorStore, sortBits, popcnt);
      // store bits to both sides (no preload)
      // left side
      mask_compressstoreu(d + writePos[0], sortBits[0], vectorStore);
      writePos[0] += popcnt[0];
      // right side
      writePos[1] -= popcnt[1];
      mask_compressstoreu(d + writePos[1], sortBits[1], vectorStore);
    }
    SortIndex split = SeqRadixBitSorterRightLimit<UP, T>::bitSorter(
      d, bitNo, writePos[0], posSeq, right);
    return split;
  }
}; // struct SimdRadixBitSorterCompress8Intrin

#endif // SIMD_RADIX_HAS_AVX512

// =========================================================================
// compare function for std::sort and sort check
// =========================================================================

template <typename KEYTYPE, int UP, typename ELEMENTTYPE>
static INLINE bool compareKeys(const ELEMENTTYPE &ae, const ELEMENTTYPE &be)
{
  // hopefully efficiently resolved by compiler
  KEYTYPE ak = getKey<KEYTYPE>(ae);
  KEYTYPE bk = getKey<KEYTYPE>(be);
  // hopefully efficiently resolved by compiler
  return UP ? (ak < bk) : (ak > bk);
}

// =========================================================================
// comparison sorter
// =========================================================================

// insertion sort, modified from Heineman et al. "Algorithms in a Nutshell",
// 2nd ed., p.59, permission granted (see book page X)
// original code taken from:
// https://github.com/heineman/algorithms-nutshell-2ed/blob/master/Code/
// Sorting/ValueBased/insertion.c

template <typename KEYTYPE, int UP, typename T>
class InsertionSort
{
protected:
  static INLINE bool compareKeysPtr(const T *a, const T *b)
  {
    return compareKeys<KEYTYPE, UP>(*a, *b);
  }

  static INLINE void memMove(T *dest, const T *src, size_t numElem)
  {
    memmove((void *) dest, (const void *) src, numElem * sizeof(T));
  }

  static INLINE void sort(T *base, SortIndex n)
  {
    T saved;
    for (SortIndex j = 1; j < n; j++) {
      /* start at end, work backward until smaller element or i < 0. */
      SortIndex i = j - 1;
      T *value    = base + j;
      while ((i >= 0) && compareKeysPtr(value, base + i)) i--;
      /* If already in place, no movement needed. Otherwise save value to be
       * inserted and move as a LARGE block intervening values. Then insert
       * into proper position. */
      if (++i == j) continue;
      // doesn't work for uint128_t ("array type is not assignable")
      // saved = *value;
      memMove(&saved, value, 1);
      memMove(base + (i + 1), base + i, (j - i));
      // doesn't work for uint128_t ("array type is not assignable")
      // *(base + i) = saved;
      memMove(base + i, &saved, 1);
    }
  }

public:
  static INLINE void sort(T *d, SortIndex left, SortIndex right)
  {
    sort(d + left, (right - left) + 1);
  };
};

// =========================================================================
// recursion framework
// =========================================================================

// same framework used for all bit sorters (seq., SIMD, baseline)

// -------------------------------------------------------------------------
// recursion
// -------------------------------------------------------------------------

template <typename KEYTYPE, int UP,
          template <typename, int, typename> class CMP_SORTER, int UP_CMP,
          template <int, typename> class RADIX_BIT_SORTER, typename T>
static void radixRecursion(T *d, int bitNo, int lowestBitNo, SortIndex left,
                           SortIndex right, SortIndex cmpSortThresh)
{
  if (right - left <= cmpSortThresh) {
    CMP_SORTER<KEYTYPE, UP_CMP, T>::sort(d, left, right);
    return;
  }
  SortIndex split = RADIX_BIT_SORTER<UP, T>::bitSorter(d, bitNo, left, right);
  bitNo--;
  if (bitNo >= lowestBitNo) {
    radixRecursion<KEYTYPE, UP, CMP_SORTER, UP_CMP, RADIX_BIT_SORTER>(
      d, bitNo, lowestBitNo, left, split - 1, cmpSortThresh);
    radixRecursion<KEYTYPE, UP, CMP_SORTER, UP_CMP, RADIX_BIT_SORTER>(
      d, bitNo, lowestBitNo, split, right, cmpSortThresh);
  }
}

// -------------------------------------------------------------------------
// handling of sign-abs, two's complement, unsigned
// -------------------------------------------------------------------------

template <int UP, bool IsFloatingPoint, bool IsSigned>
struct _Radix;

// floating point (treated as sign and absolute value (char. + mantissa))
// upHigh:
// UP==1: 1s to left, 0s to right (1-UP==0)
// UP==0: 0s to left, 1s to right (1-UP==1)
// upLeft, upRight:
// sort remaining bits (absolute value)
// UP==1:
// neg. numbers (left part): sort abs. val. downwards (0)
// pos. numbers (right part): sort abs. val. upwards (1)
// UP==0:
// pos. numbers (left part): sort abs. val. downwards (0)
// neg. numbers (right part): sort. abs. val. upwards (1)
template <int UP>
struct _Radix<UP, true, true>
{
  enum { upHigh = 1 - UP, upLeft = 0, upRight = 1 };
};

// two's complement
template <int UP>
struct _Radix<UP, false, true>
{
  enum { upHigh = 1 - UP, upLeft = UP, upRight = UP };
};

// dual
template <int UP>
struct _Radix<UP, false, false>
{
  enum { upHigh = UP, upLeft = UP, upRight = UP };
};

// hub
template <int UP, typename T>
struct Radix
  : _Radix<UP, std::is_floating_point<T>::value, std::is_signed<T>::value>
{};

// -------------------------------------------------------------------------
// start of recursion
// -------------------------------------------------------------------------

template <typename KEYTYPE, int UP,
          template <typename, int, typename> class CMP_SORTER,
          template <int, typename> class RADIX_BIT_SORTER, typename T>
static void radixSort(T *d, int highestBitNo, int lowestBitNo, SortIndex left,
                      SortIndex right, SortIndex cmpSortThresh)
{
  if (right - left <= cmpSortThresh) {
    CMP_SORTER<KEYTYPE, UP, T>::sort(d, left, right);
    return;
  }
  int bitNo       = highestBitNo;
  SortIndex split = RADIX_BIT_SORTER<Radix<UP, KEYTYPE>::upHigh, T>::bitSorter(
    d, bitNo, left, right);
  bitNo--;
  if (bitNo >= lowestBitNo) {
    radixRecursion<KEYTYPE, Radix<UP, KEYTYPE>::upLeft, CMP_SORTER, UP,
                   RADIX_BIT_SORTER>(d, bitNo, lowestBitNo, left, split - 1,
                                     cmpSortThresh);
    radixRecursion<KEYTYPE, Radix<UP, KEYTYPE>::upRight, CMP_SORTER, UP,
                   RADIX_BIT_SORTER>(d, bitNo, lowestBitNo, split, right,
                                     cmpSortThresh);
  }
}

// =========================================================================
// wrapper
// =========================================================================

template <typename KEYTYPE, int UP, typename ELEMENTTYPE>
static void seqRadixSort(ELEMENTTYPE *d, SortIndex left, SortIndex right,
                         SortIndex cmpSortThresh)
{
  radixSort<KEYTYPE, UP, InsertionSort, SeqRadixBitSorter>(
    d, BitRange<KEYTYPE>::msb, BitRange<KEYTYPE>::lsb, left, right,
    cmpSortThresh);
}

template <typename KEYTYPE, int UP, typename ELEMENTTYPE>
static void seqRadixSort2(ELEMENTTYPE *d, SortIndex left, SortIndex right,
                          SortIndex cmpSortThresh)
{
  radixSort<KEYTYPE, UP, InsertionSort, SeqRadixBitSorter2>(
    d, BitRange<KEYTYPE>::msb, BitRange<KEYTYPE>::lsb, left, right,
    cmpSortThresh);
}

template <typename KEYTYPE, int UP, typename ELEMENTTYPE>
static void baselineRadixSort(ELEMENTTYPE *d, SortIndex left, SortIndex right,
                              SortIndex cmpSortThresh)
{
  radixSort<KEYTYPE, UP, InsertionSort, BaselineRadixBitSorter>(
    d, BitRange<KEYTYPE>::msb, BitRange<KEYTYPE>::lsb, left, right,
    cmpSortThresh);
}

#ifdef SIMD_RADIX_HAS_AVX512

template <typename KEYTYPE, int UP, typename ELEMENTTYPE>
static void simdRadixSortCompress(ELEMENTTYPE *d, SortIndex left,
                                  SortIndex right, SortIndex cmpSortThresh)
{
  radixSort<KEYTYPE, UP, InsertionSort, SimdRadixBitSorterCompress>(
    d, BitRange<KEYTYPE>::msb, BitRange<KEYTYPE>::lsb, left, right,
    cmpSortThresh);
}

#endif // SIMD_RADIX_HAS_AVX512

} // namespace radix

#endif

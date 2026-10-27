// ===========================================================================
//
// TimeMeasurement.H --
// functions for time measurement
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

// 02. Mar 23 (Jonas Keller): added doxygen documentation

// https://
// en.wikibooks.org/wiki/C%2B%2B_Programming/Code/Design_Patterns#Factory

#pragma once
#ifndef TIME_MEASUREMENT_H_
#define TIME_MEASUREMENT_H_

#include <ctime>
// 20. Apr 18 (rm): thanks to Benedikt Volkmer
#include <cassert>

// 20. Sep 22 (Jonas Keller): add clock_gettime replacement implementation for
// Windows
// TODO: doesn't differentiate between CLOCK_REALTIME, CLOCK_MONOTONIC and
// CLOCK_PROCESS_CPUTIME_ID from:
// https://github.com/Alexpux/mingw-w64/blob/master/mingw-w64-libraries/winpthreads/src/clock.c
#ifdef _WIN32
#include <errno.h>
#include <stdint.h>
// windows.h typedefs PATTERN, which causes a redifinition error in
// TiltSearchDemo.C
#define PATTERN something_not_named_PATTERN
#include <windows.h>
#undef PATTERN
// max and min are defined as macros in Windows.h, which causes problems
#undef max
#undef min

#define POW10_7 10000000
#define POW10_9 1000000000

using clockid_t = int;

#define CLOCK_REALTIME           0
#define CLOCK_MONOTONIC          1
#define CLOCK_PROCESS_CPUTIME_ID 2
#define CLOCK_THREAD_CPUTIME_ID  3

/* Number of 100ns-seconds between the beginning of the Windows epoch
 * (Jan. 1, 1601) and the Unix epoch (Jan. 1, 1970)
 */
#define DELTA_EPOCH_IN_100NS INT64_C(116444736000000000)

static int lc_set_errno(int result)
{
  if (result != 0) {
    errno = result;
    return -1;
  }
  return 0;
}

/**
 * Get the time of the specified clock clock_id and stores it in the struct
 * timespec pointed to by tp.
 * @param  clock_id The clock_id argument is the identifier of the particular
 *         clock on which to act. The following clocks are supported:
 * <pre>
 *     CLOCK_REALTIME  System-wide real-time clock. Setting this clock
 *                 requires appropriate privileges.
 *     CLOCK_MONOTONIC Clock that cannot be set and represents monotonic
 *                 time since some unspecified starting point.
 *     CLOCK_PROCESS_CPUTIME_ID High-resolution per-process timer from the CPU.
 *     CLOCK_THREAD_CPUTIME_ID  Thread-specific CPU-time clock.
 * </pre>
 * @param  tp The pointer to a timespec structure to receive the time.
 * @return If the function succeeds, the return value is 0
 *         If the function fails, the return value is -1,
 *         with errno set to indicate the error.
 */
int clock_gettime(clockid_t clock_id, struct timespec *tp)
{
  uint64_t t;
  LARGE_INTEGER pf, pc;
  union {
    uint64_t u64;
    FILETIME ft;
  } ct, et, kt, ut;

  switch (clock_id) {
  case CLOCK_REALTIME: {
    GetSystemTimeAsFileTime(&ct.ft);
    t           = ct.u64 - DELTA_EPOCH_IN_100NS;
    tp->tv_sec  = t / POW10_7;
    tp->tv_nsec = ((int) (t % POW10_7)) * 100;

    return 0;
  }

  case CLOCK_MONOTONIC: {
    if (QueryPerformanceFrequency(&pf) == 0) return lc_set_errno(EINVAL);

    if (QueryPerformanceCounter(&pc) == 0) return lc_set_errno(EINVAL);

    tp->tv_sec = pc.QuadPart / pf.QuadPart;
    tp->tv_nsec =
      (int) (((pc.QuadPart % pf.QuadPart) * POW10_9 + (pf.QuadPart >> 1)) /
             pf.QuadPart);
    if (tp->tv_nsec >= POW10_9) {
      tp->tv_sec++;
      tp->tv_nsec -= POW10_9;
    }

    return 0;
  }

  case CLOCK_PROCESS_CPUTIME_ID: {
    if (0 ==
        GetProcessTimes(GetCurrentProcess(), &ct.ft, &et.ft, &kt.ft, &ut.ft))
      return lc_set_errno(EINVAL);
    t           = kt.u64 + ut.u64;
    tp->tv_sec  = t / POW10_7;
    tp->tv_nsec = ((int) (t % POW10_7)) * 100;

    return 0;
  }

  case CLOCK_THREAD_CPUTIME_ID: {
    if (0 == GetThreadTimes(GetCurrentThread(), &ct.ft, &et.ft, &kt.ft, &ut.ft))
      return lc_set_errno(EINVAL);
    t           = kt.u64 + ut.u64;
    tp->tv_sec  = t / POW10_7;
    tp->tv_nsec = ((int) (t % POW10_7)) * 100;

    return 0;
  }

  default: break;
  }

  return lc_set_errno(EINVAL);
}
#endif

namespace simd {
/**
 * @addtogroup group_time_measurements
 * @{
 */

/* long long not supported everywhere
signed long long
getTimeUsec()
{
  signed long long sec, nsec;
  struct timespec ts;
  assert(!clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts));
  sec = ts.tv_sec;
  nsec = ts.tv_nsec;
  return 1000000LL * sec + nsec / 1000LL;
}
*/

/**
 * @brief Get the current value of this processes CPU time clock.
 *
 * @warning This clock has a resolution of 100ns on some systems
 * (e.g. Windows).
 *
 * @return current value of this processes CPU time clock
 */
struct timespec getTimeSpec()
{
  struct timespec ts;
  assert(!clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts));
  return ts;
}

/**
 * @brief Get the current value of the system-wide real-time clock.
 *
 * @return current value of the system-wide real-time clock
 */
struct timespec getTimeSpecMonotonic()
{
  struct timespec ts;
  assert(!clock_gettime(CLOCK_MONOTONIC, &ts));
  return ts;
}

// https://www.gnu.org/
//   software/libc/manual/html_node/Calculating-Elapsed-Time.html
// adapted, computes x - y for timespec instead of timeval

/**
 * @brief Subtracts two timespecs.
 *
 * @param[in] xx first timespec
 * @param[in] yy second timespec
 * @param[out] result result of the subtraction
 * @return 1 if the difference is negative, otherwise 0
 */
int timespec_subtract(const struct timespec &xx, const struct timespec &yy,
                      struct timespec &result)
{
  struct timespec x = xx, y = yy;
  /* Perform the carry for the later subtraction by updating y. */
  if (x.tv_nsec < y.tv_nsec) {
    int nsec = (y.tv_nsec - x.tv_nsec) / 1000000000L + 1;
    y.tv_nsec -= 1000000000L * nsec;
    y.tv_sec += nsec;
  }
  if (x.tv_nsec - y.tv_nsec > 1000000000L) {
    int nsec = (x.tv_nsec - y.tv_nsec) / 1000000000L;
    y.tv_nsec += 1000000000L * nsec;
    y.tv_sec -= nsec;
  }
  /* Compute the time remaining to wait. tv_nsec is certainly positive. */
  result.tv_sec  = x.tv_sec - y.tv_sec;
  result.tv_nsec = x.tv_nsec - y.tv_nsec;
  /* Return 1 if result is negative. */
  return x.tv_sec < y.tv_sec;
}

// convert to us (intended for results of timespec_subtract!)

/**
 * @brief Converts a timespec to microseconds. Intended for results of
 * timespec_subtract.
 *
 * @param[in] x timespec to convert
 *
 * @return timespec converted to microseconds
 */
double timespec_usec(const struct timespec &x)
{
  return 1E6 * x.tv_sec + x.tv_nsec / 1E3;
}

/**
 * @brief Computes the difference between two timespec's in microseconds.
 *
 * @param x first timespec
 * @param y second timespec
 * @return difference between the two timespec's in microseconds
 */
double timeSpecDiffUsec(const struct timespec &x, const struct timespec &y)
{
  struct timespec diff;
  timespec_subtract(x, y, diff);
  return timespec_usec(diff);
}

// 09. Oct 22 (Jonas Keller): added timeSpecDiffNsec

// warning: may overflow if the difference is more than 2 seconds on platforms
// where long is 32 bits

/**
 * @brief Computes the difference between two timespec's in nanoseconds.
 *
 * @warning May overflow on platforms where long is 32 bits if the difference
 * is more than 2 seconds.
 *
 * @param x first timespec
 * @param y second timespec
 * @return difference between the two timespec's in nanoseconds
 */
long int timeSpecDiffNsec(const struct timespec &x, const struct timespec &y)
{
  struct timespec diff;
  timespec_subtract(x, y, diff);
  return 1000000000L * diff.tv_sec + diff.tv_nsec;
}

/** @} */

} // namespace simd

#endif

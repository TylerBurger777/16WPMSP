// ===========================================================================
//
// SIMDRadixSortGenericThreads.H --
// thread-based extension
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

#pragma once
#ifndef SIMD_RADIX_SORT_GENERIC_THREADS_H_
#define SIMD_RADIX_SORT_GENERIC_THREADS_H_

#include "SIMDRadixSortGeneric.H"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

// with ideas from Anthony Williams: C++ Concurrency in Action, Manning 2012;
// page numbers relate to this book

namespace radix {

// ------------------------------------------------------------------------
// RadixThreadConfig
// ------------------------------------------------------------------------

struct RadixThreadConfig
{
  enum RadixQueueMode { RADIX_FIFO_QUEUE = 0, RADIX_LIFO_QUEUE = 1 };

  int numThreads;
  int queueMode;
  int useSlaves;
  double slaveFac;

  RadixThreadConfig(int numThreads)
    : numThreads(numThreads), queueMode(RADIX_FIFO_QUEUE), useSlaves(1),
      slaveFac(1.0)
  {}

  RadixThreadConfig(int numThreads, int queueMode, int useSlaves,
                    double slaveFac)
    : numThreads(numThreads), queueMode(queueMode), useSlaves(useSlaves),
      slaveFac(slaveFac)
  {}
};

// ------------------------------------------------------------------------
// RadixThreadStats
// ------------------------------------------------------------------------

struct RadixThreadStats
{
  std::vector<SortIndex> elements;
  std::vector<SortIndex> chunks;
  size_t maxListSize;

  RadixThreadStats(unsigned numThreads)
  {
    elements.resize(numThreads, 0);
    chunks.resize(numThreads, 0);
    maxListSize = 0;
  }

  void zero()
  {
    fill(elements.begin(), elements.end(), 0);
    fill(chunks.begin(), chunks.end(), 0);
    maxListSize = 0;
  }
};

// ------------------------------------------------------------------------
// RadixThreadSorter
// ------------------------------------------------------------------------

template <typename KEYTYPE, int UP,
          template <typename, int, typename> class CMP_SORTER,
          template <int, typename> class RADIX_BIT_SORTER, typename T>
class RadixThreadSorter
{
protected:
  // ------------------------------------------------------------------------
  // chunk to sort
  // ------------------------------------------------------------------------

  struct Chunk
  {
    // left and right border
    SortIndex left, right;
    // bit number for sorting
    int bitNo;
    // up: direction for radix (direction for sequential sorter is
    // UP)
    int up;
    // index of master thread (or NO_MASTER if there's no master)
    int masterThreadIdx;
    // index of slave task (not the same as thread index)
    int slaveIdx;

    enum { NO_MASTER = -1 };

    Chunk()
      : left(0), right(0), bitNo(0), up(0), masterThreadIdx(0), slaveIdx(0)
    {}
    Chunk(SortIndex left, SortIndex right, int bitNo, int up,
          int masterThreadIdx, int slaveIdx)
      : left(left), right(right), bitNo(bitNo), up(up),
        masterThreadIdx(masterThreadIdx), slaveIdx(slaveIdx)
    {}
  };

  // ------------------------------------------------------------------------
  // regions and blocks for master-slave mechanism
  // ------------------------------------------------------------------------

  struct Region
  {
    SortIndex left, split, right;
    Region() : left(0), split(0), right(0) {}
    Region(SortIndex left, SortIndex split, SortIndex right)
      : left(left), split(split), right(right)
    {}
  };

  struct Block
  {
    SortIndex left, size;
    // to which side should the block be moved:
    // 0 for left side, 1 for right side
    unsigned side;
    Block() : left(0), size(0), side(0) {}
    Block(SortIndex left, SortIndex size, unsigned side)
      : left(left), size(size), side(side)
    {}
  };

  // ------------------------------------------------------------------------
  // state
  // ------------------------------------------------------------------------

  // config
  RadixThreadConfig config;
  // stats (can be null)
  RadixThreadStats *stats;
  // for chunk size <= chunkThresh we switch to recursion
  SortIndex chunkThresh;
  // for chunk size <= chunkSlaveThresh we don't use slaves
  SortIndex chunkSlaveThresh;
  // data to sort
  T *d;
  // bit range for sorting
  int highestBitNo, lowestBitNo;
  // comparison threshold
  SortIndex cmpSortThresh;

  // list of chunks which still need to be sorted
  // std::list was somewhat slower
  std::deque<Chunk> chunkList;
  // counter of sleeping threads
  size_t waitingThreads;
  // thread pool
  std::vector<std::thread> threads;
  // mutex, condition variable, p.69
  std::mutex mtx;
  std::condition_variable cnd;

  // master-slave communication
  std::vector<std::vector<Region>> slaveResults;
  std::vector<int> slavesReady;
  // https://stackoverflow.com/questions/16465633
  std::mutex *masterMtx;
  std::condition_variable *masterCnd;

public:
  // ------------------------------------------------------------------------
  // radix-like sort for regions
  // ------------------------------------------------------------------------

  // swap (non-overlapping) regions in array d
  void swapRegions(SortIndex left1, SortIndex left2, SortIndex size)
  {
    std::swap_ranges(d + left1, d + left1 + size, d + left2);
  }

  // returns overall split point
  SortIndex sortRegions(const std::vector<Region> &regions)
  {
    // puts("sortRegions start");
    SortIndex overallSplit = 0;
    // convert from regions to blocks
    std::deque<Block> blocks;
    // SortIndex totalSize = 0;
    for (size_t i = 0; i < regions.size(); i++) {
      SortIndex left = regions[i].left, right = regions[i].right,
                split = regions[i].split;
      /*
      printf("%zu: left %ld split %ld right %ld\n",
             i, left, split, right);
      */
      // turn region into max. 2 blocks, empty blocks are not added
      SortIndex lSize = split - left;
      if (lSize > 0) blocks.push_back(Block(left, lSize, 0));
      SortIndex rSize = right + 1 - split;
      if (rSize > 0) blocks.push_back(Block(split, rSize, 1));
      // SortIndex elems = right + 1 - left;
      // totalSize += elems;
    }
    // blocks are ordered with respect to starting points
    // the block queue remains ordered in the processing
    //
    // process all blocks, swap overlapping regions
    //
    // [blocks], numbers refer to Block::side
    // FF [0]    -> lFound=false -> []    -> rFound=false -> []
    // TF [1]    -> lFound=true  -> []    -> rFound=false -> []
    // FF [00]   -> lFound=false -> []    -> rFound=false -> []
    // TF [01]   -> lFound=true  -> []    -> rFound=false -> []
    // TT [10]   -> lFound=true  -> [0]   -> rFound=true  -> [] + rest
    // TF [11]   -> lFound=true  -> [1]   -> rFound=false -> []
    // FF [000]  -> lFound=false -> []    -> rFound=false -> []
    // TF [001]  -> lFound=true  -> []    -> rFound=false -> []
    // TT [010]  -> lFound=true  -> [0]   -> rFound=true  -> [] + rest
    // TF [011]  -> lFound=true  -> [1]   -> rFound=false -> []
    // TT [100]  -> lFound=true  -> [00]  -> rFound=true  -> [0] + rest
    // TT [101]  -> lFound=true  -> [01]  -> rFound=true  -> [] + rest
    // TT [110]  -> lFound=true  -> [10]  -> rFound=true  -> [1] + rest
    // TF [111]  -> lFound=true  -> [11]  -> rFound=false -> []
    //
    // summary:
    // FF [00...000]
    // TF [0..01..1]
    // TT [at least one xxx10xxx]
    // FT never
    //
    // lFound=false, rFound=true can't occur (lFound=false only if
    // all blocks are of side 0, then they are removed in the first
    // loop)
    //
    while (!blocks.empty()) {
      // printf("%zu blocks\n", blocks.size());
      // from the left, search for a block that belongs to the right part;
      // blocks that are at the correct side are ignored;
      // if all blocks are already at the correct side (0), lFound=false;
      // the first "wrong" block belonging to the right side is removed and
      // processed;
      // if there was only one wrong block, the queue is empty afterwards
      Block lBlk;
      bool lFound = false;
      while (!blocks.empty()) {
        lBlk = blocks.front();
        blocks.pop_front();
        if (lBlk.side == 1) {
          lFound = true;
          break;
        }
        // this block is at the correct side (0), it is removed, the
        // split point is set beyond the right border of the block
        overallSplit = lBlk.left + lBlk.size;
      }
      /*
      if (lFound)
        printf("lBlk %ld %ld %u\n", lBlk.left, lBlk.size, lBlk.side);
      else
        puts("lFound = false");
      */
      // from the right, search for a block that belongs to the left part
      // blocks that are at the correct side are ignored
      Block rBlk;
      bool rFound = false;
      while (!blocks.empty()) {
        rBlk = blocks.back();
        blocks.pop_back();
        if (rBlk.side == 0) {
          rFound = true;
          break;
        }
      }
      /*
      if (rFound)
        printf("rBlk %ld %ld %u\n", rBlk.left, rBlk.size, rBlk.side);
      else
        puts("rFound = false");
      */
      // if a left and a right block are found, swap the overlapping
      // part (marked by R) and put the rest into the queue
      if (lFound && rFound) {
        SortIndex overlapSize = std::min(lBlk.size, rBlk.size);
        SortIndex restSize    = std::max(lBlk.size, rBlk.size) - overlapSize;
        // printf("overlap %ld rest %ld\n", overlapSize, restSize);
        // new split point (marked by "S")
        overallSplit = lBlk.left + overlapSize;
        // here 0/1 refers to array entries contained in the blocks
        // underlined: swapped regions
        if (lBlk.size < rBlk.size) {
          // left block is smaller
          // 111xxxxxxx00000
          // ---         ---
          // 000xxxxxxx00111
          //    S      RR
          swapRegions(lBlk.left, rBlk.left + restSize, overlapSize);
          blocks.push_back(Block(rBlk.left, restSize, 0));
        } else if (lBlk.size > rBlk.size) {
          // left block is larger
          // 11111xxxxx000
          // ---       ---
          // 00011xxxxx111
          //    RR
          //    S
          swapRegions(lBlk.left, rBlk.left, overlapSize);
          blocks.push_front(Block(lBlk.left + overlapSize, restSize, 1));
        } else {
          // blocks have the same size, no rest block
          // 111xxxx000
          // ---    ---
          // 000xxxx111
          //    S
          swapRegions(lBlk.left, rBlk.left, overlapSize);
        }
      } else if (lFound)
        // lFound && !rFound
        // -----------------
        // we found a left block (underlined) that should be moved
        // but no right block (these blocks are on the correct side and
        // were already removed)
        // (if there was no preceeding 0 block, overallSplit hasn't
        // been set yet, set it here)
        // 111
        // ---
        // S
        overallSplit = lBlk.left;
      else if (rFound) {
        // !lFound && rFound
        // -----------------
        // should never happen
        fprintf(stderr, "only right block was found, should not happen\n");
        exit(-1);
      }
      //
      // !lFound && !rFound
      // ------------------
      // only if possible if all blocks have side = 0
      // in this case the split point is already set in the first inner loop
    }
    // printf("sortRegion end, overallSplit = %ld\n", overallSplit);
    return overallSplit;
  }

  // ------------------------------------------------------------------------
  // queue
  // ------------------------------------------------------------------------

  void push(const Chunk &chunk) { chunkList.push_back(chunk); }

  Chunk pop()
  {
    Chunk chunk;
    if (config.queueMode == RadixThreadConfig::RADIX_FIFO_QUEUE) {
      chunk = chunkList.front();
      chunkList.pop_front();
    } else if (config.queueMode == RadixThreadConfig::RADIX_LIFO_QUEUE) {
      chunk = chunkList.back();
      chunkList.pop_back();
    } else {
      fprintf(stderr, "invalid queue mode %d\n", config.queueMode);
      exit(-1);
    }
    return chunk;
  }

  bool empty() { return chunkList.empty(); }

  void addChunk(const Chunk &chunk)
  {
    std::unique_lock<std::mutex> lck(mtx);
    push(chunk);
    cnd.notify_one();
    // this stat update needs to be inside mutex region
    if (stats)
      stats->maxListSize = std::max(stats->maxListSize, chunkList.size());
    // lck is released at end of scope
  }

  void addFirstChunk(const Chunk &chunk)
  {
    std::unique_lock<std::mutex> lck(mtx);
    push(chunk);
    waitingThreads = 0;
    // no notification since threads are not yet running
    // this stat update needs to be inside mutex region
    if (stats)
      stats->maxListSize = std::max(stats->maxListSize, chunkList.size());
    // lck is released at end of scope
  }

  // ------------------------------------------------------------------------
  // slave preparation
  // ------------------------------------------------------------------------

  void prepareSlaveResults(int masterThreadIdx, int portions)
  {
    std::unique_lock<std::mutex> lck(masterMtx[masterThreadIdx]);
    slavesReady[masterThreadIdx] = 0;
    slaveResults[masterThreadIdx].resize(portions);
    // lck released here
  }

  void storeSlaveResult(int masterThreadIdx, int slaveIdx, const Region &region)
  {
    // increase counter, store result, signal master
    std::unique_lock<std::mutex> lck(masterMtx[masterThreadIdx]);
    slavesReady[masterThreadIdx]++;
    slaveResults[masterThreadIdx][slaveIdx] = region;
    masterCnd[masterThreadIdx].notify_one();
    // lck is released here
  }

  void waitForSlaveResults(int masterThreadIdx, int portions)
  {
    std::unique_lock<std::mutex> lck(masterMtx[masterThreadIdx]);
    while (slavesReady[masterThreadIdx] < portions)
      masterCnd[masterThreadIdx].wait(lck);
    // lck released here
  }

  // ------------------------------------------------------------------------
  // recursion
  // ------------------------------------------------------------------------

  // we have to decide which recursion template function to
  // call depending on up (turn variable up into template
  // parameter)

  void recursionTail(SortIndex left, SortIndex right, int bitNo, int up)
  {
    if (up)
      radixRecursion<KEYTYPE, 1, CMP_SORTER, UP, RADIX_BIT_SORTER>(
        d, bitNo, lowestBitNo, left, right, cmpSortThresh);
    else
      radixRecursion<KEYTYPE, 0, CMP_SORTER, UP, RADIX_BIT_SORTER>(
        d, bitNo, lowestBitNo, left, right, cmpSortThresh);
  }

  void recursionHead(SortIndex left, SortIndex right, int up)
  {
    if (up)
      radixSort<KEYTYPE, 1, CMP_SORTER, RADIX_BIT_SORTER>(
        d, highestBitNo, lowestBitNo, left, right, cmpSortThresh);
    else
      radixSort<KEYTYPE, 0, CMP_SORTER, RADIX_BIT_SORTER>(
        d, highestBitNo, lowestBitNo, left, right, cmpSortThresh);
  }

  void recursion(SortIndex left, SortIndex right, int bitNo, int up)
  {
    if (bitNo == highestBitNo)
      recursionHead(left, right, up);
    else
      recursionTail(left, right, bitNo, up);
  }

  // ------------------------------------------------------------------------
  // bit sorting
  // ------------------------------------------------------------------------

  // we have to decide which sort template function to call
  // depending on up (turn variable up into template parameter)

  SortIndex sortBitsTail(SortIndex left, SortIndex right, int bitNo, int up,
                         int &upLeft, int &upRight)
  {
    upLeft = upRight = up;
    if (up)
      return RADIX_BIT_SORTER<1, T>::bitSorter(d, bitNo, left, right);
    else
      return RADIX_BIT_SORTER<0, T>::bitSorter(d, bitNo, left, right);
  }

  SortIndex sortBitsHead(SortIndex left, SortIndex right, int up, int &upLeft,
                         int &upRight)
  {
    if (up) {
      upLeft  = Radix<1, KEYTYPE>::upLeft;
      upRight = Radix<1, KEYTYPE>::upRight;
      return RADIX_BIT_SORTER<Radix<1, KEYTYPE>::upHigh, T>::bitSorter(
        d, highestBitNo, left, right);
    } else {
      upLeft  = Radix<0, KEYTYPE>::upLeft;
      upRight = Radix<0, KEYTYPE>::upRight;
      return RADIX_BIT_SORTER<Radix<0, KEYTYPE>::upHigh, T>::bitSorter(
        d, highestBitNo, left, right);
    }
  }

  SortIndex sortBits(SortIndex left, SortIndex right, int bitNo, int up,
                     int &upLeft, int &upRight)
  {
    if (bitNo == highestBitNo)
      return sortBitsHead(left, right, up, upLeft, upRight);
    else
      return sortBitsTail(left, right, bitNo, up, upLeft, upRight);
  }

  // ------------------------------------------------------------------------
  // thread function
  // ------------------------------------------------------------------------

  // TODO:
  // noticed the following when chunkFac was reduced, but maybe the problem
  // could occur also without chunkFac:
  // if all threads from the pool are masters, no threads are available
  // as slaves and the progress stops completely (masters wait for slaves
  // but no slaves threads are available), e.g. 2 threads, chunkFac 0.5,
  // after first split we have two masters

  // sort thread
  void sortThreadFunc(int threadIdx)
  {
    // endless loop
    while (true) {
      // lock mutex
      std::unique_lock<std::mutex> lck(mtx);
      // wait on condition variable, p.70 with lambda
      while (empty()) {
        // chunk list is empty
        // one more sleeping thread
        waitingThreads++;
        // if chunk list is empty and all threads are sleeping, we're done
        // (>= instead of ==, just to be on the safe side)
        if (waitingThreads >= threads.size()) {
          // wake up all other threads, they will also terminate here
          // cnd.notify_all();
          // this probably avoids a thundering herd problem
          cnd.notify_one();
          // lck is released when leaving scope
          return;
        }
        // wait for new chunk in list
        cnd.wait(lck);
        // there could be a new chunk, test again
        waitingThreads--;
      }
      // take and remove front element
      Chunk chunk = pop();
      // release lock
      lck.unlock();
      // stats
      if (stats) stats->chunks[threadIdx]++;
      // now we do the processing
      // copy data from chunk
      SortIndex left = chunk.left, right = chunk.right;
      int bitNo = chunk.bitNo, up = chunk.up;
      int masterThreadIdx = chunk.masterThreadIdx;
      int slaveIdx        = chunk.slaveIdx;
      // - I have a master:
      //   -> sort single level, store result for my master,
      //      increase result counter for master, signal master,
      //      get new chunk
      // - I have no master:
      //   - chunk is small enough to sort alone
      //     -> sort alone (recursively),
      //        get new chunk
      //   - chunk is too large to sort alone
      //     -> get slaves, prepare vector for results,
      //        process one chunk myself, wait for results from slaves,
      //        sort regions,
      //        get new chunk
      //
      if (masterThreadIdx != Chunk::NO_MASTER) {
        // puts("have master start"); fflush(stdout);
        // --- I have a master ---
        // sort single bit-level
        // (note that we assume that the region is large, the sequential
        // sorter is never invoked here)
        // config.useSlaves == false: we never enter this branch
        // how many elements are in the region?
        SortIndex elems = right + 1 - left;
        if (stats) stats->elements[threadIdx] += elems;
        // upLeft and upRight are ignored, are the same as in the master
        int upLeft, upRight;
        SortIndex split = sortBits(left, right, bitNo, up, upLeft, upRight);
        // store result
        storeSlaveResult(masterThreadIdx, slaveIdx, Region(left, split, right));
        // puts("have master end");
        // re-enter the loop and wait for a new chunk
      } else {
        // --- I have no master ---
        // inner loop
        while (true) {
          /*
          printf("t %d l %ld r %ld b %d u %d m %d s %d\n",
                 threadIdx, left, right, bitNo, up,
                 masterThreadIdx, slaveIdx);
          */
          // how many elements are in the region?
          SortIndex elems = right + 1 - left;
          if (elems <= chunkThresh) {
            // puts("have no master and small chunk start"); fflush(stdout);
            // stats
            if (stats) stats->elements[threadIdx] += elems;
            // block is small enough to fully process recursively
            recursion(left, right, bitNo, up);
            // puts("have no master and small chunk end");
            // leave inner loop,
            // re-enter the outer loop and wait for a new chunk
            break;
          } else {
            // elems > chunkThresh
            // puts("have no master and large chunk start"); fflush(stdout);
            int upLeft, upRight;
            SortIndex overallSplit;
            if (config.useSlaves && (elems > chunkSlaveThresh)) {
              // puts("use slaves"); fflush(stdout);
              // chunk is too large to handle alone, get slaves
              // we split the chunk into portions
              // e.g.:
              // chunkThresh < elems < 2 * chunkThresh: portions = 2
              // elems = 2 * chunkThresh: portions = 3
              // 2 * chunkThresh < elems < 3 * chunkThresh: portions = 3
              // we have at least 2 portions
              // TODO: is that a good way to compute number of portions?
              // TODO: would rounding be better?
              int portions = elems / chunkThresh + 1;
              // prepare slave results (we have to do it here since
              // slaves start with addChunk afterwards)
              prepareSlaveResults(threadIdx, portions);
              // size of portions (except first one)
              SortIndex portionSize = elems / portions;
              // size of first portion is the rest
              SortIndex firstPortionSize = elems - (portions - 1) * portionSize;
              // portion for the master
              SortIndex myLeft = left, myRight = left + firstPortionSize - 1;
              // assign other portions to slave threads
              SortIndex slaveLeft = myLeft + firstPortionSize;
              /*
                printf
                ("el %ld ct %ld p %d fps %ld ps %ld myL %ld myR %ld slL %ld\n",
                elems, chunkThresh, portions, firstPortionSize, portionSize,
                myLeft, myRight, slaveLeft);
              */
              for (int slaveIdx = 1; slaveIdx < portions; slaveIdx++) {
                addChunk(Chunk(slaveLeft, slaveLeft + portionSize - 1, bitNo,
                               up, threadIdx, slaveIdx));
                slaveLeft += portionSize;
              }
              // stats (of master portion)
              if (stats) stats->elements[threadIdx] += firstPortionSize;
              // I process the first portion myself
              // (note that we assume that the region is large, the
              // sequential sorter is never invoked here)
              SortIndex mySplit =
                sortBits(myLeft, myRight, bitNo, up, upLeft, upRight);
              // and store the result (like a slave)
              storeSlaveResult(threadIdx, 0, Region(myLeft, mySplit, myRight));
              // then I wait for my slaves to finish
              waitForSlaveResults(threadIdx, portions);
              // process regions
              overallSplit = sortRegions(slaveResults[threadIdx]);
            } else {
              // puts("no slaves"); fflush(stdout);
              // !config.useSlaves || (elems <= chunkSlaveThresh)
              // sort this level without slaves
              if (stats) stats->elements[threadIdx] += elems;
              overallSplit = sortBits(left, right, bitNo, up, upLeft, upRight);
            }
            // proceed with next bit level
            bitNo--;
            // only proceed if we haven't reached the lowest bit number
            if (bitNo >= lowestBitNo) {
#if 0
		// process left part by some other thread
		addChunk(Chunk(left, overallSplit - 1, bitNo, upLeft,
			       Chunk::NO_MASTER, 0));
		// process right part by some other thread
		addChunk(Chunk(overallSplit, right, bitNo, upRight,
			       Chunk::NO_MASTER, 0));
		// leave inner loop, get new chunk
		break;
#else
              // process right part by some other thread
              addChunk(Chunk(overallSplit, right, bitNo, upRight,
                             Chunk::NO_MASTER, 0));
              // process left part in the same thread
              right = overallSplit - 1;
              up    = upLeft;
#endif
            } else {
              // we can't go deeper with bitNo, wait for a new chunk
              // (leave inner loop)
              break;
            }
            // puts("have no master and large chunk end");
            // re-enter the loop and wait for a new chunk
          }
        }
      }
    }
  }

  // start parallel sorting
  void startSorting(SortIndex left, SortIndex right)
  {
    addFirstChunk(Chunk(left, right, highestBitNo, UP, Chunk::NO_MASTER, 0));
  }

  // ------------------------------------------------------------------------
  // constructor
  // ------------------------------------------------------------------------

  // stats can be null
  RadixThreadSorter(const RadixThreadConfig &config, RadixThreadStats *stats,
                    T *d, int highestBitNo, int lowestBitNo, SortIndex left,
                    SortIndex right, SortIndex cmpSortThresh)
    : config(config), stats(stats), d(d), highestBitNo(highestBitNo),
      lowestBitNo(lowestBitNo), cmpSortThresh(cmpSortThresh)
  {
    if (config.numThreads < 1) {
      fprintf(stderr, "RadixThreadSorter: numThreads (%d) < 1\n",
              config.numThreads);
      exit(-1);
    }
    // stats
    if (stats) stats->zero();
    // compute threshold
    SortIndex elems = right + 1 - left;
    // TODO: would rounding be better here?
    chunkThresh      = elems / config.numThreads;
    chunkSlaveThresh = config.slaveFac * chunkThresh;
    // mutex and cond. var. arrays
    masterMtx = new std::mutex[config.numThreads];
    masterCnd = new std::condition_variable[config.numThreads];
    // prepare vector for slave results
    slaveResults.resize(config.numThreads);
    slavesReady.resize(config.numThreads);
    // we first put tasks into the chunk list
    startSorting(left, right);
    // create thread pool (after putting tasks into the list, otherwise
    // termination would occur immediately because list empty, all sleeping)
    for (int i = 0; i < config.numThreads; i++)
      // p.28, p.275, p. 229, https://stackoverflow.com/questions/10673585/
      threads.push_back(
        std::thread(&RadixThreadSorter::sortThreadFunc, this, i));
    // wait for threads to terminate
    for (auto &thread : threads) thread.join();
  }

  ~RadixThreadSorter()
  {
    delete[] masterMtx;
    delete[] masterCnd;
  }
};

// ------------------------------------------------------------------------
// interface
// ------------------------------------------------------------------------

// TODO: other versions (SIMD...)

template <typename KEYTYPE, int UP, typename ELEMENTTYPE>
static void seqRadixSortThreads(const RadixThreadConfig &config,
                                RadixThreadStats *stats, ELEMENTTYPE *d,
                                SortIndex left, SortIndex right,
                                SortIndex cmpSortThresh)
{
  RadixThreadSorter<KEYTYPE, UP, InsertionSort, SeqRadixBitSorter, ELEMENTTYPE>
    threadSorter(config, stats, d, BitRange<KEYTYPE>::msb,
                 BitRange<KEYTYPE>::lsb, left, right, cmpSortThresh);
}

#ifdef SIMD_RADIX_HAS_AVX512

template <typename KEYTYPE, int UP, typename ELEMENTTYPE>
static void simdRadixSortCompressThreads(const RadixThreadConfig &config,
                                         RadixThreadStats *stats,
                                         ELEMENTTYPE *d, SortIndex left,
                                         SortIndex right,
                                         SortIndex cmpSortThresh)
{
  RadixThreadSorter<KEYTYPE, UP, InsertionSort, SimdRadixBitSorterCompress,
                    ELEMENTTYPE>
    threadSorter(config, stats, d, BitRange<KEYTYPE>::msb,
                 BitRange<KEYTYPE>::lsb, left, right, cmpSortThresh);
}

#endif // SIMD_RADIX_HAS_AVX512

} // namespace radix

#endif

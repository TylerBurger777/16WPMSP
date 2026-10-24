CXX      ?= g++
CXXFLAGS ?= -O3 -march=znver4 -std=c++17 -pthread -Iinclude
INCLUDES :=
ifneq ($(wildcard external/pdqsort),)
INCLUDES += -Iexternal/pdqsort
endif
ifneq ($(wildcard external/ips4o),)
INCLUDES += -Iexternal/ips4o
endif
ifneq ($(wildcard external/highway),)
INCLUDES += -Iexternal/highway
endif
ifneq ($(wildcard external/ska_sort),)
INCLUDES += -Iexternal/ska_sort
endif
ifneq ($(wildcard external/x86-simd-sort),)
INCLUDES += -Iexternal/x86-simd-sort
endif
all: bench_16wpmsp stress_16wpmsp
bench_16wpmsp: bench/bench_16wpmsp.cpp include/16wpmsp/16wpmsp.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<
stress_16wpmsp: tests/stress_16wpmsp.cpp include/16wpmsp/16wpmsp.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<
clean:
	rm -f bench_16wpmsp stress_16wpmsp
.PHONY: all clean

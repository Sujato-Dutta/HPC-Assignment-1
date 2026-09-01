#ifndef HPC_COMMON_H
#define HPC_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

static inline double wall_time_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

static inline double average_times(const double *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += x[i];
    return s / (double)n;
}

static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return x;
}

#define SM_GAMMA UINT64_C(0x9e3779b97f4a7c15)

#ifdef QUICK
static const int HPC_THREADS[] = {1, 2, 4};
#define HPC_NTHREADS 3
#define HPC_RUNS 2
#else
static const int HPC_THREADS[] = {1, 2, 4, 6, 8, 10, 12, 14, 16};
#define HPC_NTHREADS 9
#define HPC_RUNS 5
#endif

static inline void *xaligned_alloc(size_t alignment, size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, alignment, bytes) != 0 || !p) {
        fprintf(stderr, "Allocation failed for %.3f MiB\n", bytes / 1048576.0);
        exit(EXIT_FAILURE);
    }
    return p;
}

#endif

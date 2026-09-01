#define _POSIX_C_SOURCE 200112L
#include "common.h"
#include <math.h>
#include <inttypes.h>

/* Assignment 1, Q4: generate all primes in [2^40, 2^41].
   Efficient method: odd-only segmented sieve.
   There are tens of billions of primes in this interval, so storing/printing
   them would dominate runtime and require enormous storage. The program
   still visits every prime, and verifies the result with count + two
   order-independent 64-bit fingerprints.
*/

#ifdef QUICK
#define LOW  (UINT64_C(1) << 20)
#define HIGH (UINT64_C(1) << 21)
#define SEG_SPAN (UINT64_C(1) << 18)
#else
#define LOW  (UINT64_C(1) << 40)
#define HIGH (UINT64_C(1) << 41)
#define SEG_SPAN (UINT64_C(1) << 24)
#endif

typedef struct {
    uint64_t count;
    uint64_t sum_hash;
    uint64_t xor_hash;
} PrimeSig;

static int *base_primes(int limit, int *nprimes) {
    uint8_t *comp = (uint8_t *)calloc((size_t)limit + 1, 1);
    if (!comp) { perror("calloc"); exit(EXIT_FAILURE); }

    for (int p = 2; (int64_t)p * p <= limit; ++p)
        if (!comp[p])
            for (int64_t x = (int64_t)p * p; x <= limit; x += p)
                comp[x] = 1;

    int count = 0;
    for (int p = 3; p <= limit; p += 2) if (!comp[p]) ++count;
    int *pr = (int *)xaligned_alloc(64, (size_t)count * sizeof(int));
    int k = 0;
    for (int p = 3; p <= limit; p += 2) if (!comp[p]) pr[k++] = p;

    free(comp);
    *nprimes = count;
    return pr;
}

static inline PrimeSig sieve_one_segment(uint64_t slo, uint64_t shi,
                                         const int *primes, int np,
                                         uint8_t *mark, size_t mark_capacity) {
    PrimeSig s = {0, 0, 0};
    if (slo < 3) slo = 3;
    if ((slo & 1u) == 0) ++slo;
    if (slo >= shi) return s;

    size_t nodd = (size_t)((shi - slo + 1) / 2);
    if (nodd > mark_capacity) {
        fprintf(stderr, "Internal segment buffer too small\n");
        exit(EXIT_FAILURE);
    }
    memset(mark, 0, nodd);

    for (int pi = 0; pi < np; ++pi) {
        uint64_t p = (uint64_t)primes[pi];
        uint64_t pp = p * p;
        if (pp >= shi) break;

        uint64_t start = (slo + p - 1) / p * p;
        if (start < pp) start = pp;
        if ((start & 1u) == 0) start += p;
        if (start >= shi) continue;

        size_t idx = (size_t)((start - slo) >> 1);
        for (; idx < nodd; idx += (size_t)p) mark[idx] = 1;
    }

    for (size_t i = 0; i < nodd; ++i) {
        if (!mark[i]) {
            uint64_t p = slo + ((uint64_t)i << 1);
            uint64_t h = mix64(p);
            ++s.count;
            s.sum_hash += h;
            s.xor_hash ^= mix64(h + UINT64_C(0xd1b54a32d192ed03));
        }
    }
    return s;
}

static PrimeSig segmented_serial(const int *primes, int np) {
    PrimeSig total = {0, 0, 0};
    size_t cap = (size_t)(SEG_SPAN / 2 + 2);
    uint8_t *mark = (uint8_t *)xaligned_alloc(64, cap);

    uint64_t nseg = (HIGH - LOW + SEG_SPAN - 1) / SEG_SPAN;
    for (uint64_t s = 0; s < nseg; ++s) {
        uint64_t lo = LOW + s * SEG_SPAN;
        uint64_t hi = lo + SEG_SPAN;
        if (hi > HIGH) hi = HIGH;
        PrimeSig q = sieve_one_segment(lo, hi, primes, np, mark, cap);
        total.count += q.count;
        total.sum_hash += q.sum_hash;
        total.xor_hash ^= q.xor_hash;
    }
    free(mark);
    return total;
}

static PrimeSig segmented_parallel(const int *primes, int np, int threads) {
    uint64_t count = 0, sum_hash = 0, xor_hash = 0;
    uint64_t nseg = (HIGH - LOW + SEG_SPAN - 1) / SEG_SPAN;
    size_t cap = (size_t)(SEG_SPAN / 2 + 2);

    #pragma omp parallel num_threads(threads) reduction(+:count,sum_hash) reduction(^:xor_hash)
    {
        uint8_t *mark = (uint8_t *)xaligned_alloc(64, cap);

        #pragma omp for schedule(dynamic,1)
        for (uint64_t s = 0; s < nseg; ++s) {
            uint64_t lo = LOW + s * SEG_SPAN;
            uint64_t hi = lo + SEG_SPAN;
            if (hi > HIGH) hi = HIGH;
            PrimeSig q = sieve_one_segment(lo, hi, primes, np, mark, cap);
            count += q.count;
            sum_hash += q.sum_hash;
            xor_hash ^= q.xor_hash;
        }
        free(mark);
    }

    PrimeSig total = {count, sum_hash, xor_hash};
    return total;
}

static int same_sig(PrimeSig a, PrimeSig b) {
    return a.count == b.count && a.sum_hash == b.sum_hash && a.xor_hash == b.xor_hash;
}

int main(void) {
    omp_set_dynamic(0);
    int limit = (int)floor(sqrt((long double)(HIGH - 1)));
    int np = 0;
    int *primes = base_primes(limit, &np);
    fprintf(stderr, "Base primes: %d up to %d\n", np, limit);

    FILE *runs = fopen("primes_runs.csv", "w");
    FILE *summary = fopen("primes_summary.csv", "w");
    if (!runs || !summary) { perror("fopen"); return 1; }
    fprintf(runs, "variant,low,high,threads,run,time_sec,count,verified\n");
    fprintf(summary, "variant,low,high,threads,avg_sec,speedup,count,verified\n");

    PrimeSig ref = {0,0,0};
    int ref_ready = 0;
    double baseline = 0.0;

    for (int ti = 0; ti < HPC_NTHREADS; ++ti) {
        int t = HPC_THREADS[ti];
        double tm[HPC_RUNS];
        int verified = 1;
        PrimeSig last = {0,0,0};

        for (int r = 0; r < HPC_RUNS; ++r) {
            double st = wall_time_sec();
            last = (t == 1) ? segmented_serial(primes, np)
                            : segmented_parallel(primes, np, t);
            double et = wall_time_sec();
            tm[r] = et - st;

            if (!ref_ready && t == 1) { ref = last; ref_ready = 1; }
            else if (!same_sig(ref, last)) verified = 0;

            fprintf(runs,
                    "SEGMENTED_ODD,%" PRIu64 ",%" PRIu64 ",%d,%d,%.9f,%" PRIu64 ",%d\n",
                    LOW, HIGH, t, r + 1, tm[r], last.count, verified);
            fflush(runs);
        }

        double avg = average_times(tm, HPC_RUNS);
        if (t == 1) baseline = avg;
        double sp = baseline / avg;
        fprintf(summary,
                "SEGMENTED_ODD,%" PRIu64 ",%" PRIu64 ",%d,%.9f,%.6f,%" PRIu64 ",%d\n",
                LOW, HIGH, t, avg, sp, last.count, verified);
        fflush(summary);
        fprintf(stderr,
                "Primes [%" PRIu64 ",%" PRIu64 ") T=%2d avg=%.6f s speedup=%.3f count=%" PRIu64 " verified=%s\n",
                LOW, HIGH, t, avg, sp, last.count, verified ? "YES" : "NO");
    }

    fclose(runs); fclose(summary);
    free(primes);
    return 0;
}

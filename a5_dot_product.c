#define _POSIX_C_SOURCE 200112L
#include "common.h"
#include <inttypes.h>

/* Assignment 1, Q5: dot product of vectors of length 10^9.
   Entries are {-1,0,1}. int8_t stores the input exactly while reducing the
   two vectors from 8 GB (if int32_t pairs) to 2 GB total.
*/

#ifdef QUICK
#define VLEN UINT64_C(10000000)
#else
#define VLEN UINT64_C(1000000000)
#endif
#define SEED_A UINT64_C(0x1020304050607080)
#define SEED_B UINT64_C(0x8877665544332211)

static inline int8_t ternary_from_state(uint64_t s) {
    uint64_t z = mix64(s);
    unsigned v = (unsigned)(((__uint128_t)z * 3u) >> 64);
    return (int8_t)((int)v - 1);
}

static void init_vectors(int8_t *a, int8_t *b) {
    #pragma omp parallel num_threads(16)
    {
        int tid = omp_get_thread_num(), nt = omp_get_num_threads();
        uint64_t begin = (uint64_t)(((__uint128_t)VLEN * (unsigned)tid) / (unsigned)nt);
        uint64_t end   = (uint64_t)(((__uint128_t)VLEN * (unsigned)(tid + 1)) / (unsigned)nt);
        uint64_t sa = SEED_A + (begin + 1) * SM_GAMMA;
        uint64_t sb = SEED_B + (begin + 1) * SM_GAMMA;
        for (uint64_t i = begin; i < end; ++i, sa += SM_GAMMA, sb += SM_GAMMA) {
            a[i] = ternary_from_state(sa);
            b[i] = ternary_from_state(sb);
        }
    }
}

static int64_t dot_serial(const int8_t *restrict a, const int8_t *restrict b) {
    int64_t sum = 0;
    #pragma omp simd reduction(+:sum)
    for (uint64_t i = 0; i < VLEN; ++i)
        sum += (int)a[i] * (int)b[i];
    return sum;
}

static int64_t dot_parallel(const int8_t *restrict a, const int8_t *restrict b,
                            int threads) {
    int64_t sum = 0;
    #pragma omp parallel num_threads(threads) reduction(+:sum)
    {
        int tid = omp_get_thread_num(), nt = omp_get_num_threads();
        uint64_t begin = (uint64_t)(((__uint128_t)VLEN * (unsigned)tid) / (unsigned)nt);
        uint64_t end   = (uint64_t)(((__uint128_t)VLEN * (unsigned)(tid + 1)) / (unsigned)nt);
        int64_t local = 0;
        #pragma omp simd reduction(+:local)
        for (uint64_t i = begin; i < end; ++i)
            local += (int)a[i] * (int)b[i];
        sum += local;
    }
    return sum;
}

int main(void) {
    omp_set_dynamic(0);
    int8_t *a = (int8_t *)xaligned_alloc(64, (size_t)VLEN);
    int8_t *b = (int8_t *)xaligned_alloc(64, (size_t)VLEN);
    init_vectors(a, b); /* initialization is outside timed dot-product region */

    FILE *runs = fopen("dot_product_runs.csv", "w");
    FILE *summary = fopen("dot_product_summary.csv", "w");
    if (!runs || !summary) { perror("fopen"); return 1; }
    fprintf(runs, "variant,N,threads,run,time_sec,result,verified\n");
    fprintf(summary, "variant,N,threads,avg_sec,speedup,result,verified\n");

    int64_t ref = 0;
    int ref_ready = 0;
    double baseline = 0.0;

    for (int ti = 0; ti < HPC_NTHREADS; ++ti) {
        int t = HPC_THREADS[ti];
        double tm[HPC_RUNS];
        int verified = 1;
        int64_t last = 0;

        for (int r = 0; r < HPC_RUNS; ++r) {
            double st = wall_time_sec();
            last = (t == 1) ? dot_serial(a, b) : dot_parallel(a, b, t);
            double et = wall_time_sec();
            tm[r] = et - st;

            if (!ref_ready && t == 1) { ref = last; ref_ready = 1; }
            else if (last != ref) verified = 0;

            fprintf(runs, "DOT_PRODUCT,%" PRIu64 ",%d,%d,%.9f,%" PRId64 ",%d\n",
                    VLEN, t, r + 1, tm[r], last, verified);
            fflush(runs);
        }

        double avg = average_times(tm, HPC_RUNS);
        if (t == 1) baseline = avg;
        double sp = baseline / avg;
        fprintf(summary, "DOT_PRODUCT,%" PRIu64 ",%d,%.9f,%.6f,%" PRId64 ",%d\n",
                VLEN, t, avg, sp, last, verified);
        fflush(summary);
        fprintf(stderr,
                "Dot N=%" PRIu64 " T=%2d avg=%.6f s speedup=%.3f result=%" PRId64 " verified=%s\n",
                VLEN, t, avg, sp, last, verified ? "YES" : "NO");
    }

    fclose(runs); fclose(summary);
    free(a); free(b);
    return 0;
}

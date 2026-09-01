#define _POSIX_C_SOURCE 200112L
#include "common.h"
#include <inttypes.h>

/* Assignment 1, Q3: sort 10^12 integers in [0,10^3].
   Storing 10^12 integers explicitly needs TBs of memory. Because the key
   range is only 1001 values, the exact efficient solution is counting sort.
   We stream the deterministic random sequence and keep only 1001 counts.
*/

#ifdef QUICK
#define NUM_VALUES UINT64_C(10000000)
#else
#define NUM_VALUES UINT64_C(1000000000000)
#endif
#define NBINS 1001
#define HIST_STRIDE 1024
#define SEED UINT64_C(0x0123456789abcdef)

static inline unsigned value_from_state(uint64_t state) {
    uint64_t z = mix64(state);
    return (unsigned)(((__uint128_t)z * NBINS) >> 64);
}

static void histogram_serial(uint64_t hist[NBINS]) {
    memset(hist, 0, NBINS * sizeof(uint64_t));
    uint64_t state = SEED + SM_GAMMA;
    for (uint64_t i = 0; i < NUM_VALUES; ++i, state += SM_GAMMA)
        ++hist[value_from_state(state)];
}

static void histogram_parallel(uint64_t hist[NBINS], int threads) {
    uint64_t *locals = (uint64_t *)calloc((size_t)threads * HIST_STRIDE,
                                           sizeof(uint64_t));
    if (!locals) { perror("calloc"); exit(EXIT_FAILURE); }

    #pragma omp parallel num_threads(threads)
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        uint64_t begin = (uint64_t)(((__uint128_t)NUM_VALUES * (unsigned)tid) /
                                    (unsigned)nt);
        uint64_t end = (uint64_t)(((__uint128_t)NUM_VALUES * (unsigned)(tid + 1)) /
                                  (unsigned)nt);
        uint64_t *lh = locals + (size_t)tid * HIST_STRIDE;
        uint64_t state = SEED + (begin + 1) * SM_GAMMA;

        for (uint64_t i = begin; i < end; ++i, state += SM_GAMMA)
            ++lh[value_from_state(state)];
    }

    memset(hist, 0, NBINS * sizeof(uint64_t));
    for (int t = 0; t < threads; ++t) {
        const uint64_t *lh = locals + (size_t)t * HIST_STRIDE;
        for (int v = 0; v < NBINS; ++v) hist[v] += lh[v];
    }
    free(locals);
}

static int equal_hist(const uint64_t a[NBINS], const uint64_t b[NBINS]) {
    return memcmp(a, b, NBINS * sizeof(uint64_t)) == 0;
}

int main(void) {
    omp_set_dynamic(0);
    uint64_t *ref = (uint64_t *)xaligned_alloc(64, NBINS * sizeof(uint64_t));
    uint64_t *out = (uint64_t *)xaligned_alloc(64, NBINS * sizeof(uint64_t));

    FILE *runs = fopen("sorting_runs.csv", "w");
    FILE *summary = fopen("sorting_summary.csv", "w");
    if (!runs || !summary) { perror("fopen"); return 1; }
    fprintf(runs, "variant,N,threads,run,time_sec,verified\n");
    fprintf(summary, "variant,N,threads,avg_sec,speedup,verified\n");

    double baseline = 0.0;
    int ref_ready = 0;

    for (int ti = 0; ti < HPC_NTHREADS; ++ti) {
        int t = HPC_THREADS[ti];
        double tm[HPC_RUNS];
        int verified = 1;

        for (int r = 0; r < HPC_RUNS; ++r) {
            double st = wall_time_sec();
            if (t == 1) histogram_serial(out);
            else histogram_parallel(out, t);
            double et = wall_time_sec();
            tm[r] = et - st;

            if (!ref_ready && t == 1) {
                memcpy(ref, out, NBINS * sizeof(uint64_t));
                ref_ready = 1;
            } else if (!equal_hist(ref, out)) {
                verified = 0;
            }

            fprintf(runs, "COUNTING_SORT_STREAM,%" PRIu64 ",%d,%d,%.9f,%d\n",
                    NUM_VALUES, t, r + 1, tm[r], verified);
            fflush(runs);
        }

        double avg = average_times(tm, HPC_RUNS);
        if (t == 1) baseline = avg;
        double sp = baseline / avg;
        fprintf(summary, "COUNTING_SORT_STREAM,%" PRIu64 ",%d,%.9f,%.6f,%d\n",
                NUM_VALUES, t, avg, sp, verified);
        fflush(summary);
        fprintf(stderr, "Sort N=%" PRIu64 " T=%2d avg=%.6f s speedup=%.3f verified=%s\n",
                NUM_VALUES, t, avg, sp, verified ? "YES" : "NO");
    }

    /* Sanity: the compact sorted representation must contain exactly N items. */
    uint64_t total = 0;
    for (int v = 0; v < NBINS; ++v) total += out[v];
    fprintf(stderr, "Histogram total=%" PRIu64 " expected=%" PRIu64 "\n",
            total, NUM_VALUES);

    fclose(runs); fclose(summary);
    free(ref); free(out);
    return total == NUM_VALUES ? 0 : 2;
}

#define _POSIX_C_SOURCE 200112L
#include "common.h"
#include <inttypes.h>

/* Assignment 1, Q1: A^2 for N=2048 and 4096.
   A is row-major with entries in {-1,0,1}.
   OMM = ordinary i-k-j multiplication (not blocked).
   BMM = tiled multiplication for block sizes 4,8,16,32,64.
*/

#ifdef QUICK
static const int SIZES[] = {128, 256};
#else
static const int SIZES[] = {2048, 4096};
#endif
static const int NSIZES = 2;
static const int BLOCKS[] = {4, 8, 16, 32, 64};
static const int NBLOCKS = 5;

static void init_matrix(int8_t *A, int n) {
    const uint64_t seed = UINT64_C(0x123456789abcdef0);
    size_t total = (size_t)n * (size_t)n;

    #pragma omp parallel num_threads(16)
    {
        int tid = omp_get_thread_num();
        int nt  = omp_get_num_threads();
        size_t begin = (total * (size_t)tid) / (size_t)nt;
        size_t end   = (total * (size_t)(tid + 1)) / (size_t)nt;
        uint64_t state = seed + (uint64_t)(begin + 1) * SM_GAMMA;
        for (size_t i = begin; i < end; ++i, state += SM_GAMMA) {
            uint64_t z = mix64(state);
            int v = (int)(((__uint128_t)z * 3u) >> 64); /* 0,1,2 */
            A[i] = (int8_t)(v - 1);
        }
    }
}

static uint64_t hash_i32(const int32_t *C, size_t n) {
    uint64_t h = UINT64_C(0xcbf29ce484222325);
    for (size_t i = 0; i < n; ++i) {
        uint64_t x = (uint32_t)C[i];
        h ^= mix64(x + (uint64_t)i * SM_GAMMA);
        h *= UINT64_C(0x100000001b3);
    }
    return h;
}

static void omm(const int8_t *restrict A, int32_t *restrict C, int n, int threads) {
    size_t total = (size_t)n * (size_t)n;

    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static)
        for (size_t q = 0; q < total; ++q) C[q] = 0;

        /* i-k-j keeps C[i,*] and A[k,*] contiguous; no tiling is used. */
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            int32_t *restrict ci = C + (size_t)i * n;
            const int8_t *restrict ai = A + (size_t)i * n;
            for (int k = 0; k < n; ++k) {
                int aik = (int)ai[k];
                if (aik == 0) continue;
                const int8_t *restrict ak = A + (size_t)k * n;
                #pragma omp simd
                for (int j = 0; j < n; ++j)
                    ci[j] += aik * (int)ak[j];
            }
        }
    }
}

static void bmm(const int8_t *restrict A, int32_t *restrict C,
                int n, int bs, int threads) {
    size_t total = (size_t)n * (size_t)n;

    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static)
        for (size_t q = 0; q < total; ++q) C[q] = 0;

        /* Parallelize independent C tiles. */
        #pragma omp for collapse(2) schedule(static)
        for (int ii = 0; ii < n; ii += bs) {
            for (int jj = 0; jj < n; jj += bs) {
                int iend = ii + bs < n ? ii + bs : n;
                int jend = jj + bs < n ? jj + bs : n;
                for (int kk = 0; kk < n; kk += bs) {
                    int kend = kk + bs < n ? kk + bs : n;
                    for (int i = ii; i < iend; ++i) {
                        int32_t *restrict ci = C + (size_t)i * n;
                        const int8_t *restrict ai = A + (size_t)i * n;
                        for (int k = kk; k < kend; ++k) {
                            int aik = (int)ai[k];
                            if (aik == 0) continue;
                            const int8_t *restrict ak = A + (size_t)k * n;
                            #pragma omp simd
                            for (int j = jj; j < jend; ++j)
                                ci[j] += aik * (int)ak[j];
                        }
                    }
                }
            }
        }
    }
}

static void benchmark_variant(FILE *runs, FILE *summary,
                              const char *variant, int n, int bs,
                              const int8_t *A, int32_t *C,
                              uint64_t *reference_hash, int *reference_ready) {
    double baseline = 0.0;
    size_t total = (size_t)n * (size_t)n;

    for (int ti = 0; ti < HPC_NTHREADS; ++ti) {
        int t = HPC_THREADS[ti];
        double tm[HPC_RUNS];
        int verified = 1;

        for (int r = 0; r < HPC_RUNS; ++r) {
            double st = wall_time_sec();
            if (bs == 0) omm(A, C, n, t);
            else         bmm(A, C, n, bs, t);
            double et = wall_time_sec();
            tm[r] = et - st;

            uint64_t h = hash_i32(C, total);
            if (!*reference_ready && bs == 0 && t == 1) {
                *reference_hash = h;
                *reference_ready = 1;
            } else if (*reference_ready && h != *reference_hash) {
                verified = 0;
            }
            fprintf(runs, "%s,%d,%d,%d,%d,%.9f,%d\n",
                    variant, n, bs, t, r + 1, tm[r], verified);
            fflush(runs);
        }

        double avg = average_times(tm, HPC_RUNS);
        if (t == 1) baseline = avg;
        double sp = baseline / avg;
        fprintf(summary, "%s,%d,%d,%d,%.9f,%.6f,%d\n",
                variant, n, bs, t, avg, sp, verified);
        fflush(summary);
        fprintf(stderr, "N=%d %-6s T=%2d avg=%.6f s speedup=%.3f verified=%s\n",
                n, variant, t, avg, sp, verified ? "YES" : "NO");
    }
}

int main(void) {
    omp_set_dynamic(0);
    FILE *runs = fopen("matrix_square_runs.csv", "w");
    FILE *summary = fopen("matrix_square_summary.csv", "w");
    if (!runs || !summary) { perror("fopen"); return 1; }
    fprintf(runs, "variant,N,block,threads,run,time_sec,verified\n");
    fprintf(summary, "variant,N,block,threads,avg_sec,speedup,verified\n");

    for (int si = 0; si < NSIZES; ++si) {
        int n = SIZES[si];
        size_t total = (size_t)n * (size_t)n;
        int8_t *A = (int8_t *)xaligned_alloc(64, total * sizeof(int8_t));
        int32_t *C = (int32_t *)xaligned_alloc(64, total * sizeof(int32_t));
        init_matrix(A, n);

        uint64_t ref_hash = 0;
        int ref_ready = 0;

        benchmark_variant(runs, summary, "OMM", n, 0, A, C, &ref_hash, &ref_ready);
        for (int bi = 0; bi < NBLOCKS; ++bi) {
            char name[16];
            snprintf(name, sizeof(name), "BMM%d", BLOCKS[bi]);
            benchmark_variant(runs, summary, name, n, BLOCKS[bi], A, C,
                              &ref_hash, &ref_ready);
        }

        free(A);
        free(C);
    }

    fclose(runs);
    fclose(summary);
    return 0;
}

#define _POSIX_C_SOURCE 200112L
#include "common.h"
#include <math.h>
#include <inttypes.h>

/* Assignment 1, Q2: X^15 for a 2048x2048 complex matrix X=A+iB.
   Efficient addition chain (5 multiplies):
     X2=X*X, X3=X2*X, X6=X3*X3, X12=X6*X6, X15=X12*X3.
   Four complex matrices are enough: X0, M1, M2, M3.
*/

#ifdef QUICK
#define N 64
#else
#define N 2048
#endif
#define BS 32

typedef struct {
    double *r;
    double *i;
} CMat;

static CMat alloc_cmat(int n) {
    size_t total = (size_t)n * (size_t)n;
    CMat m;
    m.r = (double *)xaligned_alloc(64, total * sizeof(double));
    m.i = (double *)xaligned_alloc(64, total * sizeof(double));
    return m;
}

static void free_cmat(CMat *m) {
    free(m->r); free(m->i);
    m->r = m->i = NULL;
}

static void init_x(CMat x, int n) {
    size_t total = (size_t)n * (size_t)n;
    const uint64_t sa = UINT64_C(0x1111222233334444);
    const uint64_t sb = UINT64_C(0xaaaabbbbccccdddd);

    #pragma omp parallel num_threads(16)
    {
        int tid = omp_get_thread_num(), nt = omp_get_num_threads();
        size_t begin = total * (size_t)tid / (size_t)nt;
        size_t end   = total * (size_t)(tid + 1) / (size_t)nt;
        uint64_t sta = sa + (uint64_t)(begin + 1) * SM_GAMMA;
        uint64_t stb = sb + (uint64_t)(begin + 1) * SM_GAMMA;
        for (size_t q = begin; q < end; ++q, sta += SM_GAMMA, stb += SM_GAMMA) {
            int va = (int)(((__uint128_t)mix64(sta) * 3u) >> 64) - 1;
            int vb = (int)(((__uint128_t)mix64(stb) * 3u) >> 64) - 1;
            x.r[q] = (double)va;
            x.i[q] = (double)vb;
        }
    }
}

static void cmatmul(CMat a, CMat b, CMat c, int n, int threads) {
    size_t total = (size_t)n * (size_t)n;

    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static)
        for (size_t q = 0; q < total; ++q) {
            c.r[q] = 0.0;
            c.i[q] = 0.0;
        }

        #pragma omp for collapse(2) schedule(static)
        for (int ii = 0; ii < n; ii += BS) {
            for (int jj = 0; jj < n; jj += BS) {
                int iend = ii + BS < n ? ii + BS : n;
                int jend = jj + BS < n ? jj + BS : n;
                for (int kk = 0; kk < n; kk += BS) {
                    int kend = kk + BS < n ? kk + BS : n;
                    for (int i = ii; i < iend; ++i) {
                        double *restrict cr = c.r + (size_t)i * n;
                        double *restrict ci = c.i + (size_t)i * n;
                        const double *restrict ar = a.r + (size_t)i * n;
                        const double *restrict ai = a.i + (size_t)i * n;
                        for (int k = kk; k < kend; ++k) {
                            double xr = ar[k], xi = ai[k];
                            const double *restrict br = b.r + (size_t)k * n;
                            const double *restrict bi = b.i + (size_t)k * n;
                            #pragma omp simd
                            for (int j = jj; j < jend; ++j) {
                                cr[j] += xr * br[j] - xi * bi[j];
                                ci[j] += xr * bi[j] + xi * br[j];
                            }
                        }
                    }
                }
            }
        }
    }
}

static void power15(CMat x, CMat m1, CMat m2, CMat m3, int n, int threads) {
    cmatmul(x,  x,  m1, n, threads); /* X^2  */
    cmatmul(m1, x,  m2, n, threads); /* X^3  */
    cmatmul(m2, m2, m3, n, threads); /* X^6  */
    cmatmul(m3, m3, m1, n, threads); /* X^12 */
    cmatmul(m1, m2, m3, n, threads); /* X^15 */
}

static uint64_t hash_complex(CMat x, int n) {
    size_t total = (size_t)n * (size_t)n;
    uint64_t h = UINT64_C(0xcbf29ce484222325);
    for (size_t q = 0; q < total; ++q) {
        uint64_t ur, ui;
        memcpy(&ur, &x.r[q], sizeof(ur));
        memcpy(&ui, &x.i[q], sizeof(ui));
        h ^= mix64(ur + (uint64_t)q * SM_GAMMA);
        h *= UINT64_C(0x100000001b3);
        h ^= mix64(ui ^ ((uint64_t)q + UINT64_C(0x9e3779b97f4a7c15)));
        h *= UINT64_C(0x100000001b3);
    }
    return h;
}

int main(void) {
    omp_set_dynamic(0);
    CMat x = alloc_cmat(N), m1 = alloc_cmat(N), m2 = alloc_cmat(N), m3 = alloc_cmat(N);
    init_x(x, N);

    FILE *runs = fopen("complex_power15_runs.csv", "w");
    FILE *summary = fopen("complex_power15_summary.csv", "w");
    if (!runs || !summary) { perror("fopen"); return 1; }
    fprintf(runs, "variant,N,threads,run,time_sec,verified\n");
    fprintf(summary, "variant,N,threads,avg_sec,speedup,verified\n");

    uint64_t ref_hash = 0;
    int ref_ready = 0;
    double baseline = 0.0;

    for (int ti = 0; ti < HPC_NTHREADS; ++ti) {
        int t = HPC_THREADS[ti];
        double tm[HPC_RUNS];
        int verified = 1;

        for (int r = 0; r < HPC_RUNS; ++r) {
            double st = wall_time_sec();
            power15(x, m1, m2, m3, N, t);
            double et = wall_time_sec();
            tm[r] = et - st;

            uint64_t h = hash_complex(m3, N);
            if (!ref_ready && t == 1) { ref_hash = h; ref_ready = 1; }
            else if (h != ref_hash) verified = 0;

            fprintf(runs, "ADDCHAIN5,%d,%d,%d,%.9f,%d\n",
                    N, t, r + 1, tm[r], verified);
            fflush(runs);
        }

        double avg = average_times(tm, HPC_RUNS);
        if (t == 1) baseline = avg;
        double sp = baseline / avg;
        fprintf(summary, "ADDCHAIN5,%d,%d,%.9f,%.6f,%d\n",
                N, t, avg, sp, verified);
        fflush(summary);
        fprintf(stderr, "X^15 N=%d T=%2d avg=%.6f s speedup=%.3f verified=%s\n",
                N, t, avg, sp, verified ? "YES" : "NO");
    }

    fclose(runs); fclose(summary);
    free_cmat(&x); free_cmat(&m1); free_cmat(&m2); free_cmat(&m3);
    return 0;
}

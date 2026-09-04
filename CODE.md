# CS4110 HPC Assignment 1 — Problem 1: Code Explanation (CODE.md)
## Line-by-Line Technical Breakdown of `a1_matrix_square.c`

This document provides a comprehensive line-by-line explanation of [`a1_matrix_square.c`](file:///c:/Users/Sujato/Downloads/SEM%20-%20VII/High%20Performance%20Computing/Assignments/HPC_Assignment1_C/a1_matrix_square.c), covering algorithm design, memory layout, OpenMP pragmas, SIMD vectorization, and benchmarking logic.

---

## 1. Header Includes and Configuration (Lines 1–19)

```c
1: #define _POSIX_C_SOURCE 200112L
2: #include "common.h"
3: #include <inttypes.h>
```
* **Line 1 (`#define _POSIX_C_SOURCE 200112L`)**: Exposes POSIX-compliant system functions, specifically `posix_memalign()` and `gettimeofday()`, ensuring strict portable C standards across Linux/Unix systems.
* **Line 2 (`#include "common.h"`)**: Includes our shared assignment header containing timing helpers (`wall_time_sec()`), OpenMP thread configurations (`HPC_THREADS`), fast pseudo-random state mixing (`mix64()`), and aligned allocation (`xaligned_alloc()`).
* **Line 3 (`#include <inttypes.h>`)**: Provides standard exact-width integer types (`int8_t`, `int32_t`, `uint64_t`) and cross-platform formatting specifiers (like `PRIu64`).

```c
5: /* Assignment 1, Q1: A^2 for N=2048 and 4096.
6:    A is row-major with entries in {-1,0,1}.
7:    OMM = ordinary i-k-j multiplication (not blocked).
8:    BMM = tiled multiplication for block sizes 4,8,16,32,64.
9: */
```
* **Lines 5–9**: Summary docstring outlining the assignment requirements for Problem 1.

```c
11: #ifdef QUICK
12: static const int SIZES[] = {128, 256};
13: #else
14: static const int SIZES[] = {2048, 4096};
15: #endif
16: static const int NSIZES = 2;
17: static const int BLOCKS[] = {4, 8, 16, 32, 64};
18: static const int NBLOCKS = 5;
```
* **Lines 11–15**: Conditional compilation for matrix sizes. If `-DQUICK` is defined (`make quick`), small sizes (128, 256) are used for fast local testing; otherwise, the exact required assignment dimensions **2048 x 2048** and **4096 x 4096** are compiled.
* **Line 16 (`NSIZES = 2`)**: The number of matrix orders evaluated.
* **Lines 17–18 (`BLOCKS[]`, `NBLOCKS`)**: Array of the five required block sizes for Block Matrix Multiplication: **4, 8, 16, 32, and 64**.

---

## 2. Matrix Initialization: `init_matrix` (Lines 20–37)

```c
20: static void init_matrix(int8_t *A, int n) {
21:     const uint64_t seed = UINT64_C(0x123456789abcdef0);
22:     size_t total = (size_t)n * (size_t)n;
```
* **Line 20**: Function signature. Takes a 1D contiguous array `A` representing the N x N row-major matrix and dimension `n`.
* **Line 21**: Fixed 64-bit seed (`0x123456789abcdef0`) ensuring deterministic reproducible input matrices across all runs and thread counts.
* **Line 22**: Calculates total elements $N^2$. Uses `size_t` to prevent 32-bit integer overflow (e.g., $4096 \times 4096 = 16,777,216$ elements).

```c
24:     #pragma omp parallel num_threads(16)
25:     {
26:         int tid = omp_get_thread_num();
27:         int nt  = omp_get_num_threads();
28:         size_t begin = (total * (size_t)tid) / (size_t)nt;
29:         size_t end   = (total * (size_t)(tid + 1)) / (size_t)nt;
```
* **Line 24**: Launches an OpenMP parallel region with 16 threads to initialize the matrix in parallel.
* **Lines 26–27**: Retrieves thread ID (`tid`) and thread count (`nt`).
* **Lines 28–29**: Divides the total $N^2$ elements into contiguous, non-overlapping index ranges `[begin, end)` for each thread without any lock contention.

```c
30:         uint64_t state = seed + (uint64_t)(begin + 1) * SM_GAMMA;
31:         for (size_t i = begin; i < end; ++i, state += SM_GAMMA) {
32:             uint64_t z = mix64(state);
33:             int v = (int)(((__uint128_t)z * 3u) >> 64); /* 0,1,2 */
34:             A[i] = (int8_t)(v - 1);
35:         }
36:     }
37: }
```
* **Line 30**: Computes a unique starting PRNG state for this thread's chunk using SplitMix64's Weyl constant (`SM_GAMMA = 0x9e3779b97f4a7c15`). This guarantees that the matrix generated is identical regardless of thread scheduling.
* **Line 31**: Iterates across thread chunk indices.
* **Line 32**: Calls `mix64()`—a 64-bit avalanche mixer providing high-quality pseudo-randomness with zero memory dependencies.
* **Line 33**: Fast unbiased range reduction (Daniel Lemire's method). Computes `(z * 3) >> 64` using 128-bit multiplication to map random 64-bit integer `z` into `{0, 1, 2}` without expensive division/modulo operations (`% 3`).
* **Line 34**: Subtracts 1 (`v - 1`), mapping `{0, 1, 2}` to the assignment's required ternary values **`{-1, 0, 1}`**, and stores it into `int8_t`.

---

## 3. 64-Bit Verification Hash: `hash_i32` (Lines 39–47)

```c
39: static uint64_t hash_i32(const int32_t *C, size_t n) {
40:     uint64_t h = UINT64_C(0xcbf29ce484222325);
41:     for (size_t i = 0; i < n; ++i) {
42:         uint64_t x = (uint32_t)C[i];
43:         h ^= mix64(x + (uint64_t)i * SM_GAMMA);
44:         h *= UINT64_C(0x100000001b3);
45:     }
46:     return h;
47: }
```
* **Line 39**: Computes an order-sensitive 64-bit polynomial checksum over the output matrix `C` ($N^2$ elements).
* **Line 40**: Initializes `h` with the standard FNV-1a 64-bit offset basis constant (`0xcbf29ce484222325`).
* **Line 41**: Loops over all $N^2$ output entries.
* **Lines 42–44**: Mixes the entry value and index position `i` using `mix64()` and multiplies by the 64-bit FNV prime (`0x100000001b3`).
* **Line 46**: Returns a compact 64-bit fingerprint. If any entry in matrix C differs by even a single bit in parallel runs, this hash changes completely, enabling instant result verification.

---

## 4. Ordinary Matrix Multiplication: `omm` (Lines 49–72)

```c
49: static void omm(const int8_t *restrict A, int32_t *restrict C, int n, int threads) {
50:     size_t total = (size_t)n * (size_t)n;
```
* **Line 49**: Signature for Ordinary Matrix Multiplication.
  * `const int8_t *restrict A`: The input matrix. The `restrict` keyword informs the compiler that pointers `A` and `C` do not alias, enabling aggressive register caching and SIMD vectorization.
  * `int32_t *restrict C`: The 32-bit output matrix ($A^2$).
  * `int n`: Dimension (2048 or 4096).
  * `int threads`: Number of OpenMP threads to use (1 to 16).
* **Line 50**: Total entries in the matrix ($N^2$).

```c
52:     #pragma omp parallel num_threads(threads)
53:     {
54:         #pragma omp for schedule(static)
55:         for (size_t q = 0; q < total; ++q) C[q] = 0;
```
* **Line 52**: Spawns the OpenMP thread pool with the specified thread count.
* **Lines 54–55**: Multi-threaded zero initialization of output matrix $C$ before accumulation. `schedule(static)` partitions chunks equally among threads.

```c
57:         /* i-k-j keeps C[i,*] and A[k,*] contiguous; no tiling is used. */
58:         #pragma omp for schedule(static)
59:         for (int i = 0; i < n; ++i) {
60:             int32_t *restrict ci = C + (size_t)i * n;
61:             const int8_t *restrict ai = A + (size_t)i * n;
```
* **Line 58**: `#pragma omp for schedule(static)` distributes outer matrix rows `i = 0 ... n-1` among the threads. Each thread computes independent output rows `C[i, *]`, completely eliminating race conditions.
* **Lines 60–61**: Pre-computes base row pointers:
  * `ci`: Pointer to row `i` of matrix $C$.
  * `ai`: Pointer to row `i` of matrix $A$.

```c
62:             for (int k = 0; k < n; ++k) {
63:                 int aik = (int)ai[k];
64:                 if (aik == 0) continue;
65:                 const int8_t *restrict ak = A + (size_t)k * n;
66:                 #pragma omp simd
67:                 for (int j = 0; j < n; ++j)
68:                     ci[j] += aik * (int)ak[j];
69:             }
70:         }
71:     }
72: }
```
* **Line 62**: Middle loop over `k`. Notice the loop order is **`i-k-j`** instead of standard `i-j-k`.
* **Line 63**: Loads scalar element $A[i, k]$ once into CPU register `aik`.
* **Line 64 (`Zero-Bypass Optimization`)**: If `aik == 0`, adding $0 \times A[k, j]$ contributes nothing to row $C[i]$. It skips the inner loop entirely! Since $A$ has values in $\{-1, 0, 1\}$, roughly **33% of all compute iterations are skipped**, saving billions of operations.
* **Line 65**: Pre-computes pointer `ak` to row `k` of matrix $A$.
* **Line 66 (`#pragma omp simd`)**: Directs the compiler to emit AVX2 / AVX-512 vector instructions (e.g., `vpaddd`, `vpmaddwd`) for the inner loop.
* **Lines 67–68**: The innermost loop iterates over column index `j`. Because `ci[j]` and `ak[j]` are accessed contiguously in row-major order (stride 1), the CPU hardware prefetcher loads complete 64-byte cache lines without cache stalls.

---

## 5. Block Matrix Multiplication: `bmm` (Lines 74–107)

```c
74: static void bmm(const int8_t *restrict A, int32_t *restrict C,
75:                 int n, int bs, int threads) {
76:     size_t total = (size_t)n * (size_t)n;
```
* **Lines 74–76**: Signature for Block Matrix Multiplication with block size `bs` (4, 8, 16, 32, or 64).

```c
78:     #pragma omp parallel num_threads(threads)
79:     {
80:         #pragma omp for schedule(static)
81:         for (size_t q = 0; q < total; ++q) C[q] = 0;
```
* **Lines 78–81**: Initializes matrix $C$ to zero in parallel across threads.

```c
83:         /* Parallelize independent C tiles. */
84:         #pragma omp for collapse(2) schedule(static)
85:         for (int ii = 0; ii < n; ii += bs) {
86:             for (int jj = 0; jj < n; jj += bs) {
87:                 int iend = ii + bs < n ? ii + bs : n;
88:                 int jend = jj + bs < n ? jj + bs : n;
```
* **Line 84 (`collapse(2)`)**: Flattens the 2D grid of output blocks `(ii, jj)` into a single 1D iteration space. For $N=4096, bs=64$, this creates $(4096/64)^2 = 4096$ independent tiles, perfectly load-balancing across 16 threads.
* **Lines 85–86**: Outer block loops stepping in increments of `bs`.
* **Lines 87–88**: Clamps block boundaries `iend` and `jend` to prevent out-of-bounds access.

```c
89:                 for (int kk = 0; kk < n; kk += bs) {
90:                     int kend = kk + bs < n ? kk + bs : n;
91:                     for (int i = ii; i < iend; ++i) {
92:                         int32_t *restrict ci = C + (size_t)i * n;
93:                         const int8_t *restrict ai = A + (size_t)i * n;
94:                         for (int k = kk; k < kend; ++k) {
95:                             int aik = (int)ai[k];
96:                             if (aik == 0) continue;
97:                             const int8_t *restrict ak = A + (size_t)k * n;
98:                             #pragma omp simd
99:                             for (int j = jj; j < jend; ++j)
100:                                 ci[j] += aik * (int)ak[j];
101:                         }
102:                     }
103:                 }
104:             }
105:         }
106:     }
107: }
```
* **Line 89**: Steps through the block multiplication dimension `kk`.
* **Lines 91–102**: Micro-kernel performing matrix multiplication on $B \times B$ tiles:
  * Loops `i` from `ii` to `iend` and `k` from `kk` to `kend`.
  * Line 96 skips zero elements (`aik == 0`).
  * Line 98 vectorizes the inner `j` loop (`jj` to `jend`) with `#pragma omp simd`.
  * **Cache Benefit**: All active $B \times B$ elements of $A$ and $C$ reside continuously inside fast L1/L2 data cache during the calculation.

---

## 6. Benchmark & Timing Harness: `benchmark_variant` (Lines 109–149)

```c
109: static void benchmark_variant(FILE *runs, FILE *summary,
110:                               const char *variant, int n, int bs,
111:                               const int8_t *A, int32_t *C,
112:                               uint64_t *reference_hash, int *reference_ready) {
113:     double baseline = 0.0;
114:     size_t total = (size_t)n * (size_t)n;
```
* **Lines 109–114**: Benchmarking engine for a specific variant (`OMM` or `BMM4..64`). Writes outputs to individual run and summary CSV files.

```c
116:     for (int ti = 0; ti < HPC_NTHREADS; ++ti) {
117:         int t = HPC_THREADS[ti];
118:         double tm[HPC_RUNS];
119:         int verified = 1;
```
* **Line 116**: Loops through the 9 thread counts: `1, 2, 4, 6, 8, 10, 12, 14, 16`.
* **Line 118**: Array to store elapsed times for the **5 required runs** per thread setting.

```c
121:         for (int r = 0; r < HPC_RUNS; ++r) {
122:             double st = wall_time_sec();
123:             if (bs == 0) omm(A, C, n, t);
124:             else         bmm(A, C, n, bs, t);
125:             double et = wall_time_sec();
126:             tm[r] = et - st;
```
* **Line 121**: Repeats the measurement 5 times.
* **Lines 122–125**: Precision timing using `gettimeofday()` wrapper (`wall_time_sec()`). Only the computation (`omm` or `bmm`) is inside the timed window.
* **Line 126**: Records execution time in seconds.

```c
128:             uint64_t h = hash_i32(C, total);
129:             if (!*reference_ready && bs == 0 && t == 1) {
130:                 *reference_hash = h;
131:                 *reference_ready = 1;
132:             } else if (*reference_ready && h != *reference_hash) {
133:                 verified = 0;
134:             }
135:             fprintf(runs, "%s,%d,%d,%d,%d,%.9f,%d\n",
136:                     variant, n, bs, t, r + 1, tm[r], verified);
137:             fflush(runs);
138:         }
```
* **Line 128**: Computes 64-bit hash of matrix $C$.
* **Lines 129–134**: The very first 1-thread OMM run sets `reference_hash`. Every subsequent parallel run verifies that its output hash matches the reference (`verified = 1`).
* **Lines 135–137**: Writes each individual run to `matrix_square_runs.csv` (Columns: `variant, N, block, threads, run, time_sec, verified`).

```c
140:         double avg = average_times(tm, HPC_RUNS);
141:         if (t == 1) baseline = avg;
142:         double sp = baseline / avg;
143:         fprintf(summary, "%s,%d,%d,%d,%.9f,%.6f,%d\n",
144:                 variant, n, bs, t, avg, sp, verified);
145:         fflush(summary);
146:         fprintf(stderr, "N=%d %-6s T=%2d avg=%.6f s speedup=%.3f verified=%s\n",
147:                 n, variant, t, avg, sp, verified ? "YES" : "NO");
148:     }
149: }
```
* **Line 140**: Computes average runtime over 5 runs.
* **Lines 141–142**: Calculates speedup: $S = \text{avg}(T_1) / \text{avg}(T_p)$.
* **Lines 143–145**: Writes summary to `matrix_square_summary.csv` (Columns: `variant, N, block, threads, avg_sec, speedup, verified`).
* **Lines 146–147**: Prints real-time formatted log to `stderr` (which Slurm captures in `slurm_error_*.log`).

---

## 7. Main Function: `main` (Lines 151–184)

```c
151: int main(void) {
152:     omp_set_dynamic(0);
153:     FILE *runs = fopen("matrix_square_runs.csv", "w");
154:     FILE *summary = fopen("matrix_square_summary.csv", "w");
155:     if (!runs || !summary) { perror("fopen"); return 1; }
156:     fprintf(runs, "variant,N,block,threads,run,time_sec,verified\n");
157:     fprintf(summary, "variant,N,block,threads,avg_sec,speedup,verified\n");
```
* **Line 152**: `omp_set_dynamic(0)` disables dynamic thread adjustment, guaranteeing that OpenMP provides the exact number of threads requested.
* **Lines 153–157**: Creates CSV output files and writes header rows.

```c
159:     for (int si = 0; si < NSIZES; ++si) {
160:         int n = SIZES[si];
161:         size_t total = (size_t)n * (size_t)n;
162:         int8_t *A = (int8_t *)xaligned_alloc(64, total * sizeof(int8_t));
163:         int32_t *C = (int32_t *)xaligned_alloc(64, total * sizeof(int32_t));
164:         init_matrix(A, n);
```
* **Line 159**: Loops over matrix dimensions: $N = 2048$ and $N = 4096$.
* **Lines 162–163**: Allocates **64-byte aligned memory** using `posix_memalign()`. 64-byte alignment matches the CPU cache line and AVX-512 register boundary, preventing costly unaligned memory loads.
* **Line 164**: Initializes matrix $A$ with random values in $\{-1, 0, 1\}$.

```c
166:         uint64_t ref_hash = 0;
167:         int ref_ready = 0;
168: 
169:         benchmark_variant(runs, summary, "OMM", n, 0, A, C, &ref_hash, &ref_ready);
170:         for (int bi = 0; bi < NBLOCKS; ++bi) {
171:             char name[16];
172:             snprintf(name, sizeof(name), "BMM%d", BLOCKS[bi]);
173:             benchmark_variant(runs, summary, name, n, BLOCKS[bi], A, C,
174:                               &ref_hash, &ref_ready);
175:         }
176: 
177:         free(A);
178:         free(C);
179:     }
```
* **Line 169**: Benchmarks Ordinary Matrix Multiplication (OMM).
* **Lines 170–175**: Benchmarks Block Matrix Multiplication across all 5 block sizes (`BMM4`, `BMM8`, `BMM16`, `BMM32`, `BMM64`).
* **Lines 177–178**: Frees memory for the current size before moving to the next order.

```c
181:     fclose(runs);
182:     fclose(summary);
183:     return 0;
184: }
```
* **Lines 181–184**: Closes CSV file handles and exits successfully with return code 0.

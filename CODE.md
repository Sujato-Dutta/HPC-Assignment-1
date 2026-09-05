# CS4110 HPC Assignment 1 — Problem 1: Code Explanation (CODE.md)
## Simple, Line-by-Line Guide to `a1_matrix_square.c`

This guide explains the code in plain English. Each section breaks down:
1. **What the code is doing**
2. **Why we wrote it this way**
3. **Key takeaway to explain in the video**

---

## 1. Setup & Configuration (Lines 1–19)

```c
1: #define _POSIX_C_SOURCE 200112L
2: #include "common.h"
3: #include <inttypes.h>
```
* **Line 1**: Enables standard Linux system functions like aligned memory allocation (`posix_memalign`) and precise timers (`gettimeofday`).
* **Line 2**: Includes `common.h`, which contains shared helper functions (timers, PRNG random helper, and thread list `1, 2, 4, 6, 8, 10, 12, 14, 16`).
* **Line 3**: Allows us to use exact integer types like `int8_t` (1 byte) and `int32_t` (4 bytes).

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
* **Lines 11–15**: If we compile with `make quick`, it uses small matrix sizes (128, 256) for quick testing. Otherwise, it uses the full assignment sizes: **2048 x 2048** and **4096 x 4096**.
* **Lines 17–18**: The 5 block sizes required for Block Matrix Multiplication: **4, 8, 16, 32, and 64**.

---

## 2. Matrix Initialization: `init_matrix` (Lines 20–37)

```c
20: static void init_matrix(int8_t *A, int n) {
21:     const uint64_t seed = UINT64_C(0x123456789abcdef0);
22:     size_t total = (size_t)n * (size_t)n;
```
* **What it does**: Fills matrix $A$ with random values from `{-1, 0, 1}`.
* **Why**:
  * `seed`: Using a fixed starting number ensures every run generates the **exact same random matrix**, making verification fair and reproducible.
  * `total = n * n`: Total elements in the matrix (e.g., $4096 \times 4096 = 16,777,216$ numbers).

```c
24:     #pragma omp parallel num_threads(16)
25:     {
26:         int tid = omp_get_thread_num();
27:         int nt  = omp_get_num_threads();
28:         size_t begin = (total * (size_t)tid) / (size_t)nt;
29:         size_t end   = (total * (size_t)(tid + 1)) / (size_t)nt;
30:         uint64_t state = seed + (uint64_t)(begin + 1) * SM_GAMMA;
```
* **What it does**: Uses 16 OpenMP threads to fill the matrix in parallel.
* **How**:
  * It splits the matrix into equal slices `[begin, end)` for each thread (`tid`).
  * `state`: Gives each thread its own starting random position so threads don't interfere with each other.

```c
31:         for (size_t i = begin; i < end; ++i, state += SM_GAMMA) {
32:             uint64_t z = mix64(state);
33:             int v = (int)(((__uint128_t)z * 3u) >> 64); /* 0,1,2 */
34:             A[i] = (int8_t)(v - 1);
35:         }
36:     }
37: }
```
* **Line 32**: Generates a pseudo-random 64-bit integer `z`.
* **Line 33**: Converts that random number into `0`, `1`, or `2` without using slow division/modulo operators.
* **Line 34**: Subtracts 1 (`v - 1`), turning `{0, 1, 2}` into **`{-1, 0, 1}`**, and stores it as an 8-bit integer (`int8_t`).

---

## 3. Result Verification: `hash_i32` (Lines 39–47)

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
* **What it does**: Creates a single 64-bit fingerprint (checksum hash) from all elements of the output matrix $C$.
* **Why**: Instead of checking millions of numbers one by one, we compare this single 64-bit hash. If even one element in a multi-threaded run is wrong, the hash will completely change.

---

## 4. Ordinary Matrix Multiplication: `omm` (Lines 49–72)

```c
49: static void omm(const int8_t *restrict A, int32_t *restrict C, int n, int threads) {
50:     size_t total = (size_t)n * (size_t)n;
```
* **`restrict`**: Tells the compiler that array `A` and array `C` are separate blocks of memory, allowing the compiler to optimize register usage safely.
* **`int8_t *A` vs `int32_t *C`**: Input `A` uses 1 byte per entry (saving RAM). Output `C` uses 4 bytes so the accumulated sum never overflows.

```c
52:     #pragma omp parallel num_threads(threads)
53:     {
54:         #pragma omp for schedule(static)
55:         for (size_t q = 0; q < total; ++q) C[q] = 0;
```
* **Lines 52–55**: Wipes the output matrix `C` to all zeros in parallel before doing the multiplication.

```c
57:         /* i-k-j keeps C[i,*] and A[k,*] contiguous; no tiling is used. */
58:         #pragma omp for schedule(static)
59:         for (int i = 0; i < n; ++i) {
60:             int32_t *restrict ci = C + (size_t)i * n;
61:             const int8_t *restrict ai = A + (size_t)i * n;
```
* **Line 58 (`#pragma omp for schedule(static)`)**: Splits the rows of the matrix among the OpenMP threads. Thread 0 gets the top rows, Thread 1 gets the next rows, etc.
* **Lines 60–61**: Pointers to the start of row `i` for matrix $C$ (`ci`) and matrix $A$ (`ai`).

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
* **Line 62 (`The i-k-j order`)**: We put the `k` loop in the middle and `j` loop on the inside.
  * **Why?**: In C, arrays are row-major. The inner `j` loop moves horizontally along row `C[i]` and row `A[k]`. This allows the CPU to fetch continuous blocks of memory (cache lines) without jumping around.
* **Line 64 (`Zero-Bypass`)**: `if (aik == 0) continue;`
  * **Why?**: Since $1/3$ of our matrix elements are zero, multiplying by zero does nothing. Skipping zero cuts out $33\%$ of the work!
* **Line 66 (`#pragma omp simd`)**: Tells the CPU to use vector registers (AVX) to multiply 8 or 16 numbers in parallel in a single hardware cycle.

---

## 5. Block Matrix Multiplication: `bmm` (Lines 74–107)

```c
74: static void bmm(const int8_t *restrict A, int32_t *restrict C,
75:                 int n, int bs, int threads) {
```
* **`bs`**: Block size (tile size: 4, 8, 16, 32, or 64).

```c
84:         #pragma omp for collapse(2) schedule(static)
85:         for (int ii = 0; ii < n; ii += bs) {
86:             for (int jj = 0; jj < n; jj += bs) {
87:                 int iend = ii + bs < n ? ii + bs : n;
88:                 int jend = jj + bs < n ? jj + bs : n;
```
* **Line 84 (`collapse(2)`)**: Flattens the 2D grid of blocks (`ii` and `jj`) into a single list of tasks so OpenMP can distribute the square tiles evenly across all 16 threads.
* **Lines 85–88**: Loops over the matrix in chunks of `bs` (e.g. $64 \times 64$ tiles). `iend` and `jend` make sure we don't go past the edge of the matrix.

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
```
* **What it does**: Computes the matrix multiplication inside small $B \times B$ square tiles.
* **Why BMM is fast**: Because each $B \times B$ block is small, the entire sub-matrix stays inside the CPU's super-fast L1/L2 cache while computing, eliminating slow round-trips to main memory (RAM).

---

## 6. Benchmarking & CSV Output: `benchmark_variant` (Lines 109–149)

```c
116:     for (int ti = 0; ti < HPC_NTHREADS; ++ti) {
117:         int t = HPC_THREADS[ti];
118:         double tm[HPC_RUNS];
119:         int verified = 1;
```
* **What it does**: Loops through thread counts **1, 2, 4, 6, 8, 10, 12, 14, 16**.
* **`HPC_RUNS`**: Repeats each test **5 times** to get an accurate average.

```c
121:         for (int r = 0; r < HPC_RUNS; ++r) {
122:             double st = wall_time_sec();
123:             if (bs == 0) omm(A, C, n, t);
124:             else         bmm(A, C, n, bs, t);
125:             double et = wall_time_sec();
126:             tm[r] = et - st;
```
* **Lines 122–125**: Records the start time (`st`) and end time (`et`) using `gettimeofday()`.
* **Important**: Notice only the actual computation (`omm` or `bmm`) is inside the timer. Memory allocation and setup are outside the timed window.

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
```
* **Lines 128–134**: The very first 1-thread run establishes the "correct answer" hash (`reference_hash`). Every parallel run checks its output against this hash. If they match, `verified = 1`.
* **Line 135**: Writes the time for this individual run to `matrix_square_runs.csv`.

```c
140:         double avg = average_times(tm, HPC_RUNS);
141:         if (t == 1) baseline = avg;
142:         double sp = baseline / avg;
143:         fprintf(summary, "%s,%d,%d,%d,%.9f,%.6f,%d\n",
144:                 variant, n, bs, t, avg, sp, verified);
```
* **Lines 140–144**: Computes the 5-run average time, calculates the speedup ($S = T_1 / T_p$), and writes the summary to `matrix_square_summary.csv`.

---

## 7. Main Program: `main` (Lines 151–184)

```c
151: int main(void) {
152:     omp_set_dynamic(0);
```
* **`omp_set_dynamic(0)`**: Prevents the OpenMP library from reducing thread counts behind the scenes, ensuring the system strictly uses the exact number of threads requested (1 to 16).

```c
159:     for (int si = 0; si < NSIZES; ++si) {
160:         int n = SIZES[si];
161:         size_t total = (size_t)n * (size_t)n;
162:         int8_t *A = (int8_t *)xaligned_alloc(64, total * sizeof(int8_t));
163:         int32_t *C = (int32_t *)xaligned_alloc(64, total * sizeof(int32_t));
164:         init_matrix(A, n);
```
* **Lines 162–163 (`xaligned_alloc(64, ...)` )**: Allocates memory aligned to **64-byte boundaries**. This aligns with CPU cache lines, making vector memory reads as fast as possible.

```c
169:         benchmark_variant(runs, summary, "OMM", n, 0, A, C, &ref_hash, &ref_ready);
170:         for (int bi = 0; bi < NBLOCKS; ++bi) {
171:             char name[16];
172:             snprintf(name, sizeof(name), "BMM%d", BLOCKS[bi]);
173:             benchmark_variant(runs, summary, name, n, BLOCKS[bi], A, C,
174:                               &ref_hash, &ref_ready);
175:         }
176:         free(A); free(C);
177:     }
```
* **Lines 169–176**: Runs benchmarks for OMM first, then runs all 5 Block Matrix Multiplication sizes (`BMM4`, `BMM8`, `BMM16`, `BMM32`, `BMM64`), and frees memory cleanly.

```c
181:     fclose(runs);
182:     fclose(summary);
183:     return 0;
184: }
```
* **Lines 181–184**: Closes both CSV files and exits cleanly.

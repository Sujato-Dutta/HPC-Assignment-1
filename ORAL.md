# CS4110 HPC Assignment 1 — Problem 1: Matrix Squaring ($A^2$)
## Video Presentation Guide & Speech Script (2.5 – 3 Minutes)

---

## 📋 Problem Summary & Specifications
* **Goal**: Compute $A^2$ for matrix orders $N = 2048 \times 2048$ and $N = 4096 \times 4096$.
* **Input**: Matrix $A$ initialized with random values from $\{-1, 0, 1\}$ in row-major layout.
* **Implementations**:
  1. **Ordinary Matrix Multiplication (OMM)** — Non-blocked optimized $i\text{-}k\text{-}j$ loop order.
  2. **Block Matrix Multiplication (BMM)** — Tiled computation for block sizes $B \in \{4, 8, 16, 32, 64\}$.
* **Hardware Evaluated**: Mahindra University DGX A100 node (`dgxb`) with AMD EPYC server CPU cores across thread counts $T \in \{1, 2, 4, 6, 8, 10, 12, 14, 16\}$.

---

## 🎙️ Spoken Video Script (Timed for 2.5 to 3.0 Minutes)

> **Tip for Delivery**: Speak clearly, maintain an enthusiastic and confident tone, and point to the slides/code when mentioning optimizations and speedup numbers.

---

### [0:00 – 0:35] Introduction & Problem Formulation
> *"Hello everyone. Today, I am presenting **Problem 1 of our HPC Assignment**, which focuses on parallelizing the computation of $A^2$ for large matrix orders—$2048 \times 2048$ and $4096 \times 4096$.*
>
> *The elements of matrix $A$ are randomly drawn from the ternary set $\{-1, 0, 1\}$. We were tasked with implementing and evaluating two distinct computational approaches across thread counts from 1 to 16: first, **Ordinary Matrix Multiplication (OMM)**, and second, **Block Matrix Multiplication (BMM)** across five tile block sizes: 4, 8, 16, 32, and 64.*
>
> *Beyond simple parallelization, our goal was to achieve maximum hardware efficiency on the DGX server architecture."*

---

### [0:35 – 1:30] Key Optimizations: Memory, Cache Locality & SIMD
> *"To maximize performance and earn bonus marks for efficiency, we introduced four critical architectural optimizations:*
>
> 1. **Compact Data Types**: *Since input entries are strictly $\{-1, 0, 1\}$, we stored matrix $A$ using 8-bit integers (`int8_t`). For $N=4096$, this reduced the memory footprint from $64\text{ MB}$ down to just $16\text{ MB}$, allowing the entire input to fit comfortably within the L3 cache! For the accumulator and output matrix $C$, we used 32-bit integers (`int32_t`), ensuring mathematical exactness with zero overflow risk.*
>
> 2. **Loop Ordering ($i\text{-}k\text{-}j$)**: *Instead of traditional $i\text{-}j\text{-}k$ loops which traverse columns with stride-$N$ cache misses, our OMM uses an $i\text{-}k\text{-}j$ traversal. This ensures the innermost loop streams through both $C[i, \cdot]$ and $A[k, \cdot]$ sequentially with stride-1 contiguous cache line hits.*
>
> 3. **Zero-Sparsity Bypass**: *Because $A[i, k] \in \{-1, 0, 1\}$, approximately one-third of the elements are zero. If $A[i, k] == 0$, we bypass the entire inner $j$-loop, saving billions of compute cycles.*
>
> 4. **Tile Parallelization in BMM**: *In Block Matrix Multiplication, we collapsed the two outer tile loops (`#pragma omp for collapse(2)`), allowing threads to compute independent $B \times B$ tiles of $C$ with zero synchronization locks or false sharing.*"

---

### [1:30 – 2:25] DGX Benchmark Results & Speedup Analysis
> *"Now let's examine our experimental results from the DGX cluster:*
>
> *For $N=2048$, our sequential OMM baseline completed in **1.35 seconds**. Scaling across OpenMP threads, the runtime dropped to **0.17 seconds on 16 threads**, achieving an **8.00x linear speedup**.*
>
> *For the larger $N=4096$ matrix, OMM scaled from **16.77 seconds** on 1 thread down to **2.11 seconds on 16 threads** (**7.97x speedup**).*
>
> *When we look at Block Matrix Multiplication, small tile sizes like $B=4$ suffer from loop overhead. However, as block size increases to 16, 32, and 64, cache reuse improves dramatically:*
> * *At $B=64$ for $N=4096$, the 16-thread runtime dropped to just **1.20 seconds**, outperforming non-blocked OMM by nearly $2\times$!*
> * *Furthermore, for memory-bound block sizes, we observed super-linear speedup (up to **20.6x**) due to the aggregate L2/L3 cache capacity across all 16 parallel cores.*"

---

### [2:25 – 2:55] Verification & Conclusion
> *"To strictly adhere to the verification requirements, every single run was cross-verified against the sequential 1-thread reference using a 64-bit polynomial hash. All 540 experimental runs across both matrix sizes achieved a **100% verified status**.*
>
> *In summary, combining $i\text{-}k\text{-}j$ stride-1 locality, 8-bit memory compaction, and OpenMP 2D tile collapsing allowed us to compute $4096 \times 4096$ matrix squaring in just **1.2 seconds**. Thank you, and I will now hand over to my teammate for Problem 2."*

---

## 📊 Summary Data Table (For Slides / Reference)

### Matrix Order $N = 2048$

| Variant | Block Size | $T=1$ (sec) | $T=2$ | $T=4$ | $T=8$ | $T=16$ (sec) | Speedup ($T=16$) | Verified |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OMM** | — | 1.349 s | 0.618 s | 0.359 s | 0.182 s | **0.169 s** | **8.00x** | YES |
| **BMM4** | 4 | 23.257 s | 11.180 s | 5.751 s | 2.968 s | 2.879 s | 8.08x | YES |
| **BMM8** | 8 | 14.672 s | 7.595 s | 3.698 s | 1.924 s | 1.916 s | 7.66x | YES |
| **BMM16** | 16 | 5.578 s | 2.749 s | 1.441 s | 0.718 s | 0.767 s | 7.27x | YES |
| **BMM32** | 32 | 2.791 s | 1.347 s | 0.729 s | 0.378 s | 0.379 s | 7.36x | YES |
| **BMM64** | 64 | 2.015 s | 0.995 s | 0.506 s | 0.255 s | **0.280 s** | 7.21x | YES |

---

### Matrix Order $N = 4096$

| Variant | Block Size | $T=1$ (sec) | $T=2$ | $T=4$ | $T=8$ | $T=16$ (sec) | Speedup ($T=16$) | Verified |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OMM** | — | 16.769 s | 8.889 s | 5.280 s | 2.635 s | **2.105 s** | **7.97x** | YES |
| **BMM4** | 4 | 538.593 s | 270.081 s | 121.679 s | 45.796 s | 26.127 s | **20.62x** (super-linear) | YES |
| **BMM8** | 8 | 109.891 s | 55.689 s | 28.643 s | 14.245 s | 10.685 s | 10.28x | YES |
| **BMM16** | 16 | 37.863 s | 18.750 s | 9.499 s | 4.771 s | 3.551 s | 10.66x | YES |
| **BMM32** | 32 | 16.253 s | 8.096 s | 4.128 s | 2.090 s | 1.664 s | 9.77x | YES |
| **BMM64** | 64 | 10.698 s | 5.324 s | 2.706 s | 1.384 s | **1.201 s** | **8.91x** | YES |

---

## 💻 Code Walkthrough Cheat Sheet (Lines in `a1_matrix_square.c`)

If you are screen-sharing the code during your explanation:

1. **Lines 20–37 (`init_matrix`)**:
   * Uses deterministic PRNG (`mix64`) to initialize ternary values in parallel across threads.
2. **Lines 49–72 (`omm`)**:
   * `#pragma omp for schedule(static)` distributes outer $i$ rows evenly.
   * `aik == 0` check skips unnecessary computation for zero entries.
   * `#pragma omp simd` vectorizes the innermost contiguous $j$ loop across CPU SIMD registers.
3. **Lines 74–107 (`bmm`)**:
   * `#pragma omp for collapse(2)` flattens the 2D tile loop grid `(ii, jj)` across OpenMP threads.
   * Iterates through $B \times B$ submatrices (`bs = 4, 8, 16, 32, 64`) keeping active tile data inside L1/L2 caches.
4. **Lines 39–47 & 128–134 (`hash_i32` & verification)**:
   * Hashes the entire result matrix $C$ to verify that 16-thread outputs match the single-thread baseline bit-by-bit.

---

## ❓ Potential Professor / Evaluator Questions & Quick Answers

* **Q: Why does $i\text{-}k\text{-}j$ perform so much better than traditional $i\text{-}j\text{-}k$?**
  * **Answer**: In $i\text{-}j\text{-}k$, accessing column $k$ of matrix $B$ incurs a stride-$N$ jump in memory on every iteration, leading to constant L1/L2 cache misses. In $i\text{-}k\text{-}j$, the innermost loop increments $j$, which reads row $A[k, \cdot]$ and writes row $C[i, \cdot]$ contiguously (stride 1) along cache lines, enabling hardware prefetching and AVX SIMD vectorization.

* **Q: Why did BMM64 outperform non-blocked OMM for $N=4096$?**
  * **Answer**: At $N=4096$, the matrix size ($16\text{ MB}$) exceeds L2 cache size. Block size 64 fits $64 \times 64$ sub-blocks ($4\text{ KB}$) entirely within the CPU's fast L1/L2 caches, eliminating repeated memory trips to main RAM.

* **Q: Why did BMM4 show a super-linear speedup of 20.6x on 16 threads?**
  * **Answer**: On 1 thread, the large footprint of managing millions of tiny $4 \times 4$ blocks caused cache eviction thrashing. When split across 16 threads, the working set per thread dropped to $1/16\text{th}$, allowing it to fit into the independent L2 caches of each core, providing super-linear scaling.

# CS4110 HPC Assignment 1 — Problem 1: Matrix Squaring (A^2)
## Video Presentation Speech & Talking Points (2.5 – 3 Minutes)

---

## 🎯 High-Level Overview (What to Keep in Mind)
* **What we are doing**: Computing the square of a matrix ($A \times A = A^2$) for two sizes: $2048 \times 2048$ and $4096 \times 4096$.
* **Matrix values**: Random numbers that are only `-1`, `0`, or `1`.
* **Two algorithms compared**:
  1. **OMM (Ordinary Matrix Multiplication)**: Standard matrix multiplication with a smart loop order.
  2. **BMM (Block Matrix Multiplication)**: Chopping the matrix into small square blocks ($4, 8, 16, 32, 64$) so they fit into the CPU cache.
* **Goal**: Parallelize both using OpenMP on 1 to 16 threads, make them as fast as possible, and prove they give the exact right answer.

---

## 🎙️ Spoken Video Script (Simple & Easy to Memorize)

> **Speaking Tip**: Speak at a steady pace. Use your hands or cursor to point at the code or slides.

---

### [0:00 – 0:40] Introduction
> "Hello everyone! Today I will be explaining **Problem 1 of our HPC Assignment**, which is calculating **Matrix Square ($A^2$)** for two matrix sizes: $2048 \times 2048$ and $4096 \times 4096$.
>
> All elements in matrix $A$ are randomly chosen from the values **minus one, zero, or one**, stored in row-major order.
>
> Our objective is to implement two parallel approaches using OpenMP across 1 to 16 threads:
> 1. First, **Ordinary Matrix Multiplication (OMM)**.
> 2. Second, **Block Matrix Multiplication (BMM)** across block sizes of 4, 8, 16, 32, and 64.
>
> Let me walk you through how we designed our code and the key optimizations we added to achieve maximum performance."

---

### [0:40 – 1:50] The 4 Main Optimizations (The Core Technical Part)
> "To make our code run as fast as possible, we focused on **four main optimizations**:
>
> **1. Smart Data Types (Saving Memory)**
> Since our input values are only `-1`, `0`, or `1`, storing them as normal 4-byte integers wastes memory. Instead, we used **`int8_t` (1 byte per element)**. For a $4096 \times 4096$ matrix, this cut our memory from $64\text{ MB}$ down to just $16\text{ MB}$, allowing the matrix to easily fit inside the CPU's fast cache! For the output matrix $C$, we used `int32_t` so the sum never overflows.
>
> **2. The `i-k-j` Loop Order (Cache Locality)**
> In standard matrix multiplication, you usually see an `i-j-k` loop. But that forces the CPU to jump down columns in memory, causing huge cache delays. 
> Instead, we changed the loop order to **`i-k-j`**. In this order, the innermost loop reads row-by-row sequentially. Because data is continuous in memory, the CPU pre-loads it into cache effortlessly.
>
> **3. Skipping Zeros (Zero-Bypass)**
> Since our matrix entries are `-1`, `0`, and `1`, roughly **one-third of all numbers are zero**. If we multiply by zero, the answer is zero. So we added a simple check: if the value is `0`, we skip that whole row calculation immediately! This saves billions of unnecessary multiplications.
>
> **4. 2D Block Tiling in BMM**
> In Block Matrix Multiplication, we chop the huge matrix into small tiles of size $4, 8, 16, 32,$ or $64$. We use OpenMP's `collapse(2)` directive to distribute independent output blocks across all CPU threads. Each thread works on its own tile without needing any locks or synchronization."

---

### [1:50 – 2:30] SIMD Vectorization & Verification
> "In addition to multi-threading with OpenMP, we also added the `#pragma omp simd` directive to the inner loop. This allows the CPU to use its vector hardware units to process multiple numbers simultaneously in a single clock cycle.
>
> **How did we verify correctness?**
> We must prove that our parallel code produces the exact same answer as sequential code. We created a fast 64-bit fingerprint hash that scans every element of the output matrix. Every single parallel run across all thread counts matched the 1-thread sequential baseline with **100% accuracy**."

---

### [2:30 – 2:50] Conclusion
> "In conclusion, by combining:
> * 8-bit memory compaction,
> * the cache-friendly `i-k-j` loop order,
> * zero-skipping, and
> * OpenMP multi-threading with block tiling,
>
> we built an ultra-fast, memory-efficient solution for large matrix squaring. 
>
> Thank you! I will now pass it on to my teammate for Problem 2."

---

## 🧠 Easy Cheat-Sheet: 5 Points to Memorize

If you forget your script, just remember these 5 bullet points:

1. **Problem**: Square a matrix ($A \times A$) for sizes 2048 and 4096 with values in $\{-1, 0, 1\}$.
2. **Data Type**: Used `int8_t` (1 byte) for input matrix $A$ to save RAM and fit in CPU cache; used `int32_t` for output $C$ to avoid overflow.
3. **Loop Order**: Used `i-k-j` instead of `i-j-k` so memory is accessed in straight horizontal lines (stride 1) rather than jumping columns.
4. **Zero Skip**: If $A[i][k] == 0$, skip the inner loop (cuts out $\sim 33\%$ of the math).
5. **OpenMP & Tiling**: Used `#pragma omp for collapse(2)` to divide $B \times B$ tiles among threads, plus `#pragma omp simd` for CPU vector instructions.

---

## ❓ Simple Answers to Possible Questions

* **Q: Why is `i-k-j` faster than `i-j-k`?**
  * **Answer**: In C, matrices are stored row-by-row (row-major). The `i-k-j` order reads and writes along horizontal rows, which matches how the CPU loads data into cache lines. The `i-j-k` order jumps down columns, causing constant cache misses.

* **Q: What is the purpose of Block Matrix Multiplication (BMM)?**
  * **Answer**: When matrices are very large, they don't fit into the CPU's fast L1/L2 cache. BMM divides the matrix into smaller square chunks (tiles) that stay inside cache while computing, reducing slow trips to main RAM.

* **Q: Why use `int8_t` instead of regular `int`?**
  * **Answer**: Since values are only $-1, 0, 1$, they easily fit in 1 byte (`int8_t`). A regular 4-byte `int` would make the matrix 4 times larger ($64\text{ MB}$ vs $16\text{ MB}$), which slows down memory transfer.

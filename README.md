# CS4110 HPC Assignment 1 — OpenMP C Solutions

These programs implement all five assignment problems in **C + OpenMP** and use `gettimeofday()` for timing. Default builds use the exact assignment sizes, thread counts **1,2,4,6,8,10,12,14,16**, and **5 runs per configuration**.

## Build

```bash
make
```

For correctness/testing only (small sizes, 2 runs, threads 1/2/4):

```bash
make quick
```

Before final lab runs, rebuild exact versions:

```bash
make clean && make
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

Run each executable separately. Every program writes a `*_runs.csv` containing every run and a `*_summary.csv` containing average runtime, speedup, and verification status. Import these CSV files into the required Excel workbook.

## Q1 — `a1_matrix_square.c`

- Computes `A^2` for 2048 and 4096.
- A is row-major and stored as `int8_t` because values are only {-1,0,1}.
- Result uses `int32_t`; each entry is at most +/-N, so this is exact.
- OMM: cache-friendly non-blocked `i-k-j` multiplication.
- BMM: block sizes 4,8,16,32,64; independent output tiles are parallelized.
- A thread-1 OMM result is the sequential reference; every result is checked against its hash.

## Q2 — `a2_complex_power15.c`

- Uses separate real/imaginary `double` matrices.
- Uses the minimal 5-multiplication addition chain:
  `X2=X*X`, `X3=X2*X`, `X6=X3*X3`, `X12=X6*X6`, `X15=X12*X3`.
- Blocked complex matrix multiplication with block size 32.
- Thread-1 output is the sequential reference; all thread counts must match its fingerprint.

## Q3 — `a3_sort_counting.c`

- Exact counting sort for 10^12 integers in [0,1000].
- Explicitly storing 10^12 integers would require multiple TB; counting sort only needs 1001 counters.
- A deterministic counter-based random sequence gives the identical input for every thread count.
- Each thread owns a padded local histogram; no atomics in the hot loop.
- The sorted sequence is represented exactly by counts for values 0..1000.

## Q4 — `a4_primes_segmented.c`

- Odd-only segmented Sieve of Eratosthenes for [2^40,2^41).
- Base primes are generated only up to sqrt(2^41).
- Segments are dynamically distributed among OpenMP threads.
- The interval contains far too many primes to print/store efficiently; every prime is still generated/visited. Verification uses count plus two order-independent 64-bit fingerprints.
- Printing the primes should not be included in benchmarking because I/O would dominate the computation.

## Q5 — `a5_dot_product.c`

- Vector length 10^9.
- Inputs use `int8_t` since values are {-1,0,1}, requiring about 2 GB total instead of 8 GB with two `int32_t` vectors.
- `int64_t` accumulation + OpenMP reduction + SIMD.
- Vector initialization is performed once outside the timed dot-product region.

## Timing / fairness

- Computation only is timed; input initialization and result verification are outside the timing region.
- Any algorithmic transformation needed by a computation would need to be timed. These implementations do not transpose matrices or perform an untimed layout conversion.
- For each variant, speedup is `average_time(1 thread) / average_time(T threads)`.

## GNUplot

For Q2/Q3/Q5 summary files:

```bash
gnuplot -e "file='dot_product_summary.csv';out='dot_speedup.png';title='Dot Product Speedup'" plot_speedup.gp
```

Q1 summary columns are: `variant,N,block,threads,avg_sec,speedup,verified`, so use `using 4:6` and filter/plot each variant and N separately.

Q4 summary columns are: `variant,low,high,threads,avg_sec,speedup,count,verified`, so use `using 4:6`.

## Important practical note

Q3 (10^12 values) and Q4 (an interval of length 2^40) are intentionally huge. Even efficient algorithms can take substantial time. Run the final exact build only on the IT-lab system, not on a laptop. `make quick` exists only to validate correctness before the final runs.

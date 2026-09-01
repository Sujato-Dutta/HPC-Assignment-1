CC      = gcc
CFLAGS  = -O3 -march=native -std=c11 -Wall -Wextra -fopenmp
LDLIBS  = -lm

TARGETS = a1_matrix_square a2_complex_power15 a3_sort_counting a4_primes_segmented a5_dot_product

all: $(TARGETS)

a1_matrix_square: a1_matrix_square.c common.h
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

a2_complex_power15: a2_complex_power15.c common.h
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

a3_sort_counting: a3_sort_counting.c common.h
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

a4_primes_segmented: a4_primes_segmented.c common.h
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

a5_dot_product: a5_dot_product.c common.h
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

quick:
	$(MAKE) clean
	$(MAKE) CFLAGS="$(CFLAGS) -DQUICK"

clean:
	rm -f $(TARGETS) *.csv *.png

.PHONY: all quick clean

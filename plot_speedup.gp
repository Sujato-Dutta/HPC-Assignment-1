# Usage example:
# gnuplot -e "file='dot_product_summary.csv';out='dot_speedup.png';title='Dot Product Speedup'" plot_speedup.gp
if (!exists("file")) file = "dot_product_summary.csv"
if (!exists("out")) out = "speedup.png"
if (!exists("title")) title = "OpenMP Speedup"
set datafile separator comma
set terminal pngcairo size 1000,650
set output out
set title title
set xlabel "Threads"
set ylabel "Speedup"
set grid
set key left top
# Summary files place threads in column 3 for Q2/Q3/Q5, but Q1/Q4 differ.
# For Q2/Q3/Q5 use this script directly. For Q1/Q4 adjust using columns as noted in README.
plot file every ::1 using 3:5 with linespoints lw 2 pt 7 title "Measured", \
     x with lines dt 2 title "Ideal"

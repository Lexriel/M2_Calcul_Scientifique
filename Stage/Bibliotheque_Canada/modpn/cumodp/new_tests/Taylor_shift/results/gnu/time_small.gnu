	set terminal postscript colour  "Arial" 12
	set terminal postscript eps 
	set output "eps/time_small_n.eps"
	set grid
	set key left
	set title "" 
	set xlabel "n degree of the polynomial"
	set ylabel "CPU computing time with prime: 958922753"
	set style line 5 lt 1 lw 3 linecolor rgb "red"
	set style line 6 lt 1 lw 3 linecolor rgb "blue"
	set style line 7 lt 1 lw 3 linecolor rgb "green"
	plot  "dat/time_small.dat" using 2:3 with lines title 'GPU Divide and Conquer' ls 7, "dat/time_small.dat" using 2:4 with lines title 'CPU Horner' ls 6, "dat/time_small.dat" using 2:5 with lines title 'CPU Divide and Conquer' ls 5

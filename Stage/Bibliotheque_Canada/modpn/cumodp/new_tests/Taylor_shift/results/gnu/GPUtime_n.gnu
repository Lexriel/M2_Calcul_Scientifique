	set terminal postscript colour  "Arial" 12
	set terminal postscript eps 
	set output "eps/GPUtime_n.eps"
	set grid
	set key left
	set title "" 
	set xlabel "n degree of the polynomial"
	set ylabel "CPU computing time with prime: 958922753"
	set style line 6 lt 1 lw 3 linecolor rgb "green"
	plot  "dat/time.dat" using 2:3 with lines title 'GPU Divide and Conquer' ls 6

	set terminal postscript colour  "Arial" 12
	set terminal postscript eps 
	set output "cdCPUtime.eps"
	set grid
	set key left
	set title "" 
	set xlabel "log2(n) with n degree of the polynomial"
	set ylabel "CPU computing time with prime: 958922753"
	set style line 5 lt 0 lw 2
	set style line 6 lt 1 lw 4
	plot  "dat/time.dat" using 1:3 with lines title 'CPU Horner' ls 6, "dat/time.dat" using 1:4 title 'CPU Divide and Conquer' ls 5

	set terminal postscript colour  "Arial" 12
	set terminal postscript eps 
	set output "eps/CPUtime_n.eps"
	set grid
	set key left
	set title "" 
	set xlabel "n degree of the polynomial"
	set ylabel "CPU computing time with prime: 958922753"
	set style line 5 lt 1 lw 3 linecolor rgb "red"
	set style line 6 lt 1 lw 3 linecolor rgb "blue"
	plot  "dat/time.dat" using 2:4 with lines title 'CPU Horner' ls 6, "dat/time.dat" using 2:5 with lines title 'CPU Divide and Conquer' ls 5

set terminal png size 600, 600
set output "fft_mul.png"
set datafile separator " "
set size square 1.0, 1.0
set title "Computation Time for Stockham-FFT based Polynomial Multiplication"
set xlabel "Length of Each Polynomial" 0.0,0.0
set ylabel "Time" 0.0,0.0
set key left top box
set pointsize 1.8
plot [0:32] [0:32] x title "time" lt 2, 1001.259736 notitle lt 2

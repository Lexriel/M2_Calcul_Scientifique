set term post eps color 
set output "fast_evaluation_step.eps"
set datafile separator " "
set size square 1.0, 1.0
set title "Time consumption on Each level in Fast Evaluation"
set key left top box
set pointsize 1.8
#set style data linespoints
set xlabel "Level in subproduct tree"
set ylabel "Time consumption on the level (ms)"
set xrange [22:1]
plot   "fast_evaluation_step.dat" with points pt 2 title "time consumption" 

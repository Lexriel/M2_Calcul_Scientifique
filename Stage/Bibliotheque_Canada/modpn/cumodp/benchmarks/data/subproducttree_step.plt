set term post eps color 
set output "sub_product_tree_step.eps"
set datafile separator " "
set size square 1.0, 1.0
set title "Time consumption on Each level of subproduct tree"
set key left top box
set pointsize 1.8
#set style data linespoints
set xlabel "Level in subproduct tree"
set ylabel "Time consumption on the level (ms)"
plot [1:22] [0:600] "subproducttree_step.dat" with points pt 2 title "time consumption" 

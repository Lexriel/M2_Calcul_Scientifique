# update the directory
svn up

# compilation
make all

# remove a strange file
rm *.linkinfo

# variables
doc='screen.txt'
doc2='diff.txt'
let "a = 10"
let "b = a - 1"
let "p = 5000011"

# create polynomials of degree 2^i with i=3..$b
./aleat_pol $a 20

# save the display on the screen in a file
echo "rm taylor_shift_cpu" > $doc
echo "rm taylor_shift_gpu" >> $doc
echo "svn up" >> $doc
echo "g++ aleat_pol.cpp -O3 -o aleat_pol" >> $doc
echo "g++ taylor_shift_1.cpp -O3 -o taylor_shift_cpu" >> $doc
echo "nvcc taylor_shift.cu -O3 -o taylor_shift_gpu" >> $doc
echo "rm *.linkinfo" >> $doc
echo "./aleat_pol 15 20" >> $doc

# compare the created files
for i in `seq 3 $b`
do
	echo "./taylor_shift_cpu Pol$i.dat $p" >> $doc
	./taylor_shift_cpu Pol$i.dat $p ARGS | tee >> $doc
	echo "./taylor_shift_gpu Pol$i.dat $p" >> $doc
	./taylor_shift_gpu Pol$i.dat $p ARGS | tee >> $doc
done

# compare the created files
echo "DIFFERENCES BETWEEN THE FILES" > $doc2
for i in `seq 3 $b`
do
	echo "CASE n = 2^$i" >> $doc2
	echo "diff Pol$i.shiftHOR.dat Pol$i.shiftGPU.dat" >> $doc2
	diff Pol$i.shiftHOR.dat Pol$i.shiftGPU.dat | tee >> $doc2
	echo "" >> $doc2
done

# clean
rm taylor_shift_cpu
rm taylor_shift_gpu
rm aleat_pol


# This is to benchmark the subproduct tree construction 
# Author: Jiajian Yang <jyang425@csd.uwo.ca>

TARGET="subproduct_tree"
TEST=$TARGET"_tst"
SRC="../src/"$TARGET".cu"
LOG=$TEST"_step.log"

TEMP_TEST_SRC="src.cu"

BEGINFLAG="BEGIN:"$TEST
ENDFLAG="END:"$TEST

# compiler
CC=nvcc

# This is for the beauty in console
for i in $(seq 1 40);
do
	printf "="
done
printf "\n"

#sed -i "s/\/\*Measure Step/\/\*Measure Step\*\//g" ../src/subproduct_tree.cu

#sed -i "s/Measure Step\*\//\/\*Measure Step\*\//g" ../src/subproduct_tree.cu

# Re-make the so
make

# include header files
printf "#include \"../include/subproduct_tree.h\"\n" > $TEMP_TEST_SRC

# find the test function in the source file by using flag
lineA=$(cat $SRC | grep -n $BEGINFLAG -o | grep "[0-9]\+" -o)
lineB=$(cat $SRC | grep -n $ENDFLAG -o | grep "[0-9]\+" -o)

# now we know where the test function is 
codelines=$(($lineB - $lineA))

# we cat the specific test function
cat ${SRC} | grep ${BEGINFLAG} -A $(($codelines + 1)) -B 1  >> $TEMP_TEST_SRC

# this is for the main function. 
printf "int main (int argc, char *argv[]){\n" >> $TEMP_TEST_SRC
printf "sfixn p = atoi (argv[1]);\n" >> $TEMP_TEST_SRC
printf "sfixn k = atoi (argv[2]);\n" >> $TEMP_TEST_SRC
printf "subproduct_tree_bench(p,k);\n" >> $TEMP_TEST_SRC
printf "return 0;\n }\n" >> $TEMP_TEST_SRC

# compile the test
$CC -DLINUXINTEL64 ./$TEMP_TEST_SRC -o bin/${TEST}  -L./bin/ -lcumodp -g

# after compilation, the test is located in bin/
cd bin/

date >> ../$LOG


# declearing tem files
TMP=cudaresult.tmp

# the p is for the prime number, m stands for how many ffts we are going to do at the same tim
# k is the length of each FFT, BTW, the length of each FFT will be 2^k
# well I want 257 and 469762049 only. The 469761000 is not a magic number
# It's just for convenient for the for loop
#for p in $(seq 257 469761000 469762049); 
p=469762049
#do
for n in $(seq 24 25);
do
	./$TEST $p $n > $TMP
	echo "k = $n"
	ll=$(cat $TMP | grep "LAYER")
	echo $ll >> ../$LOG
	echo $ll 
done

echo "" >> ../$LOG

echo "Please check the test log:"$LOG

# End of the test
for i in $(seq 1 40);
do
	printf "="
done
printf "\n"

cd ..

# Remove the garbage and we are done :)
rm -f $TEMP_TEST_SRC
rm -f bin/$TMP
rm -f ./*.linkinfo
rm -f ./src.cu

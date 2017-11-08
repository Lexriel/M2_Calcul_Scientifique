# Here is the description of your test blablabla :)
# Author: Jiajian Yang <jyang425@csd.uwo.ca>

TARGET="fast_evaluation"
TEST=$TARGET"_tst"
SRC="../src/"$TARGET".cu"
LOG=$TEST".log"

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

# Re-make the so
make 

# include header files
printf "#include <cassert>\n#include <iostream>\n" > $TEMP_TEST_SRC
printf "#include \"../include/defines.h\"\n" >> $TEMP_TEST_SRC
printf "#include \"../include/inlines.h\"\n#include \"../include/printing.h\"\n" >> $TEMP_TEST_SRC
printf "#include \"../include/cudautils.h\"\n#include \"../include/fft_aux.h\"\n" >> $TEMP_TEST_SRC
printf "#include \"../include/fast_evaluation.h\"\n" >> $TEMP_TEST_SRC

# find the test function in the source file by using flag
lineA=$(cat $SRC | grep -n $BEGINFLAG -o | grep "[0-9]\+" -o)
lineB=$(cat $SRC | grep -n $ENDFLAG -o | grep "[0-9]\+" -o)

# now we know where the test function is 
codelines=$(($lineB - $lineA))

# we cat the specific test function
cat ${SRC} | grep ${BEGINFLAG} -A $(($codelines + 1)) -B 1  >> $TEMP_TEST_SRC

# this is for the main function. 
printf "int main (int argc, char *argv[]){\n" >> $TEMP_TEST_SRC
printf "sfixn k = atoi (argv[1]);\n" >> $TEMP_TEST_SRC
printf "fast_evaluation_bench(k);\n" >> $TEMP_TEST_SRC
printf "return 0; }\n" >> $TEMP_TEST_SRC

# compile the test
$CC -DLINUXINTEL64 ./$TEMP_TEST_SRC -o bin/${TEST}  -L./bin/ -lcumodp

# after compilation, the test is located in bin/
cd bin/

date >> ../$LOG

TMP=result.tmp

RESULT=PASS
# the p is for the prime number, m stands for how many ffts we are going to do at the same time
# k is the length of each FFT, BTW, the length of each FFT will be 2^k
# well I want 257 and 469762049 only. The 469761000 is not a magic number
# It's just for convenient for the for loop
for k in $(seq 2 23); 
do
	./$TEST $k > $TMP
	ll=$(cat $TMP | grep "TIME")
	t=$(echo $ll | sed -e "s|TIME USED:||")
	echo "$k $t"  >> ../$LOG
	echo "k= $k Time =$t ms"
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
rm -f bin/$TEST

# Here is the description of your test blablabla :)
# Author: Jiajian Yang <jyang425@csd.uwo.ca>

TARGET="list_expand_to_fft"
TEST=$TARGET"_tst"
SRC="../src/list_pointwise_mul.cu"
LOG="log/"$TEST".log"
OUT="log/"$TEST"_console.log"

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
printf "#include \"../include/list_pointwise_mul.h\"\n" >> $TEMP_TEST_SRC

# find the test function in the source file by using flag
lineA=$(cat $SRC | grep -n $BEGINFLAG -o | grep "[0-9]\+" -o)
lineB=$(cat $SRC | grep -n $ENDFLAG -o | grep "[0-9]\+" -o)

# now we know where the test function is 
codelines=$(($lineB - $lineA))

# we cat the specific test function
cat ${SRC} | grep ${BEGINFLAG} -A $(($codelines + 1)) -B 1  >> $TEMP_TEST_SRC

# this is for the main function. 
printf "int main (int argc, char *argv[]){\n" >> $TEMP_TEST_SRC
printf "sfixn m = atoi (argv[1]);\n" >> $TEMP_TEST_SRC
printf "sfixn k = atoi (argv[2]);\n" >> $TEMP_TEST_SRC
printf $TEST"(m,k);\n" >> $TEMP_TEST_SRC
printf "return 0;\n }\n" >> $TEMP_TEST_SRC

# compile the test
$CC -DLINUXINTEL64 ./$TEMP_TEST_SRC -o bin/${TEST}  -L./bin/ -lcumodp

# after compilation, the test is located in bin/
cd bin/

date >> ../$LOG
date >> ../$OUT

TMP=result.tmp

RESULT=PASS
FAILS=0

# the p is for the prime number, m stands for how many ffts we are going to do at the same time
# k is the length of each FFT, BTW, the length of each FFT will be 2^k
# well I want 257 and 469762049 only. The 469761000 is not a magic number
# It's just for convenient for the for loop
for k in $(seq 6 15);
do
	for ll in $(seq 2 4);
	do
		m=$((2**$(($k-$ll))))
		./$TEST $m $k > $TMP
		cat $TMP >> ../$OUT
		rr=$(cat $TMP | grep fail | wc -l)
		if [ $rr -gt 0 ];
		then
			RESULT=FAIL
			FAILS=$(($FAILS+1))
		else
			RESULT=PASS
		fi
		echo "m="$m",k="$k :" " $RESULT >> ../$LOG
		echo "m="$m",k="$k :" " $RESULT 
	done
done
for k in 16 19 20 24;
do
	for ll in $(seq 1 4);
	do
		m=$((2**$(($k-$ll))))
		./$TEST $m $k > $TMP
		cat $TMP >> ../$OUT
		rr=$(cat $TMP | grep fail | wc -l)
		if [ $rr -gt 0 ];
		then
			RESULT=FAIL
			FAILS=$(($FAILS+1))
		else
			RESULT=PASS
		fi
		echo "m="$m",k="$k :" " $RESULT >> ../$LOG
		echo "m="$m",k="$k :" " $RESULT 
	done
done

echo "" >> ../$LOG

echo "The result of this test is $RESULT."
echo "Please check the test log:"$LOG
echo "Since the console output might be horribly long, the log of console is stored in "$OUT 

# End of the test
for i in $(seq 1 40);
do
	printf "="
done
printf "\n"

cd ..

# Remove the garbage and we are done :)
rm -f $TEMP_TEST_SRC
#rm -f bin/$TEST

# Here is the description of your test blablabla :)
# Author: Jiajian Yang <jyang425@csd.uwo.ca>

TARGET="list_shrink_after_invfft"
TEST=$TARGET"_tst"
SRC="../src/list_pointwise_mul.cu"
LOG="log/"$TEST".log"
OUT="log/list_shrink_poly_console.log"

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
printf "sfixn ln = atoi (argv[2]);\n" >> $TEMP_TEST_SRC
printf "sfixn ll = atoi (argv[3]);\n" >> $TEMP_TEST_SRC
printf $TEST"(m,ln,ll);\n" >> $TEMP_TEST_SRC
printf "return 0;\n }\n" >> $TEMP_TEST_SRC

# compile the test
$CC -DLINUXINTEL64 ./$TEMP_TEST_SRC -o bin/${TEST}  -L./bin/ -lcumodp

# after compilation, the test is located in bin/
cd bin/

date >> ../$LOG
date >> ../$OUT

RESULT=PASS
FAILS=0
TMP=result.tmp

# the p is for the prime number, m stands for how many ffts we are going to do at the same time
# k is the length of each FFT, BTW, the length of each FFT will be 2^k
# well I want 257 and 469762049 only. The 469761000 is not a magic number
# It's just for convenient for the for loop
for m in 2 4 8 16 32 256;
do
	for ln in 32 64;
	do
		for ll in 9 13 15;
		do
			./$TEST $m $ln $ll > $TMP
			rr=$(cat $TMP | grep Fail | wc -l)
			if [ $rr -gt 0 ];
			then
				RESULT=FAIL
				FAILS=$(($FAILS+1))
			else
				RESULT=PASS
			fi
			cat $TMP >> ../$OUT
			echo "m = $m, ln = $ln, ll = $ll : $RESULT" >> ../$LOG
			echo "m = $m, ln = $ln, ll = $ll : $RESULT" 
		done
	done
done

for m in 32 512 1024 $((2**14)) ;
do
	for ln in 1024 2048;
	do
		for ll in 512;
		do
			./$TEST $m $ln $ll > $TMP
			rr=$(cat $TMP | grep Fail | wc -l)
			if [ $rr -gt 0 ];
			then
				RESULT=FAIL
				FAILS=$(($FAILS+1))
			else
				RESULT=PASS
			fi
			cat $TMP >> ../$OUT
			echo "m = $m, ln = $ln, ll = $ll : $RESULT" >> ../$LOG
			echo "m = $m, ln = $ln, ll = $ll : $RESULT" 
		done
	done
done

for m in 2 4 8;
do
	for ln in $((2**23)) $((2**24));
	do
		for ll in $((2**22));
		do
			./$TEST $m $ln $ll > $TMP
			rr=$(cat $TMP | grep Fail | wc -l)
			if [ $rr -gt 0 ];
			then
				RESULT=FAIL
				FAILS=$(($FAILS+1))
			else
				RESULT=PASS
			fi
			cat $TMP >> ../$OUT
			echo "m = $m, ln = $ln, ll = $ll : $RESULT" >> ../$LOG
			echo "m = $m, ln = $ln, ll = $ll : $RESULT" 
		done
	done
done

echo "" >> ../$LOG

if [ $FAILS -gt 0 ];
then
	RESULT=FAIL
else
	RESULT=PASS
fi

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
rm -f bin/$TEST
rm -f bin/$TMP

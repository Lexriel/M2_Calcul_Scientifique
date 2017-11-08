# Here is the description of your test blablabla :)
# Author: Jiajian Yang <jyang425@csd.uwo.ca>

TARGET="list_fast_division"
TEST=$TARGET"_tst"
SRC="../src/"$TARGET".cu"
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
printf "#include \"../include/list_fast_division.h\"\n" >> $TEMP_TEST_SRC

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
printf "sfixn m = atoi (argv[2]);\n" >> $TEMP_TEST_SRC
printf "sfixn k = atoi (argv[3]);\n" >> $TEMP_TEST_SRC
printf $TEST"(p,m,k);\n" >> $TEMP_TEST_SRC
printf "return 0; }\n" >> $TEMP_TEST_SRC

# compile the test
$CC -DLINUXINTEL64 ./$TEMP_TEST_SRC -o bin/${TEST}  -L./bin/ -lcumodp

# after compilation, the test is located in bin/
cd bin/

date >> ../$LOG
date >> ../$OUT

TMP=result.tmp
RR=mapleresult.tmp
MW=com.mw

RESULT=PASS
ERRS=0
# the p is for the prime number, m stands for how many ffts we are going to do at the same time
# k is the length of each FFT, BTW, the length of each FFT will be 2^k
# well I want 257 and 469762049 only. The 469761000 is not a magic number
# It's just for convenient for the for loop
p=469762049; 
#	./$TEST $p $k $m > $TMP

for k in 3 5 8 10 12;
do
	for m in 2 4 8 10;
	do
		./$TEST $p $m $k> $TMP
		cat $TMP >> ../$OUT
		cat $TMP | grep -e "\[" > $MW
		echo "err:=0;" >> $MW
		echo "for jj from 0 to $(($m-1)) do" >> $MW
		echo "if Rm[jj] <> r[jj] then err := 1; end if;" >> $MW
		echo "od; " >> $MW
		echo "if err = 0 then fprintf(\"$RR\",\"PASS\"); else fprintf(\"$RR\",\"FAIL\"); end if;" >> $MW
		maple $MW > /dev/null
		RESULT=$(cat $RR)
		if [ $RESULT = "FAIL" ];
		then
			ERRS=$(($ERRS+1))
		fi
		echo "k=$k, m=$m : $RESULT" >> ../$LOG
		echo "k=$k, m=$m : $RESULT" 
	done
done

echo "" >> ../$LOG

if [ $ERRS -gt 0 ];
then
	RESULT=FAIL
fi

echo "The result of this test is $RESULT"
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

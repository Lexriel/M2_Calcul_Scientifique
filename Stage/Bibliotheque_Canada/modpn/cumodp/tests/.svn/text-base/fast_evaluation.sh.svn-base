# Here is the description of your test blablabla :)
# Author: Jiajian Yang <jyang425@csd.uwo.ca>

TARGET="fast_evaluation"
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
printf $TEST"(k);\n" >> $TEMP_TEST_SRC
printf "return 0; }\n" >> $TEMP_TEST_SRC

# compile the test
$CC -DLINUXINTEL64 ./$TEMP_TEST_SRC -o bin/${TEST}  -L./bin/ -lcumodp

# after compilation, the test is located in bin/
cd bin/

date >> ../$LOG
date >> ../$OUT

cp ../fast_rem.txt ./
TMP=result.tmp

RESULT=PASS

MW=com.mw
RR=mapleresult.tmp
touch $RR

RS=subresult.tmp
touch $RS
MWW=sub.mw
# the p is for the prime number, m stands for how many ffts we are going to do at the same time
# k is the length of each FFT, BTW, the length of each FFT will be 2^k
# well I want 257 and 469762049 only. The 469761000 is not a magic number
# It's just for convenient for the for loop
p=469762049
for k in $(seq 3 17);
do
	./$TEST $k > $TMP
	
	cat $TMP  >> ../$OUT
	echo "read(\"fast_rem.txt\");" > $MWW
	cat $TMP | grep "Input" >> $MWW
	echo "Result := TRDsubproduct_tree(Input, x, $p);" >> $MWW
	cat $TMP | grep "Subtree" >> $MWW
	echo "k := $(($k-1));" >> $MWW
	echo "err := 0;" >> $MWW
	echo "for ii from 2 to 2^k-1 do" >> $MWW
	echo "if Result[ii] <> Subtree[ii] then err:=err+1;" >> $MWW
	echo "end if;" >> $MWW
	echo "od;" >> $MWW
	echo "if err = 0 then fprintf(\"$RS\",\"PASS\"); else fprintf(\"$RS\",\"FAIL\"); end if;" >> $MWW
	maple $MWW > /dev/null
	RRS=$(cat $RS)
	echo "Subtree k=$k : $RRS"

	cat $TMP | grep "F:=" > $MW
	cat $TMP | grep "cF" >> $MW
	cat $TMP | grep "mF" >> $MW
	echo "k := $(($k-1));" >> $MW
	echo "err := 0;" >> $MW
	echo "for ii from 1 to 2^k do " >> $MW
	echo "if cF[ii] <> mF[ii] then err := err + 1;" >> $MW
	echo "end if;" >> $MW
	echo "od;" >> $MW
	echo "if err = 0 then fprintf(\"$RR\", \"PASS\"); else fprintf(\"$RR\", \"FAIL\");end if;" >> $MW
	maple $MW > /dev/null
	RESULT=$(cat $RR)
	rm -f $RR
	echo "Subtree: $RRS; k = $k :" $RESULT >> ../$LOG
	echo "Subtree: $RRS; k = $k :" $RESULT 
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
rm -f bin/$RR bin/$MW bin/$TMP



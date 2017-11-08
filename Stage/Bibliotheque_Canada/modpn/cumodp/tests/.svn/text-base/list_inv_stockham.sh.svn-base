# Test of list_inv_stockham
# Author: Jiajian Yang <jyang425@csd.uwo.ca>

TARGET="list_inv_stockham"
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
printf "#include \"../include/list_inv_stockham.h\"\n" > $TEMP_TEST_SRC

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
printf "return 0;\n }\n" >> $TEMP_TEST_SRC

# compile the test
$CC -DLINUXINTEL64 ./$TEMP_TEST_SRC -o bin/${TEST}  -L./bin/ -lcumodp

# after compilation, the test is located in bin/
cd bin/

date >> ../$LOG
date >> ../$OUT

TMP=result.tmp

RESULT=PASS
FAIL=0

# the p is for the prime number, m stands for how many ffts we are going to do at the same time
# k is the length of each FFT, BTW, the length of each FFT will be 2^k
# well I want 257 and 469762049 only. The 469761000 is not a magic number
# It's just for convenient for the for loop
p=469762049

for k in 8 10 11 12 14 ;
do
	for m in 512 1024;
	do
		./$TEST $p $m $k > $TMP
		cat $TMP >> ../$OUT
		rr=$(cat $TMP | grep FAIL | wc -l)
		if [ $rr -gt 0 ];
		then
			RESULT=FAIL
			FAIL=$(($FAIL+1))
		else
			RESULT=PASS
		fi
		echo "p="$p",m="$m",k="$k :" " $RESULT >> ../$LOG
		echo "p="$p",m="$m",k="$k :" " $RESULT 
	done
done

for k in $(seq 8 2 12);
do
	for m in $(seq 2 4 20);
	do
		./$TEST $p $m $k > $TMP
		cat $TMP >> ../$OUT
		rr=$(cat $TMP | grep FAIL | wc -l)
		if [ $rr -gt 0 ];
		then
			RESULT=FAIL
			FAIL=$(($FAIL+1))
		else
			RESULT=PASS
		fi
		echo "p="$p",m="$m",k="$k :" " $RESULT >> ../$LOG
		echo "p="$p",m="$m",k="$k :" " $RESULT 
	done
done

for k in $(seq 13 20);
do
	for m in $(seq 2 4);
	do
		./$TEST $p $m $k > $TMP
		cat $TMP >> ../$OUT
		rr=$(cat $TMP | grep FAIL | wc -l)
		if [ $rr -gt 0 ];
		then
			RESULT=FAIL
			FAIL=$(($FAIL+1))
		else
			RESULT=PASS
		fi
		echo "p="$p",m="$m",k="$k :" " $RESULT >> ../$LOG
		echo "p="$p",m="$m",k="$k :" " $RESULT 
	done
done

for k in $(seq 20 23);
do
	for m in $(seq 2 3);
	do
		./$TEST $p $m $k > $TMP
		cat $TMP >> ../$OUT
		rr=$(cat $TMP | grep FAIL | wc -l)
		if [ $rr -gt 0 ];
		then
			RESULT=FAIL
			FAIL=$(($FAIL+1))
		else
			RESULT=PASS
		fi
		echo "p="$p",m="$m",k="$k :" " $RESULT >> ../$LOG
		echo "p="$p",m="$m",k="$k :" " $RESULT 
	done
done
if [ $FAIL -gt 0 ];
then
	RESULT=FAIL
else
	RESULT=PASS
fi

echo "" >> ../$LOG

echo "The result of this test is $RESULT."
echo "Please check the test log:"$LOG
echo "Since the console output might be horribly long, the log of console is stored in"$OUT 

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
rm -f bin/$TMP

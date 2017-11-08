# Here is the description of your test blablabla :)
# Author: Jiajian Yang <jyang425@csd.uwo.ca>

TARGET="naive_poly_mul"
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
printf "#include \"../include/naive_poly_mul.h\"\n" > $TEMP_TEST_SRC

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
printf "sfixn p = atoi (argv[2]);\n" >> $TEMP_TEST_SRC
printf "mul_host_tst(m,p);\n" >> $TEMP_TEST_SRC
printf "return 0;\n }\n" >> $TEMP_TEST_SRC

# compile the test
$CC -DLINUXINTEL64 ./$TEMP_TEST_SRC -o bin/${TEST}  -L./bin/ -lcumodp

# after compilation, the test is located in bin/
cd bin/

date >> ../$LOG
date >> ../$OUT

RESULT=PASS

# n The size of the polynomials
# p the prime number
for p in $(seq 257 469761000 469762049);
do
	for n in $(seq 2 256); 
	do
		./$TEST $n $p > tmpresult.temp
		C=$(cat tmpresult.temp | grep "C :=" | tr -d " " )
		D=$(cat tmpresult.temp | grep "D :=" | tr -d " ")
		R1=$(echo $C | tr -d "C")
		R2=$(echo $D | tr -d "D")
		if [ $R1 = $R2 ];
		then
			echo "PASS p = $p, n = $n" 
		else
			echo "FAIL p = $p, n = $n"
			RESULT=FAIL
		fi
		cat tmpresult.temp >> ../$OUT
		echo "p=$p, n="$n":" $? >> ../$LOG
	done
done

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
rm -f bin/tmpresult.tmp

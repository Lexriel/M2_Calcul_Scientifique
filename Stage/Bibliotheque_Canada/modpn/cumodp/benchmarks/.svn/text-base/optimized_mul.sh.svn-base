# This is to benchmark Anis's opmized multiplication 
# Author: Jiajian Yang <jyang425@csd.uwo.ca>

TARGET="opt_plain_mul"
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

# include header files
printf "#include <iostream>\n #include \"../include/opt_plain_mul.h\"\n using namespace std;" > $TEMP_TEST_SRC

# find the test function in the source file by using flag
lineA=$(cat $SRC | grep -n $BEGINFLAG -o | grep "[0-9]\+" -o)
lineB=$(cat $SRC | grep -n $ENDFLAG -o | grep "[0-9]\+" -o)

# now we know where the test function is 
codelines=$(($lineB - $lineA))

# we cat the specific test function
cat ${SRC} | grep ${BEGINFLAG} -A $(($codelines + 1)) -B 1  >> $TEMP_TEST_SRC

# this is for the main function. 
printf "int main (int argc, char *argv[]){\n" >> $TEMP_TEST_SRC
printf "sfixn n=10, m=10, p=7;\n" >> $TEMP_TEST_SRC
printf "n = atoi (argv[1]);\n" >> $TEMP_TEST_SRC
printf "m = atoi (argv[2]);\n" >> $TEMP_TEST_SRC
printf "p = atoi (argv[3]);\n" >> $TEMP_TEST_SRC
printf $TARGET"(n,m,p);\n" >> $TEMP_TEST_SRC
printf "return 0;\n }\n" >> $TEMP_TEST_SRC

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
for tx in 128 256;
do
	for b in 4 8;
	do
		sed -i "s/const int BLOCK = [0-9]\+\;/const int BLOCK = ${b}\;/g" ../../include/opt_plain_mul.h 
		sed -i "s/const int Tx = [0-9]\+\;/const int Tx = ${tx}\;/g" ../../include/opt_plain_mul.h 
		cd ../ && make all 
		pwd
		#  compile the test
		$CC -DLINUXINTEL64 bin/$TEMP_TEST_SRC -o bin/${TEST}  -L./bin/ -lcumodp -g 
		cd bin/
		for n in $(seq 2 1 14);
		do
			for m in $(seq 2 1 14);
			do
				./$TEST $((2**$n)) $((2**$m)) $p > $TMP
				ll=$(cat $TMP | grep "TIME")
				mul=$(cat $TMP | grep "Multiplication")
				add=$(cat $TMP | grep "addition")
	
				c1=$(echo $ll | sed -e "s|TIME USED:|$((2**$n)) $((2**$m)) |")
				c2=$(echo $mul| sed -e "s|Multiplications: | |");
				c3=$(echo $add| sed -e "s|additions: | |");
				echo $c1 $c2 $c3 $tx $b >> ../$LOG
				con1=$(echo $ll"ms" | sed -e "s|TIME USED:|n = $((2**$n)), m = $((2**$m)) Time = |")
				echo $con1 $mul $add "Tx="$tx "B="$b
			done
		done
	done
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

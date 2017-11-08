# Here is the description of your test blablabla :)
# Author: Jiajian Yang <jyang425@csd.uwo.ca>

TARGET="subproduct_tree"
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
printf $TEST"();\n" >> $TEMP_TEST_SRC
printf "return 0;\n }\n" >> $TEMP_TEST_SRC

# compile the test
$CC -DLINUXINTEL64 ./$TEMP_TEST_SRC -o bin/${TEST}  -L./bin/ -lcumodp -g

# after compilation, the test is located in bin/
cd bin/

date >> ../$LOG
date >> ../$OUT

# declearing tem files
TMP=cudaresult.tmp
MW=com.mw
RR=mapleresult.tmp

# copy the fast_rem.txt
cp ../fast_rem.txt ./

RESULT=PASS

# the p is for the prime number, m stands for how many ffts we are going to do at the same tim
# k is the length of each FFT, BTW, the length of each FFT will be 2^k
# well I want 257 and 469762049 only. The 469761000 is not a magic number
# It's just for convenient for the for loop
#for p in $(seq 257 469761000 469762049); 
p=469762049
#do
./$TEST > $TMP
k=$(head inputx.txt -n1)
cat $TMP  >> ../$OUT
echo "read(\"fast_rem.txt\");" > $MW
cat $TMP | grep "Input" >> $MW
echo "Result := TRDsubproduct_tree(Input, x, $p);" >> $MW
# Instead of comparing the top two big guys, we now turn to compare the leafs
cat $TMP | grep "Test" >> $MW
echo "k := $(($k-1));" >> $MW
echo "err := 0;" >> $MW
echo "for ii from 2 to 2^k-1 do" >> $MW
echo "if Result[ii] <> Test[ii] then err:=err+1;" >> $MW
echo "end if;" >> $MW
echo "od;" >> $MW
echo "if err = 0 then fprintf(\"$RR\",\"PASS\"); else fprintf(\"$RR\",\"FAIL\"); end if;" >> $MW
maple $MW > /dev/null
RESULT=$(cat $RR)
echo "p="$p",k="$k :" " $RESULT "; ">> ../$LOG
echo "p="$p",k="$k :" " $RESULT "; " 

#done

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
rm -f bin/$RR
#rm -f bin/$MW
rm -f bin/$TMP

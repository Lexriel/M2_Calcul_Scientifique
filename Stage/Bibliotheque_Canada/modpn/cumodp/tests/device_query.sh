# Check CUDA device properties
# Author: Jiajian Yang <jyang425@csd.uwo.ca>

TARGET="device_query"
TEST=$TARGET"_tst"
SRC="../src/"$TARGET".cu"
LOG="log/"$TEST".log"

TEMP_TEST_SRC="src.c"

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
printf "#include <stdio.h>\n#include <stdlib.h>\n" > $TEMP_TEST_SRC
printf "#include \"../include/cumodp.h\"\n#include \"../include/printing.h\"\n" >> $TEMP_TEST_SRC
printf "void device_query_tst() {\n" >> $TEMP_TEST_SRC
printf "printf(\"Has cuda devices   : %%d\", is_cuda_enabled());\n" >> $TEMP_TEST_SRC
printf "printf(\"Is double enabled  : %%d\", is_double_float_enabled());\n" >> $TEMP_TEST_SRC
printf "printf(\"Global memory size : %%.2f(MB)\", global_memory_in_megabytes());\n}\n" >> $TEMP_TEST_SRC

# this is for the main function. 
printf "int main (){\n" >> $TEMP_TEST_SRC
printf "$TEST();\n" >> $TEMP_TEST_SRC
printf "return 0;\n }\n" >> $TEMP_TEST_SRC

# compile the test
$CC -DLINUXINTEL64 ./$TEMP_TEST_SRC -o bin/${TEST}  -L./bin/ -lcumodp

# after compilation, the test is located in bin/
cd bin/

date >> ../$LOG

#run the test binary
CON_OUT=$(./${TEST})

#check the ones there
ONES=$(echo $CON_OUT | grep -E "[1-9]" -o | wc -l)

export RESULT="FAIL"

if [ $ONES > 1 ];then
		export RESULT="PASS"
fi

echo ${RESULT} >> ../$LOG 
echo "Test "${RESULT}
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

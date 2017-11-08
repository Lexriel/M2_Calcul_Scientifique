make
cd ../new_tests/
make
cd Taylor_shift/
rm *~
rm *.linkinfo
./shift.sh $1
shift

while [ -n $1 ]
do
	./taylor_shift_gpu Pol12.dat $1
	shift
	echo "./taylor_shift_gpu Pol12.dat $1 done"
done

gedit diff.txt screen.txt &
cd ../../src/

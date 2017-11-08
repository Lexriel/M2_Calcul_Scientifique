cd opt_div_cuda
for ((n = 1000; n <= 5000; n = n+1000))
do
     for ((m = 500; m <= n; m = m+100))
     do	
	      touch "PNM.dat"
	      echo 9001 $n $m >"PNM.dat"
	      maple -q division.mm
	      ./test
	      diff -b Rem.dat RemGPU.dat
	      diff -b Quo.dat QuoGPU.dat
	      rm *.dat
     done
done
echo "Done."

all:
	g++ aleaPol.cpp -O2 -o aleaPol #-lntl  -lgmp -lm
#	g++ taylor_shift.cpp -O3 -o taylor_shift_cpu #-lntl  -lgmp -lm
	nvcc test1CPU.cu -O3 -o taylor_shift_cpu
	nvcc -O2 test1.cu -o taylor_shift_gpu
#	nvcc taylor_shift.cu -O2 -o taylor_shift_gpu

clean:
	rm *.o *~ *.linkinfo a.out *.log Pol*.dat taylor_shift_cpu taylor_shift_gpu aleaPol
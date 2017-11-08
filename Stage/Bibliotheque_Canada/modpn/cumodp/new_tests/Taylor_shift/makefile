all:
	g++ aleaPol.cpp -O2 -o aleaPol #-lntl  -lgmp -lm
#	g++ taylor_shift.cpp -O3 -o taylor_shift_cpu #-lntl  -lgmp -lm
	nvcc testHOR.cu -O2 -o taylor_shift_cpu_hor
	nvcc testCONV.cu -O2 -o taylor_shift_cpu_dnc
	nvcc -DLINUXINTEL64 testGPU.cu -I$(CUMODP_HOME)/include -L $(CUMODP_HOME)/src -lcumodp -O2 -o taylor_shift_gpu
#	nvcc taylor_shift.cu -O2 -o taylor_shift_gpu
	rm *.linkinfo
clean:
	rm *.o *~ a.out *.linkinfo *.log Pol*.dat taylor_shift_cpu taylor_shift_gpu aleaPol
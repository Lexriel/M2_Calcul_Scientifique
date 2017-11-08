all:
	g++ aleat_pol.cpp -O3 -o aleat_pol #-lntl  -lgmp -lm
#	g++ taylor_shift.cpp -O3 -o taylor_shift_cpu #-lntl  -lgmp -lm
	nvcc taylor_shiftCPU.cu -O3 -o taylor_shift_cpu
#	nvcc -DLINUXINTEL64 taylor_shift.cu -I$(CUMODP_HOME)/include -L $(CUMODP_HOME)/src -lcumodp -O3 -o taylor_shift_gpu
	nvcc taylor_shift.cu -O3 -o taylor_shift_gpu

clean:
	rm *.o *~ *.linkinfo a.out *.log 

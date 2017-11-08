all:
	nvcc -DLINUXINTEL64 testCudaGcd.cu -I$(CUMODP_HOME)/include -L $(CUMODP_HOME)/src -lcumodp -o test
	g++ -m32 -O2 testGcdNTL.c -o testGcdNTL  -lntl  -lgmp -lm
clean:
	rm *.o *~ *.linkinfo a.out *.log 





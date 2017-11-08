all:
	nvcc -DLINUXINTEL64 testCudaDiv.cu -I$(CUMODP_HOME)/include -L $(CUMODP_HOME)/src -lcumodp -o test
	g++ -m32 -O2 testDivNTL.c -o testDivNTL  -lntl  -lgmp -lm
clean:
	rm *.o *~ *.linkinfo a.out *.log 





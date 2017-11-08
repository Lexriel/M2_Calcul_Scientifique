
CC=gcc
NVCC=nvcc
CFLAGS=-c -O2 -DLINUXINTEL64 -g
CUFLAGS=-arch=sm_13 --compiler-options '-fPIC'
LIBFLAGS=-shared -o 
LDFLAGS=-lm -L/usr/local/cuda/lib -lcudart
CUMODP=cumodp



all: 
	$(NVCC) -O2 gcdCuda.cu -o gcd 
	$(NVCC) -O2 divCuda.cu -o div
clean:	
	rm -f *.o *~ *.linkinfo *.out *.ptx *.log *.prof *.obj
	rm -f lib$(CUMODP).*
	rm -f main exe

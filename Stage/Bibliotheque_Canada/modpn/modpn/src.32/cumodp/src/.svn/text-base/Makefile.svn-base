#####################################################################
# Compiler variables and flags
#####################################################################

#For Windows
#ARCH=WINDOWS

#For Linux
ARCH=LINUXINTEL64

ifeq ($(ARCH), LINUXINTEL64)
	CC=gcc
	GXX=g++
	NVCC=nvcc
	CFLAGS=-c -O2 -DLINUXINTEL64
	CUFLAGS=-arch=sm_13 --compiler-options '-fPIC'
	LIBFLAGS=-shared -o
	LDFLAGS=-lm -L/usr/local/cuda/lib -lcudart
	CUMODP=cumodp
endif

ifeq ($(ARCH), WINDOWS)
	CC=cl
	LINK=link
	NVCC=nvcc
	CFLAGS=
	CUFLAGS=
	LIBFLAGS=
	LDFLAGS=
	CUMODP=cumodp
endif

# Flags for the C/C++ compiler
# Useful flags:
# 	-O2   --  level of optimization
# 	-m64  --  needed to get 64-bit longs
# 	-g    --  debugging
#
# Flags for the CUDA compiler
# Useful flags:
#   --ptxas-options -v     

############################
# Sources for the library
############################
SOURCES=fft_aux.cu\
		list_stockham.cu\
		stockham.cu\
		rdr_poly.cu\
		subres_p.cu\
		subres.cu\
		scube.cu\
		modpn.cu

############################
#  Building 
############################

ifeq ($(ARCH), LINUXINTEL64) 

OBJECTS=$(SOURCES:.cu=.o)

all: so 

main: so $(SOURCES) 
	$(NVCC) -DLINUXINTEL64 benchmark.cu -o exe -L. -l$(CUMODP)
	$(CC) -DLINUXINTEL64 main.c -o main -L. -l$(CUMODP)

so: $(OBJECTS)
	$(NVCC) $(LIBFLAGS) lib$(CUMODP).so $(OBJECTS) $(LDFLAGS)

%.o : %.cu
	$(NVCC) $(CFLAGS) $(CUFLAGS) $*.cu

else
	
echo "To build for Windows platforms"

endif

clean:	
	rm -f *.o *~ *.linkinfo *.out *.ptx *.log *.prof *.obj
	rm -f lib$(CUMODP).*
	rm -f main exe

####################END OF THE FILE##################################

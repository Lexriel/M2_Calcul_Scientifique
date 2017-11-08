FC = ifort
LD = ifort
LIBS = -L/opt/intel/Compiler/11.1/073/mkl/lib/32/ -I/opt/intel/Compiler/11.1/073/mkl/include -lmkl_intel -lmkl_core -lmkl_sequential -lmkl_solver_sequential -lmkl_blas95 -lmkl_lapack95
INCLUDE = -I/opt/intel/Compiler/11.1/073/mkl/lib/32/
EXE = beulerN

OBJS = broyden_module.o beulerN_module.o beulerN_program.o 

all : $(OBJS)
	$(LD) $(INCLUDE) $(OBJS) $(LIBS) -o $(EXE)


fonction3_module.o : fonction3_module.f90
	$(LD) $(INCLUDE) -c fonction3_module.f90

broyden_module.o : broyden_module.f90
	$(LD) $(INCLUDE) -c broyden_module.f90

beulerN_module.o : beulerN_module.f90
	$(LD) $(INCLUDE) -c beulerN_module.f90

beulerN_program.o : beulerN_program.f90
	$(LD) $(INCLUDE) -c beulerN_program.f90

clean:
	rm *.o
	rm *.dat

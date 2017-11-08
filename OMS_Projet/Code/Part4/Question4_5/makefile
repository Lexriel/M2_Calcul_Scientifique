LIBS = -L/opt/intel/Compiler/11.1/073/mkl/lib/32/  -lmkl_intel -lmkl_core -lmkl_sequential -lmkl_solver_sequential -lmkl_blas95 -lmkl_lapack95
INCLUDE = -I/opt/intel/Compiler/11.1/073/mkl/lib/32/

part4 : part4_module.o part4_program.o
	ifort $(INCLUDE) part4_module.o part4_program.o $(LIBS) -o part4

part4_module.o : part4_module.f90
	ifort $(INCLUDE) -c part4_module.f90

part4_program.o : part4_program.f90
	ifort $(INCLUDE) -c part4_program.f90
clean :
	rm *.o
	rm *.dat
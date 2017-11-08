LIBS = -L/opt/intel/Compiler/11.1/073/mkl/lib/32/  -lmkl_intel -lmkl_core -lmkl_sequential -lmkl_solver_sequential -lmkl_blas95 -lmkl_lapack95
INCLUDE = -I/opt/intel/Compiler/11.1/073/mkl/lib/32/

part5 : part5_module.o part5_program.o
	ifort $(INCLUDE) part5_module.o part5_program.o $(LIBS) -o part5

part5_module.o : part5_module.f90
	ifort $(INCLUDE) -c part5_module.f90

part5_program.o : part5_program.f90
	ifort $(INCLUDE) -c part5_program.f90
clean :
	rm *.o
	rm *.dat

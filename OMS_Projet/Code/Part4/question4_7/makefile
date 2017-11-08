interpolation : interpolation_module.o interpolation_program.o
	mpif90 -o interpolation interpolation_module.o interpolation_program.o

interpolation_program.o : interpolation_program.f90
	mpif90 -c interpolation_program.f90

interpolation_module.o : interpolation_module.f90 mpif.h
	mpif90 -c interpolation_module.f90

clean :
	rm *.o



all:
	feuler
	beuler
	crni

feuler: fonction_module.o feuler_module.o feuler_program.o
	gfortran -o feuler fonction_module.o feuler_module.o feuler_program.o

beuler: fonction_module.o beuler_module.o beuler_program.o
	gfortran -o beuler fonction_module.o beuler_module.o beuler_program.o

crni: fonction_module.o crni_module.o crni_program.o
	gfortran -o crni fonction_module.o crni_module.o crni_program.o

fonction_module.o : fonction_module.f90
	gfortran -c fonction_module.f90

feuler_module.o : feuler_module.f90
	gfortran -c feuler_module.f90

beuler_module.o : beuler_module.f90
	gfortran -c beuler_module.f90

crni_module.o : crni_module.f90
	gfortran -c crni_module.f90

feuler_program.o : feuler_program.f90
	gfortran -c feuler_program.f90

beuler_program.o : beuler_program.f90
	gfortran -c beuler_program.f90

crni_program.o : crni_program.f90
	gfortran -c crni_program.f90

clean:
	rm *.o
	rm feuler
	rm beuler
	rm crni

mrproper: clean
	rm *.mod
	rm *.dat
feuler : init_data_module.o feuler_module.o feuler_program.o
	ifort -o feuler init_data_module.o feuler_module.o feuler_program.o

feuler_module.o : feuler_module.f90
	ifort -c feuler_module.f90

feuler_program.o : feuler_program.f90
	ifort -c feuler_program.f90

init_data_module.o : init_data_module.f90
	ifort -c init_data_module.f90

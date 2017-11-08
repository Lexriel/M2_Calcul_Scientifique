RK2 : init_data_module.o RK2_module.o RK2_program.o
	ifort -o RK2 init_data_module.o RK2_module.o RK2_program.o

RK2_module.o : RK2_module.f90
	ifort -c RK2_module.f90

RK2_program.o : RK2_program.f90
	ifort -c RK2_program.f90

init_data_module.o : init_data_module.f90
	ifort -c init_data_module.f90

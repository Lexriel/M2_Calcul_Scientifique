RK4 : init_data_module.o RK4_module.o RK4_program.o
	ifort -o RK4 init_data_module.o RK4_module.o RK4_program.o

RK4_program.o : RK4_program.f90
	ifort -c RK4_program.f90

RK4_module.o : RK4_module.f90
	ifort -c RK4_module.f90

init_data_module.o : init_data_module.f90
	ifort -c init_data_module.f90

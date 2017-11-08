spline : spline.o spline_program.o
	mpif90 -o spline spline.o spline_program.o

spline_program.o : spline_program.f90
	mpif90 -c spline_program.f90

spline.o : spline.f90 mpif.h
	mpif90 -c spline.f90

clean :
	rm *.o



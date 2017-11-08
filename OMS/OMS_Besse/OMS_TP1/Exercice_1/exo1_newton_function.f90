MODULE exo1_newton_function

IMPLICIT NONE



CONTAINS

FUNCTION f(x)

	REAL, INTENT(IN) :: x
	REAL :: f

        f = tanh(x)*cos(x*x)+x+2

END FUNCTION

FUNCTION g(x)

	REAL, INTENT(IN) :: x
	REAL :: g

        g = (1-tanh(x)*tanh(x))*cos(x*x)-2*tanh(x)*sin(x*x)*x+1

END FUNCTION

FUNCTION newton(a,b,n)

	REAL, INTENT(INOUT) :: a,b
	REAL :: newton
	INTEGER, INTENT(IN) :: n
	INTEGER  :: i
	

	DO i=1,n
        
	a = -f(a)/g(a) + a
	
        END DO

	newton = a

END FUNCTION

END MODULE exo1_newton_function

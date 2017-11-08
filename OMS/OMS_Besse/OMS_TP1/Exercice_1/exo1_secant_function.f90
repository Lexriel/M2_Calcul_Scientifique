MODULE exo1_secant_function

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

FUNCTION secant(a,b,n)

	REAL, INTENT(INOUT) :: a,b
	REAL :: secant,c
	INTEGER, INTENT(IN) :: n
	INTEGER  :: i
	

	DO i=1,n
		
	c = a -f(a)*(a-b)/(f(a)-f(b))

		IF ( f(c)*f(a) > 0 ) THEN
		a=c
		ELSE
		b=c
		END IF
	
	
        END DO

	secant = (a+b)/2

END FUNCTION

END MODULE exo1_secant_function

MODULE newton_method

  IMPLICIT NONE 

CONTAINS

  FUNCTION newton_method(a, b)
    INTEGER :: newton_method
    INTEGER :: a
    INTEGER :: b
    INTEGER, INTENT(INOUT) :: x
    INTEGER :: f
    
    
    f=tanh(x)*cos(x*x)+x+2
    
    IF ( (tanh(a)*cos(a*a)+a+2)*(tanh(b)*cos(b*b)+b+2) > 0 ) THEN
       print*, "L'intervalle ["a";"b"] proposé n'est pas bon."
       x=-1

       ELSE IF (tanh(a)*cos(a*a)+a+2)=0) THEN
       f=




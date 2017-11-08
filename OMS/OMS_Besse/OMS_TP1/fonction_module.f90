MODULE fonction_module
  
  IMPLICIT NONE
  
CONTAINS


  REAL FUNCTION f(x)
    IMPLICIT NONE
    REAL, INTENT(IN) :: x
    f=cos(2*x)
    END FUNCTION f


  REAL FUNCTION fprime(x)
    IMPLICIT NONE
    REAL, INTENT(IN) :: x
    fprime=-2*sin(2*x)
   END FUNCTION fprime


END MODULE fonction_module

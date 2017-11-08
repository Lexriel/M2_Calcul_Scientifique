MODULE facto_M

  IMPLICIT NONE

CONTAINS
  
  SUBROUTINE facto(n, fn)

    INTEGER, INTENT(IN) :: n
    INTEGER, INTENT(INOUT) :: fn
    INTEGER :: i
    
    IF (n<0) THEN
       print*, "valeur negative"
       factorielle=-1
       
    ELSE IF (n==0) THEN
       factorielle=1
    ELSE
       factorielle=1
       DO i=1, n
          factorielle=factorielle*i
       END DO
    END IF

END SUBROUTINE facto


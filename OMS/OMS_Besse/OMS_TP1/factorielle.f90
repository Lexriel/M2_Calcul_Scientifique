MODULE factorielle_M

  IMPLICIT NONE
CONTAINS
  
  FUNCTION factorielle(n)
    
    INTEGER, INTENT(IN) :: n
    INTEGER :: factorielle
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
    
  END FUNCTION factorielle
  

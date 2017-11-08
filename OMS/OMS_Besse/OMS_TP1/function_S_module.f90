MODULE function_S_module

IMPLICIT NONE

FUNCTION S(x,y) return z
  REAL, INTENT(IN) :: x, y
  REAL :: z, r
  r = sqrt(x**2+y**2)

  IF ( (1.5 <= r) .AND. (r <= 2.0) ) THEN
    z = 20.*(r-1.5)**2*r**2
  ELSE IF ( (0. <= r) .AND. (r <= 1.5) ) THEN
    z = -0.5*r**2*(r-1)**2
  ELSE
    z = 0
  END IF

END FUNCTION S

END MODULE function_S_module

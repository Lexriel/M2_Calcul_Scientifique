MODULE fonction3_module
  
  IMPLICIT NONE
  
CONTAINS


! Crée une fonction vectorielle F de dimension 3
  FUNCTION F(x,N) result(Y)
    IMPLICIT NONE
    REAL, DIMENSION(1:3), INTENT(IN) :: x
    REAL, DIMENSION(1:3) :: Y
    INTEGER :: N
    REAL :: pi
    pi= 4.*atan(1.)

    Y(1) = 3.*x(1)-cos(x(2)*x(3))-0.5
    Y(2) = x(1)**2-81.*(x(2)+0.1)**2+sin(x(3))+1.06
    Y(3) = exp(-x(1)*x(2))+20.*x(3)+(10*pi-3.)/3.

    END FUNCTION F



! Crée la jacobienne de F (matrice de taille 3x3)
  FUNCTION JacF(x,N) result(Mat)
    IMPLICIT NONE
    REAL, DIMENSION(1:3), INTENT(IN) :: x
    REAL, DIMENSION(3,3) :: Mat
    INTEGER :: N

    Mat=reshape( (/          3.          ,       2.*x(1)       ,  -x(2)*exp(-x(1)*x(2)),                 &
          &         x(3)*sin(x(2)*x(3))  ,  -2*81.*(x(2)+0.1)  ,  -x(1)*exp(-x(1)*x(2)),                 &
          &         x(2)*sin(x(2)*x(3))  ,       cos(x(3))     ,           20.                /), (/3,3/))

    END FUNCTION JacF


END MODULE fonction3_module

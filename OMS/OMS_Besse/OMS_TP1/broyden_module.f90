MODULE broyden_module

IMPLICIT NONE

CONTAINS

FUNCTION broyden(F, y0, h)

    USE lapack95
    USE blas95
  IMPLICIT NONE

! Déclaration des arguments    
    REAL, DIMENSION(:), INTENT(IN) :: y0
    REAL, INTENT(IN) :: h

! Déclaration des variables locales
    INTEGER :: i, j, itermax, cpt, N, incx
    REAL :: eps
    LOGICAL :: test
    REAL, DIMENSION(:), ALLOCATABLE :: u, res, broyden, y0p, y0m, yp, r, s, y
    REAL, DIMENSION(:,:), ALLOCATABLE :: M, A
    INTEGER, DIMENSION(:), ALLOCATABLE :: ipiv

! Utiliser deux fonctions définies dans un autre module
  INTERFACE
    FUNCTION F(x,N)
      INTEGER, INTENT(IN) :: N
      REAL, DIMENSION(1:N), INTENT(IN) :: x
      REAL, DIMENSION(N) :: F
    END FUNCTION F

  END INTERFACE

    N = size(y0)
    cpt = 1
    test = .TRUE.
    eps = 1e-8
    itermax = 50


! Allocation de mémoire pour les tableaux dynamiques
    allocate( u(N), y(N), s(N), yp(N), y0p(N), y0m(N), broyden(N), res(N) )
    allocate( M(N,N), A(N,N) )

    u = y0
    A(:,:) = 0.


  DO j = 1,N
    y0p = y0
    y0m = y0
    y0p(j) = y0p(j)+h
    y0m(j) = y0m(j)-h
    
    A(:,j)=( F(y0p,N) - F(y0m,N) ) / (2.*h)

  END DO


! Calcul itératif des éléments t et u (on aurait également pu les rentrer dans un tableau).

! initialise la matrice A à l'aide des différences finies
! A(i,j) = [   F_i( x(1) , ... , x(j)+h , ... ,x(N) )
!            - F_i( x(1) , ... , x(j)-h , ... ,x(N) ) ] / (2h) .

DO WHILE (test)
   yp = y

   res = F(y,N)
   M = A
   call getrf(M,ipiv)
   call getrs(M,ipiv,res) ! ces 2 instructions sont équivalentes à res = A\f(y)

   y = y-res
   u = F(y,N)-F(yp,N) ! yp est le précédent y, bref yp=y(n) et y=y(n+1).
   s = y-yp

   CALL gemv(A, s, u, -1., 1.) ! u=-1*A*s+1*u

   cpt = cpt+1
   M = mat_prod(u,s,N)
   M = M / (nrm2(s)**2)
   A = A + M
   
      test = ( nrm2(F(u,N)) > eps ).and.( cpt < itermax )
    
END DO

    broyden = y

! Désallocation de mémoire pour les tableaux sauf celui de sortie (ici NewtonN)    
    deallocate(u, M, A, res, ipiv, y0p, y0m, yp, y, r, s) 

! Définition de la fonction mat_prod :
    CONTAINS
      FUNCTION mat_prod(v1,v2,N) result(mat)
        IMPLICIT NONE
        INTEGER :: N
        REAL, DIMENSION(N), INTENT(IN) :: v1,v2
        REAL, DIMENSION(N,N) :: mat
        integer :: j,k
        DO k=1,N
           DO j=1,N
              mat(j,k)=v1(j)*v2(k)
           END DO
        END DO
      END FUNCTION mat_prod

  END FUNCTION broyden


END MODULE broyden_module

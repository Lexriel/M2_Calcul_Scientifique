MODULE newtonN_module

    USE linpack95
    USE lapack95
    USE blas95

IMPLICIT NONE

CONTAINS

REAL FUNCTION newtonN(F, JacF, y0)


  IMPLICIT NONE

! Déclaration des arguments    
    REAL, DIMENSION(:), INTENT(IN) :: y0 

! Déclaration des variables locales
    INTEGER :: i, itermax, cpt, N
    REAL :: eps
    LOGICAL :: test
    REAL, DIMENSION(:), ALLOCATABLE :: u, res, NewtonN
    REAL, DIMENSION(:,:), ALLOCATABLE :: M
    INTEGER, DIMENSION(:), ALLOCATABLE :: ipiv

! Utiliser deux fonctions définies dans un autre module
  INTERFACE
    FUNCTION F(x,N)
      INTEGER, INTENT(IN) :: N
      REAL, DIMENSION(1:N), INTENT(IN) :: x
      REAL, DIMENSION(N) :: F
    END FUNCTION F

    FUNCTION JacF(x,N)
      INTEGER, INTENT(IN) :: N
      REAL, DIMENSION(1:N), INTENT(IN) :: x
      REAL, DIMENSION(N,N) :: JacF
    END FUNCTION JacF
  END INTERFACE

    N=size(y0)
    cpt=1
    test=.TRUE.
    eps=1e-8
    itermax=50


! Allocation de mémoire pour les tableaux dynamiques
    allocate(u(N))
    allocate(NewtonN(N))
    allocate(M(N,N))
    allocate(res(N))
    allocate(ipiv(N))

    u=y0


! Calcul itératif des éléments t et u.

    DO WHILE (test)

       res=F(u,N)
       M=JacF(u,N)
       call getrf(M,ipiv)
       call getrs(M,ipiv,res)
       ! u=u-JacF(u,N)\F(u,N);
      cpt=cpt+1
      test=( nrm2(F(u,N)) > eps ).and.( cpt < itermax )

    END DO

    NewtonN = u

! Désallocation de mémoire pour les tableaux sauf celui de sortie (ici NewtonN)    
    deallocate(u, M, res, ipiv) 

  END FUNCTION newtonN


END MODULE newtonN_module

MODULE exo1_module

  use lapack95
  use blas95

  IMPLICIT NONE

  CONTAINS

  SUBROUTINE exo1(N)
  
  INTEGER, INTENT(IN) :: N
  INTEGER :: i, j, l
  INTEGER, DIMENSION(N*N) :: ipiv
  REAL, DIMENSION(N) :: x, y
  REAL, DIMENSION(N*N,2) :: Cxy
  REAL, DIMENSION(2) :: tspan
  REAL :: r, h
  REAL, DIMENSION(N*N) :: s, q, u, res
  REAL, DIMENSION(N*N,N*N) :: A, Atemp
  
  
  open(unit=100, file="heat_equa1.dat", status='unknown')

! variables utilisées
  tspan = [-3.,3.]
  h = (tspan(2)-tspan(1))/N
  x(1) = tspan(1)
  x(2:N) = x(1:N-1) + h
  y = x
  q(:) = 0

! Définition de l'ensemble de tous les couples de points (x;y) que l'on
! puisse former à l'aide de x et y :
  do i = 1,N
     do j=1,N
        l = j+(i-1)*N
        Cxy(l,1) = x(i)
        Cxy(l,2) = y(j)
     end do
  end do

! Définition de S
  do i = 1,N*N
     r = sqrt(Cxy(i,1)**2+Cxy(i,2)**2)
     s(i) = 20*(r-1.5)**2*(r-2)**2*(1.5<r)*(r<2) - 0.5*r**2*(r-1)**2*(0<r)*(r<1)
  end do

! Définition des éléments diagonaux de A(i,j) (sans les 0)
  do i = 1,N*N
     do j = 1,N*N
        if (i == j) then
           A(i,j) = 4/h**2
        else if ( (i == j+1) .or. (i == j-1) .or. (i == j+N) .or. (i == j-N) ) then
           A(i,j) = -1/h**2
        end if
     end do
  end do

! a:b:c signifie de a à c avec un pas de b
! Définition des 0 'diagonaux' de A(i,j)
  A(N:N:N*N, N+1:N:(N-1)*N) = 0
  A(N+1:N:(N-1)*N, N:N:N*N) = 0

! A est désormais bien définie

! Définition de q
  q(1:N) = q(1:N) + 20/h**2
  q(N:N:N*N) = q(N:N:N*N) + 20/h**2
  q(N+1:N:(N-1)*N) = q(N+1:N:(N-1)*N) + 20/h**2
  q(N**2-N+1:N**2) = q(N**2-N+1:N**2) + 20/h**2

  s = q+s

! on a A*u = s
  Atemp = A
  res = s
! u = A⁻¹*s
  call getrf(Atemp, ipiv)
  call getrs(Atemp, ipiv, res)
  u = res
  
  do i=1,N*N
    write(100, *) Cxy(i,1), Cxy(i,2), u(i)
  end do

  close(100)
 
! On peut simplifier :
! Atemp, res, q peuvent etre substitués par A, u et s qui ne servent plus

  END SUBROUTINE exo1

END MODULE exo1_module

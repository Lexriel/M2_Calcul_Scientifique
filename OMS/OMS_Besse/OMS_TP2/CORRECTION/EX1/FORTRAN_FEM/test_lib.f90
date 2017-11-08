program test_lib
  !-----------------------------------------------------------------------
  ! test program for iters -- the basic iterative solvers
  !
  !     this program generates a sparse matrix using
  !     GEN57PT and then solves a linear system with an 
  !     artificial rhs (the solution is a vector of (1,1,...,1)^T).
  !-----------------------------------------------------------------------
  !      implicit none
  !      implicit real*8 (a-h,o-z)
  use F90_Kind
  USE SysLinCSR
  USE Precond
  USE Blass

  real (kind=8), dimension(:,:), allocatable :: values
  real (kind=8), dimension(:), allocatable :: e,x
  integer, dimension(:), allocatable :: distance
  integer :: N,N2, ndiag, j,k
  real (kind=8) :: xl,xr,dx,dvar,tol,mnrm
  integer, dimension(6) :: job
  real (kind=8), dimension(:), allocatable :: a,rhs,sol
  integer, dimension(:), allocatable :: ja,ia

  TYPE(MatriceCSR) :: Mat2D

  xr=1.d0;xl=0.d0;
  N2=4; N=16
  dx=(xr-xl)/(N-1.d0)
  dx=1.d0
  ndiag=5
  allocate(values(N,ndiag), distance(ndiag), e(N), x(N))
  allocate(rhs(N),sol(N))

  do j=1,N2
     x(j)=xl+dx*(j-1.d0)
  end do

  e(:)=1.d0
  values(:,1)=-e/dx**2.d0
  values(:,2)=-e/dx**2.d0
  values(:,3)=4.d0*e/dx**2.d0
  values(:,4)=-e/dx**2.d0
  values(:,5)=-e/dx**2.d0
  values(N2+1,2)=0.d0; 
  values(2*N2+1,2)=0.d0; 
  values(3*N2+1,2)=0.d0; 
  values(N2,4)=0.d0
  values(2*N2,4)=0.d0
  values(3*N2,4)=0.d0
  distance=(/-N2,-1,0,1,N2/)

  allocate(a(N*ndiag),ja(N*ndiag),ia(N+1))
  call diacsr (n,0,ndiag,values,n,distance,a,ja,ia)

  !
  !     generate a linear system with known solution
  !
  do i = 1, n
     sol(i) = 1.0D0
  end do

  rhs(:)=0.
  call amux(n, sol, rhs, a, ja, ia) ! rhs= A * Sol

  ! Init Mat
  ! Number of Non Zero elements
  Nnz = ia(n+1)-1

  CALL CREATE(Mat2D,n,n*ndiag,nnz,TAG='LAPLACE')
  ALLOCATE (Mat2D%Guess(Mat2d%NbLine)) ! Necessaire pour methode iterative
 
  ! Copie des donnees dans la matrice Mat2D
  Mat2D%Resolution = 1
  Mat2D%Coeff(:) = a(:)
  Mat2D%IA(:) = ia(:)
  Mat2D%JA(:) = ja(:)
  Mat2D%Guess(:) = 0.

  deallocate(a,ia,ja)
  !
  !     set-up the preconditioner ILUT(15, 1E-4)  ! new definition of lfil
  !
  Mat2D%Precond=3        ! Type de preconditioneur ILUT
  Mat2D%LFill=3          ! Niveau de remplissage
  Mat2D%DropTol=1.0D-4   ! Tolerance
  CALL CalPrecond(Mat2D)

  CALL RESOL(Mat2D,RHS(1:N),SOL(1:N))

  Write(6,*) '---- SOLUTION ----'
  WRITE(6,*) SOL(1:N)

  CALL DESTROY(Mat2D)
end program test_lib

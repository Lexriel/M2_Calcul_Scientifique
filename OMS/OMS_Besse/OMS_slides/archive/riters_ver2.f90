program riters
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
  
  integer nmax, nzmax, maxits,lwk
  parameter (nmax=5000,nzmax=100000,maxits=60,lwk=nmax*40)
  integer ia(nmax),ja(nzmax),jau(nzmax),ju(nzmax),iw(nmax*3)
  integer ipar(16),nx,ny,nz,i,lfil,nwk,nrow,ierr,nnz
  real*8  a(nzmax),sol(nmax),rhs(nmax),au(nzmax),wk(nmax*40)
  real*8  xran(nmax), fpar(16), al(nmax)
  real*8  gammax,gammay,alpha,tol
  TYPE(MatriceCSR) :: Mat2D
  !external gen57pt!,cg,bcg,dbcg,bcgstab,tfqmr,gmres,fgmres,dqgmres
  !external cgnr, fom, runrc, ilut
  !     
  common /func/ gammax, gammay, alpha
  !-----------------------------------------------------------------------  
  ! pde to be discretized is :
  !---------------------------
  !
  ! -Lap u + gammax exp (xy)delx u + gammay exp (-xy) dely u +alpha u
  !
  ! where Lap = 2-D laplacean, delx = part. der. wrt x,
  ! dely = part. der. wrt y.
  ! gammax, gammay, and alpha are passed via the commun func.
  ! 
  !-----------------------------------------------------------------------  
  !
  ! data for PDE:
  !
  nx = 6
  ny = 6
  nz = 1
  alpha = 0.0
  gammax = 0.0
  gammay = 0.0
  !
  !     set the parameters for the iterative solvers
  !
  ipar(2) = 2
  ipar(3) = 1
  ipar(4) = lwk
  ipar(5) = 10 
  ipar(6) = maxits
  fpar(1) = 1.0D-5
  fpar(2) = 1.0D-10
  !--------------------------------------------------------------
  ! call GEN57PT to generate matrix in compressed sparse row format
  !
  !     al(1:6) are used to store part of the boundary conditions
  !     (see documentation on GEN57PT.)
  !--------------------------------------------------------------
  al(1) = 0.0
  al(2) = 0.0
  al(3) = 0.0
  al(4) = 0.0
  al(5) = 0.0
  al(6) = 0.0
  nrow = nx * ny * nz
  call gen57pt(nx,ny,nz,al,0,nrow,a,ja,ia,ju,rhs)
  print *, 'RITERS: generated a finite difference matrix'
  print *, '        grid size = ', nx, ' X ', ny, ' X ', nz
  print *, '        matrix size = ', nrow

  !
  !     generate a linear system with known solution
  !
  do i = 1, nrow
     sol(i) = 1.0D0
     xran(i) = 0.d0
  end do

  rhs(:)=0.
  call amux(nrow, sol, rhs, a, ja, ia) ! rhs= A * Sol
  write(6,*) rhs(1:nrow)

  ! Init Mat
  ! Number of Non Zero elements
  Nnz = ia(nrow+1)-1

  CALL CREATE(Mat2D,nrow,nzmax,nnz,TAG='LAPLACE')
  ALLOCATE (Mat2D%Guess(Mat2d%NbLine)) ! Necessaire pour methode iterative
 
  ! Copie des donnees dans la matrice Mat2D
  Mat2D%Resolution = 1
  Mat2D%Coeff(:) = a(:)
  Mat2D%IA(:) = ia(:)
  Mat2D%JA(:) = ja(:)
  Mat2D%Guess(:) = 0.

  !
  !     set-up the preconditioner ILUT(15, 1E-4)  ! new definition of lfil
  !
  Mat2D%Precond=1        ! Type de preconditioneur ILUT
  Mat2D%LFill=3          ! Niveau de remplissage
  Mat2D%DropTol=1.0D-4   ! Tolerance
  CALL CalPrecond(Mat2D)

  CALL RESOL(Mat2D,RHS(1:NROW),SOL(1:NROW))

!!$  Write(6,*) '---- SOLUTION ----'
!!$  WRITE(6,*) SOL(1:NROW)

  CALL DESTROY(Mat2D)
end program riters

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
  !     set-up the preconditioner ILUT(15, 1E-4)  ! new definition of lfil
  !
  lfil = 3
  tol = 1.0D-4
  nwk = nzmax
  call ilut (nrow,a,ja,ia,lfil,tol,au,jau,ju,nwk,wk,iw,ierr)
  ipar(2) = 2
  !
  !     generate a linear system with known solution
  !
  do i = 1, nrow
     sol(i) = 1.0D0
     xran(i) = 0.d0
  end do

  rhs(:)=0.
  call amux(nrow, sol, rhs, a, ja, ia)

  ! Init Mat
  ! Number of Non Zero elements
  Nnz = ia(nrow+1)-1

  CALL CREATE(Mat2D,nrow,nzmax,nnz,TAG='LAPLACE')
!!$  Mat2D%NbLine = nrow
!!$  Mat2D%Dim = nzmax
!!$  ALLOCATE (Mat2D%Coeff(Mat2d%Dim), Mat2D%JA(Mat2d%Dim))
!ALLOCATE (Mat2D%PCoeff(Mat2d%Dim), Mat2D%PJA(Mat2d%Dim), Mat2D%PIA(Mat2d%Dim))
!!$  ALLOCATE (Mat2D%IA(Mat2d%NbLine+1))
ALLOCATE (Mat2D%Guess(Mat2d%NbLine))
 
  Mat2D%Resolution = 1
  Mat2D%Coeff(:) = a(:)
  Mat2D%IA(:) = ia(:)
  Mat2D%JA(:) = ja(:)
!!$  Mat2D%PCoeff(:) = au(:)
!!$  Mat2D%PIA(:) = ju(:)
!!$  Mat2D%PJA(:) = jau(:)
  Mat2D%Guess(:) = 0.

  Mat2D%Precond=1
  Mat2D%LFill=3
  Mat2D%DropTol=1.0D-4
  CALL CalPrecond(Mat2D)

  CALL RESOL(Mat2D,rhs(1:nrow),Sol(1:nrow))

  CALL Destroy(Mat2D)
!!$  DEALLOCATE(Mat2D%Coeff, Mat2D%JA, Mat2D%IA)
!!$  DEALLOCATE(Mat2D%PCoeff, Mat2D%PJA, Mat2D%PIA)
!!$  DEALLOCATE(Mat2D%Guess)

!!$  print *, ' '
!!$  print *, '	*** CG ***'
!!$  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,cg)
!!$  print *, ' '
!!$  print *, '	*** BCG ***'
!!$  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,bcg)
!!$  print *, ' '
!!$  print *, '	*** DBCG ***'
!!$  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,dbcg)
!!$  print *, ' '
!!$  print *, '	*** CGNR ***'
!!$  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,cgnr)
!!$  print *, ' '
!!$  print *, '	*** BCGSTAB ***'
!!$  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,bcgstab)
!!$  print *, ' '
!!$  print *, '	*** TFQMR ***'
!!$  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,tfqmr)
!!$  print *, ' '
!!$  print *, '	*** FOM ***'
!!$  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,fom)
!!$  print *, ' '
!!$  print *, '	*** GMRES ***'
!!$  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,gmres)
!!$  print *, ' '
!!$  print *, '	*** FGMRES ***'
!!$  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,fgmres)
!!$  print *, ' '
!!$  print *, '	*** DQGMRES ***'
!!$  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,dqgmres)
!!$  stop
end program riters
!-----end-of-main
!-----------------------------------------------------------------------

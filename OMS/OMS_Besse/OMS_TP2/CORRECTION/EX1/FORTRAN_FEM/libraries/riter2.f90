program riters
  !-----------------------------------------------------------------------
  ! test program for iters -- the basic iterative solvers
  !
  !     this program reads a Harwell/Boeing matrix from standard input
  !     and solves the linear system with an artifical right-hand side
  !     (the solution is a vector of (1,1,...,1)^T)
  !-----------------------------------------------------------------------
  !      implicit none
  !      implicit real*8 (a-h,o-z)
  use F90_Kind
  use Iters
  
  integer nmax, nzmax, maxits,lwk
  parameter (nmax=5000,nzmax=100000,maxits=60,lwk=nmax*40)
  integer ia(nmax),ja(nzmax),jau(nzmax),ju(nzmax),iw(nmax*3)
  integer ipar(16),i,lfil,nwk,nrow,ierr
  real*8  a(nzmax),sol(nmax),rhs(nmax),au(nzmax),wk(nmax*40)
  real*8  xran(nmax), fpar(16), tol
  character guesol*2, title*72, key*8, type*3
  external cg,bcg,dbcg,bcgstab,tfqmr,gmres,fgmres,dqgmres
  external cgnr, fom, runrc, ilut
  !
  !     set the parameters for the iterative solvers
  !
  ipar(2) = 2
  ipar(3) = 1
  ipar(4) = lwk
  ipar(5) = 16
  ipar(6) = maxits
  fpar(1) = 1.0D-5
  fpar(2) = 1.0D-10
  !--------------------------------------------------------------
  !     read in a matrix from standard input
  !--------------------------------------------------------------
  iounit = 5
  job = 2
  nrhs = 0
  call readmt (nmax,nzmax,job,iounit,a,ja,ia,a,nrhs,guesol,nrow,ncol,nnz,title,key,type,ierr)
  print *, 'READ the matrix ', key, type
  print *, title
  print *
  !
  !     set-up the preconditioner ILUT(15, 1E-4) ! new definition of lfil
  !
  lfil = 15
  tol = 1.0D-4 ! this is too high for ilut for saylr1
  tol = 1.0D-7
  nwk = nzmax
  call ilut (nrow,a,ja,ia,lfil,tol,au,jau,ju,nwk,wk,iw,ierr)
  ipar(2) = 2
  !
  !     generate a linear system with known solution
  !
  do i = 1, nrow
     sol(i) = 1.0D0
     xran(i) = 0.D0
  end do
  call amux(nrow, sol, rhs, a, ja, ia)
  print *, ' '
  print *, '	*** CG ***'
  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,cg)
  print *, ' '
  print *, '	*** BCG ***'
  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,bcg)
  print *, ' '
  print *, '	*** DBCG ***'
  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,dbcg)
  print *, ' '
  print *, '	*** CGNR ***'
  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,cgnr)
  print *, ' '
  print *, '	*** BCGSTAB ***'
  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,bcgstab)
  print *, ' '
  print *, '	*** TFQMR ***'
  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,tfqmr)
  print *, ' '
  print *, '	*** FOM ***'
  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,fom)
  print *, ' '
  print *, '	*** GMRES ***'
  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,gmres)
  print *, ' '
  print *, '	*** FGMRES ***'
  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,fgmres)
  print *, ' '
  print *, '	*** DQGMRES ***'
  call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,dqgmres)
  stop
end program riters
!-----end-of-main
!-----------------------------------------------------------------------

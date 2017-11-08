Module ResolRun
  Use F90_Kind
  Implicit None
Contains
!

  Subroutine RunSolver(n,Rhs,Sol,Ipar,Fpar,A,JA,IA,AU,JAU,JU)
!            ^^^^^^^^^
    Use Iters
    Use Precond, Only : LuSol, LutSol
    Use Blass, Only : Amux, Atmux
    Implicit none
! Arguments
    Real(Double), Dimension(:) :: A, AU, FPar
    Integer, Dimension(:) :: IA, JA, JAU, JU, Ipar
    Intent(In) :: A, AU, IA, JA, JAU, JU
    Intent(InOut) :: Ipar, FPar
    Integer , Intent(In) :: n
    Real(Double), Dimension(:) :: Rhs, Sol
    Intent(In) :: Rhs
    Intent(Out) :: Sol
! Variables Locales
    Real(Double), Dimension(n*11) :: Wk
    !Real(Double), Dimension((n+3)*(10+2) + (10+1)*10/2) :: Wk
    Integer :: toto
    Integer :: i, iou, its
    Real(Double) :: res
    Character(Len=50) :: LeFormat
!
!     ipar(2) can be 0, 1, 2, please don't use 3
!
    if (ipar(2).gt.2) then
       print *, 'I can not do both left and right preconditioning.'
       return
    endif
!
!     normal execution
!
    its = 0 ; res = 0.0D0
    iou = 6
    !
!!$    toto=1
!!$10  Select Case (toto)
!!$    Case (1)
!!$       Call CG(n,Rhs,Sol,Ipar,Fpar,Wk)
!!$    Case (2)
!!$       Call GMRES(n,Rhs,Sol,Ipar,Fpar,Wk)
!!$    End Select
!10  Call GMRES(n,Rhs,Sol,Ipar,Fpar,Wk)  !! ATTENTION : CNAGER TAILLE WK + INITSOLVER IPAR(4)
!10  Call FOM(n,Rhs,Sol,Ipar,Fpar,Wk)  !! ATTENTION : CNAGER TAILLE WK + INITSOLVER IPAR(4)
!10  Call TFQMR(n,Rhs,Sol,Ipar,Fpar,Wk)
10  Call CG(n,Rhs,Sol,Ipar,Fpar,Wk)
!10  Call CGNR(n,Rhs,Sol,Ipar,Fpar,Wk)
!10  Call BCG(n,Rhs,Sol,Ipar,Fpar,Wk)
!10  Call DBCG(n,Rhs,Sol,Ipar,Fpar,Wk)
!10  Call BCGSTAB(n,Rhs,Sol,Ipar,Fpar,Wk)

!
!     output the residuals
!
    If (ipar(7).ne.its) Then
         !write (*, *) its, (real(res))
       its = ipar(7)
    End If
    res = fpar(5)
!
    if (ipar(1).eq.1) then
       call Amux(n, wk(ipar(8):), wk(ipar(9):), a, ja, ia) 
       goto 10
    else if (ipar(1).eq.2) then
       call Atmux(n, wk(ipar(8)), wk(ipar(9)), a, ja, ia) 
       goto 10
    else if (ipar(1).eq.3 .or. ipar(1).eq.5) then
       call Lusol(n,wk(ipar(8)),wk(ipar(9)), au, jau, ju)
       goto 10
    else if (ipar(1).eq.4 .or. ipar(1).eq.6) then
       call Lutsol(n,wk(ipar(8)),wk(ipar(9)), au, jau, ju)
       goto 10
    else if (ipar(1).le.0) then
       if (ipar(1).eq.0) then
!          print *, 'Iterative solver has satisfied convergence test.'
       else if (ipar(1).eq.-1) then
          print *, 'Iterative solver has iterated too many times.'
       else if (ipar(1).eq.-2) then
          print *, 'Iterative solver was not given enough work space.'
          print *, 'The work space should at least have ', ipar(4),&
               ' elements.'
       else if (ipar(1).eq.-3) then
          print *, 'Iterative sovler is facing a break-down.'
       else
          print *, 'Iterative solver terminated. code =', ipar(1)
       endif
    endif
!!$    write (iou, *) ipar(7), real(fpar(6))
!!$    write (iou, *) '# return code =', ipar(1), &
!!$         '	convergence rate =', fpar(7)
    LeFormat = '(XXX,A,I3,XX,A,pe8.2,XX,A,pe8.2)'
    LeFormat(9:9) = NbPositions(ipar(7)) 
    Write(iou, FMT=Trim(LeFormat)) &
         'Iter=',ipar(7),'Residu=',fpar(6),'CvRate=',fpar(7)
!
!     check the error
!
!!$      Call Amux(n,sol,wk,A,JA,IA) ! 
!!$      Do i = 1, n
!!$!         Wk(n+i) = Sol(i) !-1.0D0
!!$         Wk(i) = (wk(i) - Rhs(i))/(Rhs(i)+1.d-5)
!!$      Enddo
!!$      write (iou, *) '# the actual residual norm is', dnrm2(n,wk,1)/n,&
!!$           maxval(abs(wk(1:n)))
!!$      write (iou, *) '# the error norm is', dnrm2(n,wk(1+n),1)

    If (iou.ne.6) close(iou)
    Return
  End Subroutine RunSolver




!-----------------------------------------------------------------------
  function distdot(n,x,ix,y,iy)
    Use Blas1
    integer n, ix, iy
    Real(Double) :: distdot, x(:), y(:) !, ddot
!      external ddot
    distdot = ddot(n,x,ix,y,iy)
    return
  End function distdot
!-----end-of-distdot
!

!
Character Function NbPositions(Entier)
Implicit None
Integer, Intent(In) :: Entier
Integer :: i, NbPos
Integer, Parameter :: Code0 = iachar('0')
NbPos = 1 ; i = 10
Do While (i < Entier)
   i = i *10
   NbPos = NbPos + 1
End Do
NbPositions = achar( NbPos + Code0 )
Return
End Function NbPositions



End Module ResolRun




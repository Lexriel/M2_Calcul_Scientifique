!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!   Module d'operations sur les matrices en 'row format'    !!!!!!!
!!!!!!!!                    S P A R S K I T                    !!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


      Module Blass

      Contains
!
      Subroutine AplB 
!      Implicit None
!     & (nrow,ncol,job,a,ja,ia,b,jb,ib,c,jc,ic,nzmax,iw,ierr)
     & (nrow,ncol,a,ja,ia,b,jb,ib,c,jc,ic,nzmax,iw,ierr)
!
      Integer :: NCol, NRow, NZmax
      Double Precision a(:), b(:), c(:) 
      integer ja(:),jb(:),jc(:),ia(nrow+1),ib(nrow+1),ic(nrow+1),
     *     iw(ncol)
      Integer :: Ierr, II,J, JCol, JPOs, K, LEn, ka , kb
!-----------------------------------------------------------------------
! performs the matrix sum  C = A+B. 
!-----------------------------------------------------------------------
! on entry:
! ---------
! nrow	= integer. The row dimension of A and B
! ncol  = integer. The column dimension of A and B.
! job   = integer. Job indicator. When job = 0, only the structure
!                  (i.e. the arrays jc, ic) is computed and the
!                  real values are ignored.
!
! a,
! ja,
! ia   = Matrix A in compressed sparse row format.
! 
! b, 
! jb, 
! ib	=  Matrix B in compressed sparse row format.
!
! nzmax	= integer. The  length of the arrays c and jc.
!         amub will stop if the result matrix C  has a number 
!         of elements that exceeds exceeds nzmax. See ierr.
! 
! on return:
!----------
! c, 
! jc, 
! ic	= resulting matrix C in compressed sparse row sparse format.
!	    
! ierr	= integer. serving as error message. 
!         ierr = 0 means normal return,
!         ierr .gt. 0 means that amub stopped while computing the
!         i-th row  of C with i=ierr, because the number 
!         of elements in C exceeds nzmax.
!
! work arrays:
!------------
! iw	= integer work array of length equal to the number of
!         columns in A.
!
!-----------------------------------------------------------------------
!      logical values
!      values = (job .ne. 0) 
      ierr = 0
      len = 0
      ic(1) = 1 
      do 1 j=1, ncol
         iw(j) = 0
 1    continue
!     
      do 500 ii=1, nrow
!     row i 
         do 200 ka=ia(ii), ia(ii+1)-1 
            len = len+1
            jcol    = ja(ka)
            if (len .gt. nzmax) goto 999
            jc(len) = jcol 
            c(len)  = a(ka) 
            iw(jcol)= len
 200     continue
!     
         do 300 kb=ib(ii),ib(ii+1)-1
            jcol = jb(kb)
            jpos = iw(jcol)
            if (jpos .eq. 0) then
               len = len+1
               if (len .gt. nzmax) goto 999
               jc(len) = jcol
               c(len)  = b(kb)
               iw(jcol)= len
            else
               c(jpos) = c(jpos) + b(kb)
            endif
 300     continue
         do 301 k=ic(ii), len
	    iw(jc(k)) = 0
 301     continue
         ic(ii+1) = len+1
 500  continue
      return
 999  ierr = ii
      Print*, 'len',len
      return
!
      End Subroutine AplB
!
!!! Produit matrice vecteur
!
      Subroutine AmuX (n, x, y, a,ja,ia) 
      Implicit None
      Double Precision, Intent(In) ::  x(*), a(:) 
      Double Precision, Intent(InOut) :: y(*)
      Integer, Intent(In) :: n, ja(:), ia(:)
!-----------------------------------------------------------------------
!         A times a vector
!----------------------------------------------------------------------- 
! multiplies a matrix by a vector using the dot product form
! Matrix A is stored in compressed sparse row storage.
!
! on entry:
!----------
! n     = row dimension of A
! x     = real array of length equal to the column dimension of
!         the A matrix.
! a, ja,
!    ia = input matrix in compressed sparse row format.
!
! on return:
!-----------
! y     = real array of length n, containing the product y=Ax
!
!-----------------------------------------------------------------------
! local variables
!
      Double Precision :: t
      Integer ::  i, k
!-----------------------------------------------------------------------
       do 100 i = 1,n
!
!     compute the inner product of row i with vector x
! 
         t = 0.0d0
         do 99 k=ia(i), ia(i+1)-1
            t = t + a(k)*x(ja(k))
 99      continue
!
!     store result in y(i) 
!
         y(i) = t
 100  continue         
!
      return
!
      End Subroutine AmuX
!
!!!
!
      Subroutine Atmux (n, x, y, a, ja, ia)
      Double Precision x(*), y(*), a(:) 
      Integer n, ia(:), ja(:)
!-----------------------------------------------------------------------
!         transp( A ) times a vector
!----------------------------------------------------------------------- 
! multiplies the transpose of a matrix by a vector when the original
! matrix is stored in compressed sparse row storage. Can also be
! viewed as the product of a matrix by a vector when the original
! matrix is stored in the compressed sparse column format.
!-----------------------------------------------------------------------
!
! on entry:
!----------
! n     = row dimension of A
! x     = real array of length equal to the column dimension of
!         the A matrix.
! a, ja,
!    ia = input matrix in compressed sparse row format.
!
! on return:
!-----------
! y     = real array of length n, containing the product y=transp(A)*x
!
!-----------------------------------------------------------------------
!     local variables 
!
      Integer i, k 
!-----------------------------------------------------------------------
!
!     zero out output vector
! 
      do 1 i=1,n
         y(i) = 0.0
 1    continue
!
! loop over the rows
!
      do 100 i = 1,n
         do 99 k=ia(i), ia(i+1)-1 
            y(ja(k)) = y(ja(k)) + x(i)*a(k)
 99      continue
 100  continue
!
      Return
!-------------end-of-atmux---------------------------------------------- 
!-----------------------------------------------------------------------
      End Subroutine Atmux
      End Module Blass 

      Function distdot(n,x,ix,y,iy)
      Use Blas1
      integer n, ix, iy
      Double Precision distdot, x(*), y(*) !, ddot
!      external ddot
      distdot = ddot(n,x,ix,y,iy)
      Return
      End Function distdot









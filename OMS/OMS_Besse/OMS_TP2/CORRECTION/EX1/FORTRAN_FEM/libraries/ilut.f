      Module PreCond
        Use F90_Kind
        Use Blas1
      Contains
      subroutine ilut(n,a,ja,ia,lfil,droptol,alu,jlu,ju,iwk,w,jw,ierr)
!-----------------------------------------------------------------------
!     implicit none 
      Real(Double) a(:),alu(:),w(:),droptol
      integer ja(:),ia(:),jlu(:),ju(:),jw(:),n,lfil,iwk,ierr
!----------------------------------------------------------------------*
!                      *** ILUT preconditioner ***                     *
!      incomplete LU factorization with dual truncation mechanism      *
!----------------------------------------------------------------------*
!     Author: Yousef Saad *May, 5, 1990, Latest revision, August 1996  *
!----------------------------------------------------------------------*
! PARAMETERS                                                           
!-----------                                                           
!
! on entry:
!========== 
! n       = integer. The row dimension of the matrix A. The matrix 
!
! a,ja,ia = matrix stored in Compressed Sparse Row format.              
!
! lfil    = integer. The fill-in parameter. Each row of L and each row
!           of U will have a maximum of lfil elements (excluding the 
!           diagonal element). lfil must be .ge. 0.
!           ** WARNING: THE MEANING OF LFIL HAS CHANGED WITH RESPECT TO
!           EARLIER VERSIONS. 
!
! droptol = Double Precision. Sets the threshold for dropping small terms in the
!           factorization. See below for details on dropping strategy.
!
!  
! iwk     = integer. The lengths of arrays alu and jlu. If the arrays
!           are not big enough to store the ILU factorizations, ilut
!           will stop with an error message. 
!
! On return:
!===========
!
! alu,jlu = matrix stored in Modified Sparse Row (MSR) format containing
!           the L and U factors together. The diagonal (stored in
!           alu(1:n) ) is inverted. Each i-th row of the alu,jlu matrix
!           contains the i-th row of L (excluding the diagonal entry=1)
!           followed by the i-th row of U.
!
! ju      = integer array of length n containing the pointers to
!           the beginning of each row of U in the matrix alu,jlu.
!
! ierr    = integer. Error message with the following meaning.
!           ierr  = 0    --> successful return.
!           ierr .gt. 0  --> zero pivot encountered at step number ierr.
!           ierr  = -1   --> Error. input matrix may be wrong.
!                            (The elimination process has generated a
!                            row in L or U whose length is .gt.  n.)
!           ierr  = -2   --> The matrix L overflows the array al.
!           ierr  = -3   --> The matrix U overflows the array alu.
!           ierr  = -4   --> Illegal value for lfil.
!           ierr  = -5   --> zero row encountered.
!
! work arrays:
!=============
! jw      = integer work array of length 2*n.
! w       = real work array of length n 
!  
!----------------------------------------------------------------------
! w, ju (1:n) store the working array [1:ii-1 = L-part, ii:n = u] 
! jw(n+1:2n)  stores nonzero indicators
! 
! Notes:
! ------
! The diagonal elements of the input matrix must be  nonzero (at least
! 'structurally'). 
!
!----------------------------------------------------------------------* 
!---- Dual drop strategy works as follows.                             *
!                                                                      *
!     1) Theresholding in L and U as set by droptol. Any element whose *
!        magnitude is less than some tolerance (relative to the abs    *
!        value of diagonal element in u) is dropped.                   *
!                                                                      *
!     2) Keeping only the largest lfil elements in the i-th row of L   * 
!        and the largest lfil elements in the i-th row of U (excluding *
!        diagonal elements).                                           *
!                                                                      *
! Flexibility: one  can use  droptol=0  to get  a strategy  based on   *
! keeping  the largest  elements in  each row  of L  and U.   Taking   *
! droptol .ne.  0 but lfil=n will give  the usual threshold strategy   *
! (however, fill-in is then mpredictible).                             *
!----------------------------------------------------------------------*
!     locals
      integer ju0,k,j1,j2,j,ii,i,lenl,lenu,jj,jrow,jpos,len 
      Real(Double) tnorm, t, s, fact 
      if (lfil .lt. 0) goto 998
!-----------------------------------------------------------------------
!     initialize ju0 (points to next element to be added to alu,jlu)
!     and pointer array.
!-----------------------------------------------------------------------
      ju0 = n+2
      jlu(1) = ju0
!
!     initialize nonzero indicator array. 
!
      do 1 j=1,n
         jw(n+j)  = 0
 1    continue
!-----------------------------------------------------------------------
!     beginning of main loop.
!-----------------------------------------------------------------------
      do 500 ii = 1, n
         j1 = ia(ii)
         j2 = ia(ii+1) - 1
         tnorm = 0.0d0
         do 501 k=j1,j2
            tnorm = tnorm+abs(a(k))
 501     continue
         if (tnorm .eq. 0.0) goto 999
         tnorm = tnorm/real(j2-j1+1)
!     
!     unpack L-part and U-part of row of A in arrays w 
!     
         lenu = 1
         lenl = 0
         jw(ii) = ii
         w(ii) = 0.d0
         jw(n+ii) = ii
!
         do 170  j = j1, j2
            k = ja(j)
            t = a(j)
            if (k .lt. ii) then
               lenl = lenl+1
               jw(lenl) = k
               w(lenl) = t
               jw(n+k) = lenl
            else if (k .eq. ii) then
               w(ii) = t
            else
               lenu = lenu+1
               jpos = ii+lenu-1 
               jw(jpos) = k
               w(jpos) = t
               jw(n+k) = jpos
            endif
 170     continue
         jj = 0
         len = 0 
!     
!     eliminate previous rows
!     
 150     jj = jj+1
         if (jj .gt. lenl) goto 160
!-----------------------------------------------------------------------
!     in order to do the elimination in the correct order we must select
!     the smallest column index among jw(k), k=jj+1, ..., lenl.
!-----------------------------------------------------------------------
         jrow = jw(jj)
         k = jj
!     
!     determine smallest column index
!     
         do 151 j=jj+1,lenl
            if (jw(j) .lt. jrow) then
               jrow = jw(j)
               k = j
            endif
 151     continue
!
         if (k .ne. jj) then
!     exchange in jw
            j = jw(jj)
            jw(jj) = jw(k)
            jw(k) = j
!     exchange in jr
            jw(n+jrow) = jj
            jw(n+j) = k
!     exchange in w
            s = w(jj)
            w(jj) = w(k)
            w(k) = s
         endif
!
!     zero out element in row by setting jw(n+jrow) to zero.
!     
         jw(n+jrow) = 0
!
!     get the multiplier for row to be eliminated (jrow).
!     
         fact = w(jj)*alu(jrow)
         if (abs(fact) .le. droptol) goto 150
!     
!     combine current row and row jrow
!
         do 203 k = ju(jrow), jlu(jrow+1)-1
            s = fact*alu(k)
            j = jlu(k)
            jpos = jw(n+j)
            if (j .ge. ii) then
!     
!     dealing with upper part.
!     
               if (jpos .eq. 0) then
!
!     this is a fill-in element
!     
                  lenu = lenu+1
                  if (lenu .gt. n) goto 995
                  i = ii+lenu-1
                  jw(i) = j
                  jw(n+j) = i
                  w(i) = - s
               else
!
!     this is not a fill-in element 
!
                  w(jpos) = w(jpos) - s
               endif
            else
!     
!     dealing  with lower part.
!     
               if (jpos .eq. 0) then
!
!     this is a fill-in element
!     
                  lenl = lenl+1
                  if (lenl .gt. n) Then
                     Ierr = - 1 !!! Rajout pour le compilo IFC
                     goto 995
                  End if
                  jw(lenl) = j
                  jw(n+j) = lenl
                  w(lenl) = - s
               else
!     
!     this is not a fill-in element 
!     
                  w(jpos) = w(jpos) - s
               endif
            endif
 203     continue
!     
!     store this pivot element -- (from left to right -- no danger of
!     overlap with the working elements in L (pivots). 
!     
         len = len+1 
         w(len) = fact
         jw(len)  = jrow
         goto 150
 160     continue
!     
!     reset double-pointer to zero (U-part)
!     
         do 308 k=1, lenu
            jw(n+jw(ii+k-1)) = 0
 308     continue
!     
!     update L-matrix
!     
         lenl = len 
         len = min0(lenl,lfil)
!     
!     sort by quick-split
!
         call qsplit (w,jw,lenl,len)
!
!     store L-part
! 
         do 204 k=1, len 
            if (ju0 .gt. iwk) goto 996
            alu(ju0) =  w(k)
            jlu(ju0) =  jw(k)
            ju0 = ju0+1
 204     continue
!     
!     save pointer to beginning of row ii of U
!     
         ju(ii) = ju0
!
!     update U-matrix -- first apply dropping strategy 
!
         len = 0
         do k=1, lenu-1
            if (abs(w(ii+k)) .gt. droptol*tnorm) then 
               len = len+1
               w(ii+len) = w(ii+k) 
               jw(ii+len) = jw(ii+k) 
            endif
         enddo
         lenu = len+1
         len = min0(lenu,lfil)
!
         call qsplit (w(ii+1:), jw(ii+1:), lenu-1,len)
!
!     copy
! 
         t = abs(w(ii))
         if (len + ju0 .gt. iwk) goto 997
         do 302 k=ii+1,ii+len-1 
            jlu(ju0) = jw(k)
            alu(ju0) = w(k)
            t = t + abs(w(k) )
            ju0 = ju0+1
 302     continue
!     
!     store inverse of diagonal element of u
!     
         if (w(ii) .eq. 0.0) w(ii) = (0.0001 + droptol)*tnorm
!     
         alu(ii) = 1.0d0/ w(ii) 
!     
!     update pointer to beginning of next row of U.
!     
         jlu(ii+1) = ju0
!-----------------------------------------------------------------------
!     end main loop
!-----------------------------------------------------------------------
 500  continue
      ierr = 0
      return
!
!     incomprehensible error. Matrix must be wrong.
!     
 995  ierr = -1
      return
!     
!     insufficient storage in L.
!     
 996  ierr = -2
      return
!     
!     insufficient storage in U.
!     
 997  ierr = -3
      return
!     
!     illegal lfil entered.
!     
 998  ierr = -4
      return
!     
!     zero row encountered
!     
 999  ierr = -5
      return
!----------------end-of-ilut--------------------------------------------
!-----------------------------------------------------------------------
      End Subroutine Ilut
      subroutine ilutp(n,a,ja,ia,lfil,droptol,permtol,mbloc,alu,
     *     jlu,ju,iwk,w,jw,iperm,ierr)
!-----------------------------------------------------------------------
      implicit none
      integer n,ja(:),ia(:),lfil,jlu(:),ju(:),jw(:),iwk,
     *     iperm(:),ierr
      Real(Double) a(:), alu(:), w(:), droptol
!----------------------------------------------------------------------*
!       *** ILUTP preconditioner -- ILUT with pivoting  ***            *
!      incomplete LU factorization with dual truncation mechanism      *
!----------------------------------------------------------------------*
! author Yousef Saad *Sep 8, 1993 -- Latest revision, August 1996.     *
!----------------------------------------------------------------------*
! on entry:
!==========
! n       = integer. The dimension of the matrix A.
!
! a,ja,ia = matrix stored in Compressed Sparse Row format.
!           ON RETURN THE COLUMNS OF A ARE PERMUTED. SEE BELOW FOR 
!           DETAILS. 
!
! lfil    = integer. The fill-in parameter. Each row of L and each row
!           of U will have a maximum of lfil elements (excluding the 
!           diagonal element). lfil must be .ge. 0.
!           ** WARNING: THE MEANING OF LFIL HAS CHANGED WITH RESPECT TO
!           EARLIER VERSIONS. 
!
! droptol = real*8. Sets the threshold for dropping small terms in the
!           factorization. See below for details on dropping strategy.
!
! lfil    = integer. The fill-in parameter. Each row of L and
!           each row of U will have a maximum of lfil elements.
!           WARNING: THE MEANING OF LFIL HAS CHANGED WITH RESPECT TO
!           EARLIER VERSIONS. 
!           lfil must be .ge. 0.
!
! permtol = tolerance ratio used to  determne whether or not to permute
!           two columns.  At step i columns i and j are permuted when 
!
!                     abs(a(i,j))*permtol .gt. abs(a(i,i))
!
!           [0 --> never permute; good values 0.1 to 0.01]
!
! mbloc   = if desired, permuting can be done only within the diagonal
!           blocks of size mbloc. Useful for PDE problems with several
!           degrees of freedom.. If feature not wanted take mbloc=n.
!
!  
! iwk     = integer. The lengths of arrays alu and jlu. If the arrays
!           are not big enough to store the ILU factorizations, ilut
!           will stop with an error message. 
!
! On return:
!===========
!
! alu,jlu = matrix stored in Modified Sparse Row (MSR) format containing
!           the L and U factors together. The diagonal (stored in
!           alu(1:n) ) is inverted. Each i-th row of the alu,jlu matrix
!           contains the i-th row of L (excluding the diagonal entry=1)
!           followed by the i-th row of U.
!
! ju      = integer array of length n containing the pointers to
!           the beginning of each row of U in the matrix alu,jlu.
!
! iperm   = contains the permutation arrays. 
!           iperm(1:n) = old numbers of unknowns
!           iperm(n+1:2*n) = reverse permutation = new unknowns.
!
! ierr    = integer. Error message with the following meaning.
!           ierr  = 0    --> successful return.
!           ierr .gt. 0  --> zero pivot encountered at step number ierr.
!           ierr  = -1   --> Error. input matrix may be wrong.
!                            (The elimination process has generated a
!                            row in L or U whose length is .gt.  n.)
!           ierr  = -2   --> The matrix L overflows the array al.
!           ierr  = -3   --> The matrix U overflows the array alu.
!           ierr  = -4   --> Illegal value for lfil.
!           ierr  = -5   --> zero row encountered.
!
! work arrays:
!=============
! jw      = integer work array of length 2*n.
! w       = real work array of length n 
!
! IMPORTANR NOTE:
! --------------
! TO AVOID PERMUTING THE SOLUTION VECTORS ARRAYS FOR EACH LU-SOLVE, 
! THE MATRIX A IS PERMUTED ON RETURN. [all column indices are
! changed]. SIMILARLY FOR THE U MATRIX. 
! To permute the matrix back to its original state use the loop:
!
!      do k=ia(1), ia(n+1)-1
!         ja(k) = iperm(ja(k)) 
!      enddo
! 
!-----------------------------------------------------------------------
!     local variables
!
      integer k,i,j,jrow,ju0,ii,j1,j2,jpos,len,imax,lenu,lenl,jj,mbloc,
     *     icut
      Real(Double) :: s, tmp, tnorm,xmax,xmax0, fact,  t, permtol
!     
      if (lfil .lt. 0) goto 998
!----------------------------------------------------------------------- 
!     initialize ju0 (points to next element to be added to alu,jlu)
!     and pointer array.
!-----------------------------------------------------------------------
      ju0 = n+2
      jlu(1) = ju0
!
!  integer double pointer array.
!
      do 1 j=1, n
         jw(n+j)  = 0
         iperm(j) = j
         iperm(n+j) = j
 1    continue
!-----------------------------------------------------------------------
!     beginning of main loop.
!-----------------------------------------------------------------------
      do 500 ii = 1, n
         j1 = ia(ii)
         j2 = ia(ii+1) - 1
         tnorm = 0.0d0
         do 501 k=j1,j2
            tnorm = tnorm+abs(a(k))
 501     continue
         if (tnorm .eq. 0.0) goto 999
         tnorm = tnorm/(j2-j1+1)
!
!     unpack L-part and U-part of row of A in arrays  w  --
!
         lenu = 1
         lenl = 0
         jw(ii) = ii
         w(ii) = 0.0
         jw(n+ii) = ii
!
         do 170  j = j1, j2
            k = iperm(n+ja(j))
            t = a(j)
            if (k .lt. ii) then
               lenl = lenl+1
               jw(lenl) = k
               w(lenl) = t
               jw(n+k) = lenl
            else if (k .eq. ii) then
               w(ii) = t
            else
               lenu = lenu+1
               jpos = ii+lenu-1 
               jw(jpos) = k
               w(jpos) = t
               jw(n+k) = jpos
            endif
 170     continue
         jj = 0
         len = 0 
!
!     eliminate previous rows
!
 150     jj = jj+1
         if (jj .gt. lenl) goto 160
!-----------------------------------------------------------------------
!     in order to do the elimination in the correct order we must select
!     the smallest column index among jw(k), k=jj+1, ..., lenl.
!-----------------------------------------------------------------------
         jrow = jw(jj)
         k = jj
!
!     determine smallest column index
!
         do 151 j=jj+1,lenl
            if (jw(j) .lt. jrow) then
               jrow = jw(j)
               k = j
            endif
 151     continue
!
         if (k .ne. jj) then
!     exchange in jw
            j = jw(jj)
            jw(jj) = jw(k)
            jw(k) = j
!     exchange in jr
            jw(n+jrow) = jj
            jw(n+j) = k
!     exchange in w
            s = w(jj)
            w(jj) = w(k)
            w(k) = s
         endif
!
!     zero out element in row by resetting jw(n+jrow) to zero.
!     
         jw(n+jrow) = 0
!
!     get the multiplier for row to be eliminated: jrow
!
         fact = w(jj)*alu(jrow)
!
!     drop term if small
!     
         if (abs(fact) .le. droptol) goto 150
!
!     combine current row and row jrow
!
         do 203 k = ju(jrow), jlu(jrow+1)-1
            s = fact*alu(k)
!     new column number
            j = iperm(n+jlu(k))
            jpos = jw(n+j)
            if (j .ge. ii) then
!
!     dealing with upper part.
!
               if (jpos .eq. 0) then
!
!     this is a fill-in element
!
                  lenu = lenu+1
                  i = ii+lenu-1 
                  if (lenu .gt. n) goto 995
                  jw(i) = j
                  jw(n+j) = i 
                  w(i) = - s
               else
!     no fill-in element --
                  w(jpos) = w(jpos) - s
               endif
            else
!
!     dealing with lower part.
!
               if (jpos .eq. 0) then
!
!     this is a fill-in element
!
                 lenl = lenl+1
                 if (lenl .gt. n) Then
                    Ierr = -1
                    goto 995
                 End if
                 jw(lenl) = j
                 jw(n+j) = lenl
                 w(lenl) = - s
              else
!
!     this is not a fill-in element
!
                 w(jpos) = w(jpos) - s
              endif
           endif
 203	continue
!     
!     store this pivot element -- (from left to right -- no danger of
!     overlap with the working elements in L (pivots). 
!     
        len = len+1 
        w(len) = fact
        jw(len)  = jrow
	goto 150
 160    continue
!
!     reset double-pointer to zero (U-part)
!     
        do 308 k=1, lenu
           jw(n+jw(ii+k-1)) = 0
 308	continue
!
!     update L-matrix
!
        lenl = len 
        len = min0(lenl,lfil)
!     
!     sort by quick-split
!
        call qsplit (w,jw,lenl,len)
!
!     store L-part -- in original coordinates ..
!
        do 204 k=1, len
           if (ju0 .gt. iwk) goto 996
           alu(ju0) =  w(k)  
           jlu(ju0) = iperm(jw(k))
           ju0 = ju0+1
 204    continue
!
!     save pointer to beginning of row ii of U
!
        ju(ii) = ju0
!
!     update U-matrix -- first apply dropping strategy 
!
         len = 0
         do k=1, lenu-1
            if (abs(w(ii+k)) .gt. droptol*tnorm) then 
               len = len+1
               w(ii+len) = w(ii+k) 
               jw(ii+len) = jw(ii+k) 
            endif
         enddo
         lenu = len+1
         len = min0(lenu,lfil)
         call qsplit (w(ii+1:), jw(ii+1:), lenu-1,len)
!
!     determine next pivot -- 
!
        imax = ii
        xmax = abs(w(imax))
        xmax0 = xmax
        icut = ii - 1 + mbloc - mod(ii-1,mbloc)
        do k=ii+1,ii+len-1
           t = abs(w(k))
           if (t .gt. xmax .and. t*permtol .gt. xmax0 .and.
     *          jw(k) .le. icut) then
              imax = k
              xmax = t
           endif
        enddo
!
!     exchange ws
!
        tmp = w(ii)
        w(ii) = w(imax)
        w(imax) = tmp
!
!     update iperm and reverse iperm
!
        j = jw(imax)
        i = iperm(ii)
        iperm(ii) = iperm(j)
        iperm(j) = i
!
!     reverse iperm
!
        iperm(n+iperm(ii)) = ii
        iperm(n+iperm(j)) = j
!----------------------------------------------------------------------- 
!
        if (len + ju0 .gt. iwk) goto 997
!
!     copy U-part in original coordinates
!     
        do 302 k=ii+1,ii+len-1 
           jlu(ju0) = iperm(jw(k))
           alu(ju0) = w(k)
           ju0 = ju0+1
 302	continue
!
!     store inverse of diagonal element of u
!
        if (w(ii) .eq. 0.0) w(ii) = (1.0D-4 + droptol)*tnorm
        alu(ii) = 1.0d0/ w(ii) 
!
!     update pointer to beginning of next row of U.
!
	jlu(ii+1) = ju0
!-----------------------------------------------------------------------
!     end main loop
!-----------------------------------------------------------------------
 500  continue
!
!     permute all column indices of LU ...
!
      do k = jlu(1),jlu(n+1)-1
         jlu(k) = iperm(n+jlu(k))
      enddo
!
!     ...and of A
!
      do k=ia(1), ia(n+1)-1
         ja(k) = iperm(n+ja(k))
      enddo
!
      ierr = 0
      return
!
!     incomprehensible error. Matrix must be wrong.
!
 995  ierr = -1
      return
!
!     insufficient storage in L.
!
 996  ierr = -2
      return
!
!     insufficient storage in U.
!
 997  ierr = -3
      return
!
!     illegal lfil entered.
!
 998  ierr = -4
      return
!
!     zero row encountered
!
 999  ierr = -5
      return
!----------------end-of-ilutp-------------------------------------------
!-----------------------------------------------------------------------
      end Subroutine IlutP


!!$!-----------------------------------------------------------------------
      subroutine ilud(n,a,ja,ia,alph,tol,alu,jlu,ju,iwk,w,jw,ierr)
!-----------------------------------------------------------------------
      implicit none 
      Integer n
      Real(Double) a(:),alu(:),w(:),tol, alph 
      integer ja(:),ia(:),jlu(:),ju(:),jw(:),iwk,ierr
!----------------------------------------------------------------------*
!                     *** ILUD preconditioner ***                      *
!    incomplete LU factorization with standard droppoing strategy      *
!----------------------------------------------------------------------*
! Author: Yousef Saad * Aug. 1995 --                                   * 
!----------------------------------------------------------------------*
! This routine computes the ILU factorization with standard threshold  *
! dropping: at i-th step of elimination, an element a(i,j) in row i is *
! dropped  if it satisfies the criterion:                              *
!                                                                      *
!  abs(a(i,j)) < tol * [average magnitude of elements in row i of A]   *
!                                                                      *
! There is no control on memory size required for the factors as is    *
! done in ILUT. This routines computes also various diagonal compensa- * 
! tion ILU's such MILU. These are defined through the parameter alph   *
!----------------------------------------------------------------------* 
! on entry:
!========== 
! n       = integer. The row dimension of the matrix A. The matrix 
!
! a,ja,ia = matrix stored in Compressed Sparse Row format              
!
! alph    = diagonal compensation parameter -- the term: 
!
!           alph*(sum of all dropped out elements in a given row) 
!
!           is added to the diagonal element of U of the factorization 
!           Thus: alph = 0 ---> ~ ILU with threshold,
!                 alph = 1 ---> ~ MILU with threshold. 
! 
! tol     = Threshold parameter for dropping small terms in the
!           factorization. During the elimination, a term a(i,j) is 
!           dropped whenever abs(a(i,j)) .lt. tol * [weighted norm of
!           row i]. Here weighted norm = 1-norm / number of nnz 
!           elements in the row. 
!  
! iwk     = The length of arrays alu and jlu -- this routine will stop
!           if storage for the factors L and U is not sufficient 
!
! On return:
!=========== 
!
! alu,jlu = matrix stored in Modified Sparse Row (MSR) format containing
!           the L and U factors together. The diagonal (stored in
!           alu(1:n) ) is inverted. Each i-th row of the alu,jlu matrix
!           contains the i-th row of L (excluding the diagonal entry=1)
!           followed by the i-th row of U.
!
! ju      = integer array of length n containing the pointers to
!           the beginning of each row of U in the matrix alu,jlu.
!
! ierr    = integer. Error message with the following meaning.
!           ierr  = 0    --> successful return.
!           ierr .gt. 0  --> zero pivot encountered at step number ierr.
!           ierr  = -1   --> Error. input matrix may be wrong.
!                            (The elimination process has generated a
!                            row in L or U whose length is .gt.  n.)
!           ierr  = -2   --> Insufficient storage for the LU factors --
!                            arrays alu/ jalu are  overflowed. 
!           ierr  = -3   --> Zero row encountered.
!
! Work Arrays:
!=============
! jw      = integer work array of length 2*n.
! w       = real work array of length n 
!  
!----------------------------------------------------------------------
!
! w, ju (1:n) store the working array [1:ii-1 = L-part, ii:n = u] 
! jw(n+1:2n)  stores the nonzero indicator. 
! 
! Notes:
! ------
! All diagonal elements of the input matrix must be  nonzero.
!
!----------------------------------------------------------------------- 
!     locals
      integer ju0,k,j1,j2,j,ii,i,lenl,lenu,jj,jrow,jpos,len 
      Real(Double) tnorm, t, s, fact, dropsum  
!-----------------------------------------------------------------------
!     initialize ju0 (points to next element to be added to alu,jlu)
!     and pointer array.
!-----------------------------------------------------------------------
      ju0 = n+2
      jlu(1) = ju0
!
!     initialize nonzero indicator array. 
!
      do 1 j=1,n
         jw(n+j)  = 0
 1    continue
!-----------------------------------------------------------------------
!     beginning of main loop.
!-----------------------------------------------------------------------
      do 500 ii = 1, n
         j1 = ia(ii)
         j2 = ia(ii+1) - 1
         dropsum = 0.0d0 
         tnorm = 0.0d0
         do 501 k=j1,j2
            tnorm = tnorm + abs(a(k)) 
 501     continue
         if (tnorm .eq. 0.0) goto 997
         tnorm = tnorm / real(j2-j1+1) 
!     
!     unpack L-part and U-part of row of A in arrays w 
!     
         lenu = 1
         lenl = 0
         jw(ii) = ii
         w(ii) = 0.0
         jw(n+ii) = ii
!
         do 170  j = j1, j2
            k = ja(j)
            t = a(j)
            if (k .lt. ii) then
               lenl = lenl+1
               jw(lenl) = k
               w(lenl) = t
               jw(n+k) = lenl
            else if (k .eq. ii) then
               w(ii) = t
            else
               lenu = lenu+1
               jpos = ii+lenu-1 
               jw(jpos) = k
               w(jpos) = t
               jw(n+k) = jpos
            endif
 170     continue
         jj = 0
         len = 0 
!     
!     eliminate previous rows
!     
 150     jj = jj+1
         if (jj .gt. lenl) goto 160
!-----------------------------------------------------------------------
!     in order to do the elimination in the correct order we must select
!     the smallest column index among jw(k), k=jj+1, ..., lenl.
!-----------------------------------------------------------------------
         jrow = jw(jj)
         k = jj
!     
!     determine smallest column index
!     
         do 151 j=jj+1,lenl
            if (jw(j) .lt. jrow) then
               jrow = jw(j)
               k = j
            endif
 151     continue
!
         if (k .ne. jj) then
!     exchange in jw
            j = jw(jj)
            jw(jj) = jw(k)
            jw(k) = j
!     exchange in jr
            jw(n+jrow) = jj
            jw(n+j) = k
!     exchange in w
            s = w(jj)
            w(jj) = w(k)
            w(k) = s
         endif
!
!     zero out element in row by setting resetting jw(n+jrow) to zero.
!     
         jw(n+jrow) = 0
!
!     drop term if small
!     
!         if (abs(w(jj)) .le. tol*tnorm) then
!            dropsum = dropsum + w(jj) 
!            goto 150
!         endif
!     
!     get the multiplier for row to be eliminated (jrow).
!     
         fact = w(jj)*alu(jrow)
!
!     drop term if small
!     
         if (abs(fact) .le. tol) then
            dropsum = dropsum + w(jj) 
            goto 150
         endif
!     
!     combine current row and row jrow
!
         do 203 k = ju(jrow), jlu(jrow+1)-1
            s = fact*alu(k)
            j = jlu(k)
            jpos = jw(n+j)
            if (j .ge. ii) then
!     
!     dealing with upper part.
!     
               if (jpos .eq. 0) then
!
!     this is a fill-in element
!     
                  lenu = lenu+1
                  if (lenu .gt. n) goto 995
                  i = ii+lenu-1
                  jw(i) = j
                  jw(n+j) = i
                  w(i) = - s
               else
!
!     this is not a fill-in element 
!
                  w(jpos) = w(jpos) - s
               endif
            else
!     
!     dealing with lower part.
!     
               if (jpos .eq. 0) then
!
!     this is a fill-in element
!
                  lenl = lenl+1
                  if (lenl .gt. n) Then
                     Ierr = -1 ; goto 995
                  End if
                  jw(lenl) = j
                  jw(n+j) = lenl
                  w(lenl) = - s
               else
!
!     this is not a fill-in element 
!
                  w(jpos) = w(jpos) - s
               endif
            endif
 203     continue
         len = len+1 
         w(len) = fact
         jw(len)  = jrow
         goto 150
 160     continue
!     
!     reset double-pointer to zero (For U-part only)
!     
         do 308 k=1, lenu
            jw(n+jw(ii+k-1)) = 0
 308     continue
!
!     update l-matrix
!
         do 204 k=1, len
            if (ju0 .gt. iwk) goto 996
            alu(ju0) =  w(k) 
            jlu(ju0) =  jw(k)
            ju0 = ju0+1
 204     continue
!     
!     save pointer to beginning of row ii of U
!     
         ju(ii) = ju0
!
!     go through elements in U-part of w to determine elements to keep
!
         len = 0
         do k=1, lenu-1
!            if (abs(w(ii+k)) .gt. tnorm*tol) then 
            if (abs(w(ii+k)) .gt. abs(w(ii))*tol) then 
               len = len+1
               w(ii+len) = w(ii+k) 
               jw(ii+len) = jw(ii+k)
            else
               dropsum = dropsum + w(ii+k) 
            endif
         enddo
!
!     now update u-matrix
!
         if (ju0 + len-1 .gt. iwk) goto 996
         do 302 k=ii+1,ii+len
            jlu(ju0) = jw(k)
            alu(ju0) = w(k)
            ju0 = ju0+1
 302     continue
!
!     define diagonal element 
! 
         w(ii) = w(ii) + alph*dropsum 
!
!     store inverse of diagonal element of u
!              
         if (w(ii) .eq. 0.0) w(ii) = (0.0001 + tol)*tnorm
!     
         alu(ii) = 1.0d0/ w(ii) 
!     
!     update pointer to beginning of next row of U.
!     
         jlu(ii+1) = ju0
!-----------------------------------------------------------------------
!     end main loop
!-----------------------------------------------------------------------
 500  continue
      ierr = 0
      return
!
!     incomprehensible error. Matrix must be wrong.
!     
 995  ierr = -1
      return
!     
!     insufficient storage in alu/ jlu arrays for  L / U factors 
!     
 996  ierr = -2
      return
!     
!     zero row encountered
!     
 997  ierr = -3 
      return
!----------------end-of-ilud  ------------------------------------------
!-----------------------------------------------------------------------
      End Subroutine Ilud

      subroutine iludp(n,a,ja,ia,alph,droptol,permtol,mbloc,alu,
     *     jlu,ju,iwk,w,jw,iperm,ierr)
!-----------------------------------------------------------------------
      implicit none
      integer n,ja(:),ia(:),mbloc,jlu(:),ju(:),jw(:),iwk,
     *     iperm(:),ierr
      Real(Double) ::  a(:), alu(:), w(:), alph, droptol, permtol 
!----------------------------------------------------------------------*
!                     *** ILUDP preconditioner ***                     *
!    incomplete LU factorization with standard droppoing strategy      *
!    and column pivoting                                               * 
!----------------------------------------------------------------------*
! author Yousef Saad -- Aug 1995.                                      *
!----------------------------------------------------------------------*
! on entry:
!==========
! n       = integer. The dimension of the matrix A.
!
! a,ja,ia = matrix stored in Compressed Sparse Row format.
!           ON RETURN THE COLUMNS OF A ARE PERMUTED.
!
! alph    = diagonal compensation parameter -- the term: 
!
!           alph*(sum of all dropped out elements in a given row) 
!
!           is added to the diagonal element of U of the factorization 
!           Thus: alph = 0 ---> ~ ILU with threshold,
!                 alph = 1 ---> ~ MILU with threshold. 
! 
! droptol = tolerance used for dropping elements in L and U.
!           elements are dropped if they are .lt. norm(row) x droptol
!           row = row being eliminated
!
! permtol = tolerance ratio used for determning whether to permute
!           two columns.  Two columns are permuted only when 
!           abs(a(i,j))*permtol .gt. abs(a(i,i))
!           [0 --> never permute; good values 0.1 to 0.01]
!
! mbloc   = if desired, permuting can be done only within the diagonal
!           blocks of size mbloc. Useful for PDE problems with several
!           degrees of freedom.. If feature not wanted take mbloc=n.
!
! iwk     = integer. The declared lengths of arrays alu and jlu
!           if iwk is not large enough the code will stop prematurely
!           with ierr = -2 or ierr = -3 (see below).
!
! On return:
!===========
!
! alu,jlu = matrix stored in Modified Sparse Row (MSR) format containing
!           the L and U factors together. The diagonal (stored in
!           alu(1:n) ) is inverted. Each i-th row of the alu,jlu matrix
!           contains the i-th row of L (excluding the diagonal entry=1)
!           followed by the i-th row of U.
!
! ju      = integer array of length n containing the pointers to
!           the beginning of each row of U in the matrix alu,jlu.
! iperm   = contains the permutation arrays ..
!           iperm(1:n) = old numbers of unknowns
!           iperm(n+1:2*n) = reverse permutation = new unknowns.
!
! ierr    = integer. Error message with the following meaning.
!           ierr  = 0    --> successful return.
!           ierr .gt. 0  --> zero pivot encountered at step number ierr.
!           ierr  = -1   --> Error. input matrix may be wrong.
!                            (The elimination process has generated a
!                            row in L or U whose length is .gt.  n.)
!           ierr  = -2   --> The L/U matrix overflows the arrays alu,jlu
!           ierr  = -3   --> zero row encountered.
!
! work arrays:
!=============
! jw      = integer work array of length 2*n.
! w       = real work array of length 2*n 
!
! Notes:
! ------
! IMPORTANT: TO AVOID PERMUTING THE SOLUTION VECTORS ARRAYS FOR EACH 
! LU-SOLVE, THE MATRIX A IS PERMUTED ON RETURN. [all column indices are
! changed]. SIMILARLY FOR THE U MATRIX. 
! To permute the matrix back to its original state use the loop:
!
!      do k=ia(1), ia(n+1)-1
!         ja(k) = perm(ja(k)) 
!      enddo
! 
!-----------------------------------------------------------------------
!     local variables
!
      integer k,i,j,jrow,ju0,ii,j1,j2,jpos,len,imax,lenu,lenl,jj,icut
      Real(Double) :: s,tmp,tnorm,xmax,xmax0,fact,t,dropsum 
!----------------------------------------------------------------------- 
!     initialize ju0 (points to next element to be added to alu,jlu)
!     and pointer array.
!-----------------------------------------------------------------------
      ju0 = n+2
      jlu(1) = ju0
!
!  integer double pointer array.
!
      do 1 j=1,n
         jw(n+j)  = 0
         iperm(j) = j
         iperm(n+j) = j
 1    continue
!-----------------------------------------------------------------------
!     beginning of main loop.
!-----------------------------------------------------------------------
      do 500 ii = 1, n
         j1 = ia(ii)
         j2 = ia(ii+1) - 1
         dropsum = 0.0d0 
         tnorm = 0.0d0
         do 501 k=j1,j2
            tnorm = tnorm+abs(a(k))
 501     continue
         if (tnorm .eq. 0.0) goto 997
         tnorm = tnorm/(j2-j1+1)
!
!     unpack L-part and U-part of row of A in arrays  w  --
!
         lenu = 1
         lenl = 0
         jw(ii) = ii
         w(ii) = 0.0
         jw(n+ii) = ii
!
         do 170  j = j1, j2
            k = iperm(n+ja(j))
            t = a(j)
            if (k .lt. ii) then
               lenl = lenl+1
               jw(lenl) = k
               w(lenl) = t
               jw(n+k) = lenl
            else if (k .eq. ii) then
               w(ii) = t
            else
               lenu = lenu+1
               jpos = ii+lenu-1 
               jw(jpos) = k
               w(jpos) = t
               jw(n+k) = jpos
            endif
 170     continue
         jj = 0
         len = 0 
!
!     eliminate previous rows
!
 150     jj = jj+1
         if (jj .gt. lenl) goto 160
!-----------------------------------------------------------------------
!     in order to do the elimination in the correct order we must select
!     the smallest column index among jw(k), k=jj+1, ..., lenl.
!-----------------------------------------------------------------------
         jrow = jw(jj)
         k = jj
!
!     determine smallest column index
!
         do 151 j=jj+1,lenl
            if (jw(j) .lt. jrow) then
               jrow = jw(j)
               k = j
            endif
 151     continue
!
         if (k .ne. jj) then
!     exchange in jw
            j = jw(jj)
            jw(jj) = jw(k)
            jw(k) = j
!     exchange in jr
            jw(n+jrow) = jj
            jw(n+j) = k
!     exchange in w
            s = w(jj)
            w(jj) = w(k)
            w(k) = s
         endif
!
!     zero out element in row by resetting jw(n+jrow) to zero.
!     
         jw(n+jrow) = 0
!
!     drop term if small
!     
         if (abs(w(jj)) .le. droptol*tnorm) then
            dropsum = dropsum + w(jj) 
            goto 150
         endif      
!
!     get the multiplier for row to be eliminated: jrow
!
         fact = w(jj)*alu(jrow)
!
!     combine current row and row jrow
!
         do 203 k = ju(jrow), jlu(jrow+1)-1
            s = fact*alu(k)
!     new column number
            j = iperm(n+jlu(k))
            jpos = jw(n+j)
!
!     if fill-in element is small then disregard:
!     
            if (j .ge. ii) then
!
!     dealing with upper part.
!
               if (jpos .eq. 0) then
!     this is a fill-in element
                  lenu = lenu+1
                  i = ii+lenu-1 
                  if (lenu .gt. n) goto 995
                  jw(i) = j
                  jw(n+j) = i 
                  w(i) = - s
               else
!     no fill-in element --
                  w(jpos) = w(jpos) - s
               endif
            else
!
!     dealing with lower part.
!
               if (jpos .eq. 0) then
!     this is a fill-in element
                 lenl = lenl+1
                 if (lenl .gt. n) Then 
                    Ierr = -1 ; goto 995
                 End if
                 jw(lenl) = j
                 jw(n+j) = lenl
                 w(lenl) = - s
              else
!     no fill-in element --
                 w(jpos) = w(jpos) - s
              endif
           endif
 203	continue
        len = len+1 
        w(len) = fact
        jw(len)  = jrow
	goto 150
 160    continue
!
!     reset double-pointer to zero (U-part)
!     
        do 308 k=1, lenu
           jw(n+jw(ii+k-1)) = 0
 308	continue
!
!     update L-matrix
!
        do 204 k=1, len
           if (ju0 .gt. iwk) goto 996
           alu(ju0) =  w(k)
           jlu(ju0) = iperm(jw(k))
           ju0 = ju0+1
 204    continue
!
!     save pointer to beginning of row ii of U
!
        ju(ii) = ju0
!
!     update u-matrix -- first apply dropping strategy 
!
         len = 0
         do k=1, lenu-1
            if (abs(w(ii+k)) .gt. tnorm*droptol) then 
               len = len+1
               w(ii+len) = w(ii+k) 
               jw(ii+len) = jw(ii+k) 
            else
               dropsum = dropsum + w(ii+k) 
            endif
         enddo
!
        imax = ii
        xmax = abs(w(imax))
        xmax0 = xmax
        icut = ii - 1 + mbloc - mod(ii-1,mbloc)
!
!     determine next pivot -- 
! 
        do k=ii+1,ii+len 
           t = abs(w(k))
           if (t .gt. xmax .and. t*permtol .gt. xmax0 .and.
     *          jw(k) .le. icut) then
              imax = k
              xmax = t
           endif
        enddo
!
!     exchange w
!
        tmp = w(ii)
        w(ii) = w(imax)
        w(imax) = tmp
!
!     update iperm and reverse iperm
!
        j = jw(imax)
        i = iperm(ii)
        iperm(ii) = iperm(j)
        iperm(j) = i
!     reverse iperm
        iperm(n+iperm(ii)) = ii
        iperm(n+iperm(j)) = j
!----------------------------------------------------------------------- 
        if (len + ju0-1 .gt. iwk) goto 996
!
!     copy U-part in original coordinates
!     
        do 302 k=ii+1,ii+len
           jlu(ju0) = iperm(jw(k))
           alu(ju0) = w(k)
           ju0 = ju0+1
 302	continue
!
!     define diagonal element 
! 
         w(ii) = w(ii) + alph*dropsum 
!
!     store inverse of diagonal element of u
!
        if (w(ii) .eq. 0.0) w(ii) = (1.0D-4 + droptol)*tnorm
!
        alu(ii) = 1.0d0/ w(ii) 
!
!     update pointer to beginning of next row of U.
!
	jlu(ii+1) = ju0
!-----------------------------------------------------------------------
!     end main loop
!-----------------------------------------------------------------------
 500  continue
!
!     permute all column indices of LU ...
!
      do k = jlu(1),jlu(n+1)-1
         jlu(k) = iperm(n+jlu(k))
      enddo
!
!     ...and of A
!
      do k=ia(1), ia(n+1)-1
         ja(k) = iperm(n+ja(k))
      enddo
!
      ierr = 0
      return
!
!     incomprehensible error. Matrix must be wrong.
! 
 995  ierr = -1 
      return 
!
!     insufficient storage in arrays alu, jlu to store factors
!
 996  ierr = -2
      return
!
!     zero row encountered
!
 997  ierr = -3 
      return
!----------------end-of-iludp-------------------------------------------
!-----------------------------------------------------------------------
      end subroutine iludp


!!$!----------------------------------------------------------------------- 
        subroutine qsplit(a,ind,n,ncut)
        Integer n
        Real(Double) a(n)
        integer ind(n), ncut, Mid, j
!-----------------------------------------------------------------------
!     does a quick-sort split of a real array.
!     on input a(1:n). is a real array
!     on output a(1:n) is permuted such that its elements satisfy:
!
!     abs(a(i)) .ge. abs(a(ncut)) for i .lt. ncut and
!     abs(a(i)) .le. abs(a(ncut)) for i .gt. ncut
!
!     ind(1:n) is an integer array which permuted in the same way as a(*).
!-----------------------------------------------------------------------
        Real(Double) tmp, abskey
        integer itmp, first, last
!-----
        first = 1
        last = n
        if (ncut .lt. first .or. ncut .gt. last) return
!
!     outer loop -- while mid .ne. ncut do
!
 1      mid = first
        abskey = abs(a(mid))
        do 2 j=first+1, last
           if (abs(a(j)) .gt. abskey) then
              mid = mid+1
!     interchange
              tmp = a(mid)
              itmp = ind(mid)
              a(mid) = a(j)
              ind(mid) = ind(j)
              a(j)  = tmp
              ind(j) = itmp
           endif
 2      continue
!
!     interchange
!
        tmp = a(mid)
        a(mid) = a(first)
        a(first)  = tmp
!
        itmp = ind(mid)
        ind(mid) = ind(first)
        ind(first) = itmp
!
!     test for while loop
!
        if (mid .eq. ncut) return
        if (mid .gt. ncut) then
           last = mid-1
        else
           first = mid+1
        endif
        goto 1
!----------------end-of-qsplit------------------------------------------
!-----------------------------------------------------------------------
        End Subroutine Qsplit
!
!-----------------------------------------------------------------------
      subroutine lusol(n, y, x, alu, jlu, ju)
      Integer :: n
      Real(Double) x(n), y(n), alu(:)
      integer jlu(:), ju(:)
!-----------------------------------------------------------------------
!
! This routine solves the system (LU) x = y, 
! given an LU decomposition of a matrix stored in (alu, jlu, ju) 
! modified sparse row format 
!
!-----------------------------------------------------------------------
! on entry:
! n   = dimension of system 
! y   = the right-hand-side vector
! alu, jlu, ju 
!     = the LU matrix as provided from the ILU routines. 
!
! on return
! x   = solution of LU x = y.     
!-----------------------------------------------------------------------
! 
! Note: routine is in place: call lusol (n, x, x, alu, jlu, ju) 
!       will solve the system with rhs x and overwrite the result on x . 
!
!-----------------------------------------------------------------------
! local variables
!
        integer i,k
!
! forward solve
!
        do 40 i = 1, n
           x(i) = y(i)
           do 41 k=jlu(i),ju(i)-1
              x(i) = x(i) - alu(k)* x(jlu(k))
 41        continue
 40     continue
!
!     backward solve.
!
      do 90 i = n, 1, -1
         do 91 k=ju(i),jlu(i+1)-1
              x(i) = x(i) - alu(k)*x(jlu(k))
 91       continue
           x(i) = alu(i)*x(i)
 90     continue
!
      return
!----------------end of lusol ------------------------------------------
!-----------------------------------------------------------------------
      End Subroutine lusol
!!$!-----------------------------------------------------------------------
      subroutine lutsol(n, y, x, alu, jlu, ju) 
      Integer :: n
      Real(Double) x(n), y(n), alu(*)
      integer jlu(*), ju(*)
!-----------------------------------------------------------------------
!
! This routine solves the system  Transp(LU) x = y,
! given an LU decomposition of a matrix stored in (alu, jlu, ju) 
! modified sparse row format. Transp(M) is the transpose of M. 
!----------------------------------------------------------------------- 
! on entry:
! n   = dimension of system 
! y   = the right-hand-side vector
! alu, jlu, ju 
!     = the LU matrix as provided from the ILU routines. 
!
! on return
! x   = solution of transp(LU) x = y.   
!-----------------------------------------------------------------------
!
! Note: routine is in place: call lutsol (n, x, x, alu, jlu, ju) 
!       will solve the system with rhs x and overwrite the result on x . 
! 
!-----------------------------------------------------------------------
! local variables
!
        integer i,k
!
        do 10 i = 1, n
           x(i) = y(i)
 10     continue
!
! forward solve (with U^T)
!
        do 20 i = 1, n
           x(i) = x(i) * alu(i)
           do 30 k=ju(i),jlu(i+1)-1
              x(jlu(k)) = x(jlu(k)) - alu(k)* x(i)
 30        continue
 20     continue
!     
!     backward solve (with L^T)
!     
      do 40 i = n, 1, -1 
         do 50 k=jlu(i),ju(i)-1
              x(jlu(k)) = x(jlu(k)) - alu(k)*x(i)
 50        continue
 40     continue
!
      return
!----------------end of lutsol -----------------------------------------
!-----------------------------------------------------------------------
      End Subroutine Lutsol
!----------------------------------------------------------------------- 
      End Module PreCond
          

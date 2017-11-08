c$$$      subroutine csort (n,a,ja,ia,iwork,values) 
c$$$      IMPLICIT NONE
c$$$      logical values
c$$$      integer n, ja(*), ia(*), iwork(*) 
c$$$      real*8 a(*) 
c$$$c-----------------------------------------------------------------------
c$$$c This routine sorts the elements of  a matrix (stored in Compressed
c$$$c Sparse Row Format) in increasing order of their column indices within 
c$$$c each row. It uses a form of bucket sort with a cost of O(nnz) where
c$$$c nnz = number of nonzero elements. 
c$$$c requires an integer work array of length 2*nnz.  
c$$$c-----------------------------------------------------------------------
c$$$c on entry:
c$$$c--------- 
c$$$c n     = the row dimension of the matrix
c$$$c a     = the matrix A in compressed sparse row format.
c$$$c ja    = the array of column indices of the elements in array a.
c$$$c ia    = the array of pointers to the rows. 
c$$$c iwork = integer work array of length max ( n+1, 2*nnz ) 
c$$$c         where nnz = (ia(n+1)-ia(1))  ) .
c$$$c values= logical indicating whether or not the real values a(*) must 
c$$$c         also be permuted. if (.not. values) then the array a is not
c$$$c         touched by csort and can be a dummy array. 
c$$$c 
c$$$c on return:
c$$$c----------
c$$$c the matrix stored in the structure a, ja, ia is permuted in such a
c$$$c way that the column indices are in increasing order within each row.
c$$$c iwork(1:nnz) contains the permutation used  to rearrange the elements.
c$$$c----------------------------------------------------------------------- 
c$$$c Y. Saad - Feb. 1, 1991.
c$$$c-----------------------------------------------------------------------
c$$$c local variables
c$$$      integer :: i, k, j, ifirst, nnz, next
c$$$      integer :: ko, irow
c$$$
c$$$      Print*, 'csort'
c$$$      Print*, 'n',n
c$$$      Print*, 'Values',Values
c$$$c
c$$$c count the number of elements in each column
c$$$c
c$$$      do 1 i=1,n+1
c$$$         iwork(i) = 0
c$$$ 1    continue
c$$$      do 3 i=1, n
c$$$         do 2 k=ia(i), ia(i+1)-1 
c$$$            j = ja(k)+1
c$$$            iwork(j) = iwork(j)+1
c$$$ 2       continue 
c$$$ 3    continue
c$$$c
c$$$c compute pointers from lengths. 
c$$$c
c$$$      iwork(1) = 1
c$$$      do 4 i=1,n
c$$$         iwork(i+1) = iwork(i) + iwork(i+1)
c$$$ 4    continue
c$$$c 
c$$$c get the positions of the nonzero elements in order of columns.
c$$$c
c$$$      ifirst = ia(1) 
c$$$      nnz = ia(n+1)-ifirst
c$$$      do 5 i=1,n
c$$$         do 51 k=ia(i),ia(i+1)-1 
c$$$            j = ja(k) 
c$$$            next = iwork(j) 
c$$$            iwork(nnz+next) = k
c$$$            iwork(j) = next+1
c$$$ 51      continue
c$$$ 5    continue
c$$$c
c$$$c convert to coordinate format
c$$$c 
c$$$      do 6 i=1, n
c$$$         do 61 k=ia(i), ia(i+1)-1 
c$$$            iwork(k) = i
c$$$ 61      continue
c$$$ 6    continue
c$$$c
c$$$c loop to find permutation: for each element find the correct 
c$$$c position in (sorted) arrays a, ja. Record this in iwork. 
c$$$c 
c$$$      do 7 k=1, nnz
c$$$         ko = iwork(nnz+k) 
c$$$         irow = iwork(ko)
c$$$         next = ia(irow)
c$$$c
c$$$c the current element should go in next position in row. iwork
c$$$c records this position. 
c$$$c 
c$$$         iwork(ko) = next
c$$$         ia(irow)  = next+1
c$$$ 7       continue
c$$$c
c$$$c perform an in-place permutation of the  arrays.
c$$$c 
c$$$         call ivperm (nnz, ja(ifirst), iwork) 
c$$$         if (values) call dvperm (nnz, a(ifirst), iwork) 
c$$$c
c$$$c reshift the pointers of the original matrix back.
c$$$c 
c$$$      do 8 i=n,1,-1
c$$$         ia(i+1) = ia(i)
c$$$ 8    continue
c$$$      ia(1) = ifirst 
c$$$c
c$$$      return 
c$$$c---------------end-of-csort-------------------------------------------- 
c$$$c-----------------------------------------------------------------------
c$$$      end

      subroutine dvperm (n, x, perm) 
      integer n, perm(n) 
      real*8 x(n)
c-----------------------------------------------------------------------
c this subroutine performs an in-place permutation of a real vector x 
c according to the permutation array perm(*), i.e., on return, 
c the vector x satisfies,
c
c	x(perm(j)) :== x(j), j=1,2,.., n
c
c-----------------------------------------------------------------------
c on entry:
c---------
c n 	= length of vector x.
c perm 	= integer array of length n containing the permutation  array.
c x	= input vector
c
c on return:
c---------- 
c x	= vector x permuted according to x(perm(*)) :=  x(*)
c
c----------------------------------------------------------------------c
c           Y. Saad, Sep. 21 1989                                      c
c----------------------------------------------------------------------c
c local variables 
      real*8 tmp, tmp1
c
      init      = 1
      tmp	= x(init)	
      ii        = perm(init)
      perm(init)= -perm(init)
      k         = 0
c     
c loop
c 
 6    k = k+1
c
c save the chased element --
c 
      tmp1	  = x(ii) 
      x(ii)     = tmp
      next	  = perm(ii) 
      if (next .lt. 0 ) goto 65
c     
c test for end 
c
      if (k .gt. n) goto 101
      tmp       = tmp1
      perm(ii)  = - perm(ii)
      ii        = next 
c
c end loop 
c
      goto 6
c
c reinitilaize cycle --
c
 65   init      = init+1
      if (init .gt. n) goto 101
      if (perm(init) .lt. 0) goto 65
      tmp	= x(init)
      ii	= perm(init)
      perm(init)=-perm(init)
      goto 6
c     
 101  continue
      do 200 j=1, n
         perm(j) = -perm(j)
 200  continue 
c     
      return
c-------------------end-of-dvperm--------------------------------------- 
c-----------------------------------------------------------------------
      end


      subroutine ivperm (n, ix, perm) 
      integer n, perm(n), ix(n)
c-----------------------------------------------------------------------
c this subroutine performs an in-place permutation of an integer vector 
c ix according to the permutation array perm(*), i.e., on return, 
c the vector x satisfies,
c
c	ix(perm(j)) :== ix(j), j=1,2,.., n
c
c-----------------------------------------------------------------------
c on entry:
c---------
c n 	= length of vector x.
c perm 	= integer array of length n containing the permutation  array.
c ix	= input vector
c
c on return:
c---------- 
c ix	= vector x permuted according to ix(perm(*)) :=  ix(*)
c
c----------------------------------------------------------------------c
c           Y. Saad, Sep. 21 1989                                      c
c----------------------------------------------------------------------c
c local variables
      integer tmp, tmp1
c
      init      = 1
      tmp	= ix(init)	
      ii        = perm(init)
      perm(init)= -perm(init)
      k         = 0
c     
c loop
c 
 6    k = k+1
c
c save the chased element --
c 
      tmp1	  = ix(ii) 
      ix(ii)     = tmp
      next	  = perm(ii) 
      if (next .lt. 0 ) goto 65
c     
c test for end 
c
      if (k .gt. n) goto 101
      tmp       = tmp1
      perm(ii)  = - perm(ii)
      ii        = next 
c
c end loop 
c
      goto 6
c
c reinitilaize cycle --
c
 65   init      = init+1
      if (init .gt. n) goto 101
      if (perm(init) .lt. 0) goto 65
      tmp	= ix(init)
      ii	= perm(init)
      perm(init)=-perm(init)
      goto 6
c     
 101  continue
      do 200 j=1, n
         perm(j) = -perm(j)
 200  continue 
c     
      return
c-------------------end-of-ivperm--------------------------------------- 
c-----------------------------------------------------------------------
      end



      subroutine transp (nrow,ncol,a,ja,ia,iwk,ierr)
      integer nrow, ncol, ia(*), ja(*), iwk(*), ierr
      real*8 a(*) 
c------------------------------------------------------------------------
c In-place transposition routine.
c------------------------------------------------------------------------
c this subroutine transposes a matrix stored in compressed sparse row 
c format. the transposition is done in place in that the arrays a,ja,ia
c of the transpose are overwritten onto the original arrays.
c------------------------------------------------------------------------
c on entry:
c--------- 
c nrow	= integer. The row dimension of A.
c ncol	= integer. The column dimension of A.
c a	= real array of size nnz (number of nonzero elements in A).
c         containing the nonzero elements 
c ja	= integer array of length nnz containing the column positions
c 	  of the corresponding elements in a.
c ia	= integer of size n+1, where n = max(nrow,ncol). On entry
c         ia(k) contains the position in a,ja of  the beginning of 
c         the k-th row.
c
c iwk	= integer work array of same length as ja.
c
c on return:
c----------
c
c ncol	= actual row dimension of the transpose of the input matrix.
c         Note that this may be .le. the input value for ncol, in
c         case some of the last columns of the input matrix are zero
c         columns. In the case where the actual number of rows found
c         in transp(A) exceeds the input value of ncol, transp will
c         return without completing the transposition. see ierr.
c a,
c ja,
c ia	= contains the transposed matrix in compressed sparse
c         row format. The row dimension of a, ja, ia is now ncol.
c
c ierr	= integer. error message. If the number of rows for the
c         transposed matrix exceeds the input value of ncol,
c         then ierr is  set to that number and transp quits.
c         Otherwise ierr is set to 0 (normal return).
c
c Note: 
c----- 1) If you do not need the transposition to be done in place
c         it is preferrable to use the conversion routine csrcsc 
c         (see conversion routines in formats).
c      2) the entries of the output matrix are not sorted (the column
c         indices in each are not in increasing order) use csrcsc
c         if you want them sorted.
c----------------------------------------------------------------------c
c           Y. Saad, Sep. 21 1989                                      c
c  modified Oct. 11, 1989.                                             c
c----------------------------------------------------------------------c
c local variables
      real*8 t, t1
      ierr = 0
      nnz = ia(nrow+1)-1
c
c     determine column dimension
c
      jcol = 0
      do 1 k=1, nnz
         jcol = max(jcol,ja(k))
 1    continue
      if (jcol .gt. ncol) then
         ierr = jcol
         return
      endif
c     
c     convert to coordinate format. use iwk for row indices.
c     
      ncol = jcol
c     
      do 3 i=1,nrow
         do 2 k=ia(i),ia(i+1)-1
            iwk(k) = i
 2       continue 
 3    continue
c     find pointer array for transpose. 
      do 35 i=1,ncol+1
         ia(i) = 0
 35   continue
      do 4 k=1,nnz
         i = ja(k)
         ia(i+1) = ia(i+1)+1
 4    continue 
      ia(1) = 1 
c------------------------------------------------------------------------
      do 44 i=1,ncol
         ia(i+1) = ia(i) + ia(i+1)
 44   continue 
c     
c     loop for a cycle in chasing process. 
c     
      init = 1
      k = 0
 5    t = a(init)
      i = ja(init)
      j = iwk(init)
      iwk(init) = -1
c------------------------------------------------------------------------
 6    k = k+1 		
c     current row number is i.  determine  where to go. 
      l = ia(i)
c     save the chased element. 
      t1 = a(l)
      inext = ja(l)
c     then occupy its location.
      a(l)  = t
      ja(l) = j
c     update pointer information for next element to be put in row i. 
      ia(i) = l+1
c     determine  next element to be chased
      if (iwk(l) .lt. 0) goto 65
      t = t1
      i = inext
      j = iwk(l)
      iwk(l) = -1
      if (k .lt. nnz) goto 6
      goto 70
 65   init = init+1
      if (init .gt. nnz) goto 70
      if (iwk(init) .lt. 0) goto 65
c     restart chasing --	
      goto 5
 70   continue
      do 80 i=ncol,1,-1 
         ia(i+1) = ia(i)
 80   continue
      ia(1) = 1
c
      return
c------------------end-of-transp ----------------------------------------
c------------------------------------------------------------------------
      end 


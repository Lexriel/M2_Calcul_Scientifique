module some_fun_saad
  implicit none
contains 

    subroutine amux (n, x, y, a,ja,ia) 
      implicit none
      integer :: n
      real (kind=8), dimension(:) :: x,y,a
      integer, dimension(:) :: ja,ia
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
      real (kind=8) :: t
      integer :: i, k
      !-----------------------------------------------------------------------
      do i = 1,n
         !
         !     compute the inner product of row i with vector x
         ! 
         t = 0.00
         do k=ia(i), ia(i+1)-1 
            t = t + a(k)*x(ja(k))
         end do
         !
         !     store result in y(i) 
         !
         y(i) = t
      end do
      !
      return
      !---------end-of-amux---------------------------------------------------
      !-----------------------------------------------------------------------
    end subroutine amux


  FUNCTION GETELM (i,j,a,ja,ia,iadd,sorted) 
    !-----------------------------------------------------------------------
    !     purpose:
    !     -------- 
    !     this function returns the element a(i,j) of a matrix a, 
    !     for any pair (i,j).  the matrix is assumed to be stored 
    !     in compressed sparse row (csr) format. getelm performs a
    !     binary search in the case where it is known that the elements 
    !     are sorted so that the column indices are in increasing order. 
    !     also returns (in iadd) the address of the element a(i,j) in 
    !     arrays a and ja when the search is successsful (zero if not).
    !----- 
    !     first contributed by noel nachtigal (mit). 
    !     recoded jan. 20, 1991, by y. saad [in particular
    !     added handling of the non-sorted case + the iadd output] 
    !-----------------------------------------------------------------------
    !     parameters:
    !     ----------- 
    ! on entry: 
    !---------- 
    !     i      = the row index of the element sought (input).
    !     j      = the column index of the element sought (input).
    !     a      = the matrix a in compressed sparse row format (input).
    !     ja     = the array of column indices (input).
    !     ia     = the array of pointers to the rows' data (input).
    !     sorted = logical indicating whether the matrix is knonw to 
    !              have its column indices sorted in increasing order 
    !              (sorted=.true.) or not (sorted=.false.).
    !              (input). 
    ! on return:
    !----------- 
    !     getelm = value of a(i,j). 
    !     iadd   = address of element a(i,j) in arrays a, ja if found,
    !              zero if not found. (output) 
    !
    !     note: the inputs i and j are not checked for validity. 
    !-----------------------------------------------------------------------
    !     noel m. nachtigal october 28, 1990 -- youcef saad jan 20, 1991.
    !----------------------------------------------------------------------- 
    real(kind=8) :: getelm
    integer :: i,j,iadd
    integer, dimension(:) :: ia,ja
    real(kind=8), dimension(:) :: a
    logical :: sorted 
    !
    !     local variables.
    !
    integer ibeg, iend, imid, k
    !
    !     initialization 
    !
    iadd = 0 
    getelm = 0.0
    ibeg = ia(i)
    iend = ia(i+1)-1
    !
    !     case where matrix is not necessarily sorted
    !     
    if (.not. sorted) then 
       !
       ! scan the row - exit as soon as a(i,j) is found
       !
       do k=ibeg, iend
          if (ja(k) .eq.  j) then
             iadd = k 
             goto 20
          endif
       end do
       !     
       !     end unsorted case. begin sorted case
       !     
    else
       !     
       !     begin binary search.   compute the middle index.
       !     
10     imid = ( ibeg + iend ) / 2
       !     
       !     test if  found
       !     
       if (ja(imid).eq.j) then
          iadd = imid 
          goto 20
       endif
       if (ibeg .ge. iend) goto 20
       !     
       !     else     update the interval bounds. 
       !     
       if (ja(imid).gt.j) then
          iend = imid -1
       else 
          ibeg = imid +1
       endif
       goto 10  
       !     
       !     end both cases
       !     
    endif
    !     
20  if (iadd .ne. 0) getelm = a(iadd) 
    !
    return
    !--------end-of-getelm--------------------------------------------------
    !-----------------------------------------------------------------------
  end FUNCTION GETELM

  subroutine csort (n,a,ja,ia,iwork,values) 
    implicit none
    logical, intent(in) ::  values
    integer, intent(in) ::  n
    integer, dimension(n+1), intent(inout) :: ia
    integer, dimension(:), intent(inout) :: ja,iwork
    real (kind=8), dimension(:), intent(inout) ::  a
    !-----------------------------------------------------------------------
    ! This routine sorts the elements of  a matrix (stored in Compressed
    ! Sparse Row Format) in increasing order of their column indices within 
    ! each row. It uses a form of bucket sort with a cost of O(nnz) where
    ! nnz = number of nonzero elements. 
    ! requires an integer work array of length 2*nnz.  
    !-----------------------------------------------------------------------
    ! on entry:
    !--------- 
    ! n     = the row dimension of the matrix
    ! a     = the matrix A in compressed sparse row format.
    ! ja    = the array of column indices of the elements in array a.
    ! ia    = the array of pointers to the rows. 
    ! iwork = integer work array of length max ( n+1, 2*nnz ) 
    !         where nnz = (ia(n+1)-ia(1))  ) .
    ! values= logical indicating whether or not the real values a(*) must 
    !         also be permuted. if (.not. values) then the array a is not
    !         touched by csort and can be a dummy array. 
    ! 
    ! on return:
    !----------
    ! the matrix stored in the structure a, ja, ia is permuted in such a
    ! way that the column indices are in increasing order within each row.
    ! iwork(1:nnz) contains the permutation used  to rearrange the elements.
    !----------------------------------------------------------------------- 
    ! Y. Saad - Feb. 1, 1991.
    !-----------------------------------------------------------------------
    ! local variables
    integer :: i, k, j, ifirst, nnz, next, ko,irow
    !
    ! count the number of elements in each column
    !
!!$    do i=1,n+1
!!$       iwork(i) = 0
!!$    end do
    iwork=0
    do i=1, n
       do k=ia(i), ia(i+1)-1 
          j = ja(k)+1
          iwork(j) = iwork(j)+1
       end do
    end do
    !
    ! compute pointers from lengths. 
    !
    iwork(1) = 1
    do i=1,n
       iwork(i+1) = iwork(i) + iwork(i+1)
    end do
    ! 
    ! get the positions of the nonzero elements in order of columns.
    !
    ifirst = ia(1) 
    nnz = ia(n+1)-ifirst
    do i=1,n
       do k=ia(i),ia(i+1)-1 
          j = ja(k) 
          next = iwork(j) 
          iwork(nnz+next) = k
          iwork(j) = next+1
       end do
    end do
    !
    ! convert to coordinate format
    ! 
      do i=1, n
         do k=ia(i), ia(i+1)-1 
            iwork(k) = i
         end do
      end do
      !
      ! loop to find permutation: for each element find the correct 
      ! position in (sorted) arrays a, ja. Record this in iwork. 
      ! 
      do k=1, nnz
         ko = iwork(nnz+k) 
         irow = iwork(ko)
         next = ia(irow)
         !
         ! the current element should go in next position in row. iwork
         ! records this position. 
         ! 
         iwork(ko) = next
         ia(irow)  = next+1
      end do
      !
      ! perform an in-place permutation of the  arrays.
      ! 
      call ivperm (nnz, ja(ifirst:), iwork) 
      if (values) call dvperm (nnz, a(ifirst:), iwork) 
      !
      ! reshift the pointers of the original matrix back.
      ! 
      do i=n,1,-1
         ia(i+1) = ia(i)
      end do
      ia(1) = ifirst 
      !
      return 
      !---------------end-of-csort-------------------------------------------- 
      !-----------------------------------------------------------------------
    end subroutine csort

    subroutine dvperm (n, ix, perm) 
      integer, intent(in) :: n
      integer, intent(inout), dimension(n) :: perm
      real (kind=8), intent(inout), dimension(n) ::  ix
      !-----------------------------------------------------------------------
      ! this subroutine performs an in-place permutation of a real vector x 
      ! according to the permutation array perm(*), i.e., on return, 
      ! the vector x satisfies,
      !
      !	x(perm(j)) :== x(j), j=1,2,.., n
      !
      !-----------------------------------------------------------------------
      ! on entry:
      !---------
      ! n 	= length of vector x.
      ! perm 	= integer array of length n containing the permutation  array.
      ! x	= input vector
      !
      ! on return:
      !---------- 
      ! x	= vector x permuted according to x(perm(*)) :=  x(*)
      !
      !----------------------------------------------------------------------c
      !           Y. Saad, Sep. 21 1989                                      c
      !----------------------------------------------------------------------c
      ! local variables 
      real (kind=8) ::  tmp, tmp1
      integer :: init,ii,next,k,j
      !
      init       = 1
      tmp	       = ix(init)	
      ii         = perm(init)
      perm(init) = -perm(init)
      k          = 0
      loop6 : DO
         k=k+1
         tmp1   = ix(ii) 
         ix(ii) = tmp
         next   = perm(ii) 
         IF (next.LT.0) THEN
            DO 
               init=init+1
               IF (init.GT.n) EXIT loop6
               IF (perm(init).GE.0) EXIT
            END DO
            tmp	= ix(init)
            ii	= perm(init)
            perm(init)=-perm(init)
            CYCLE loop6
         END IF
         IF (k.GT.n) EXIT loop6
         tmp       = tmp1
         perm(ii)  = - perm(ii)
         ii        = next 
      END DO loop6
      !     
      DO j=1, n
         perm(j) = -perm(j)
      END DO
      !     
      return
      !-------------------end-of-dvperm--------------------------------------- 
      !-----------------------------------------------------------------------
    end subroutine dvperm

    !-----------------------------------------------------------------------
    subroutine ivperm (n, ix, perm) 
      integer, intent(in) :: n
      integer, intent(inout), dimension(n) :: perm, ix
      !-----------------------------------------------------------------------
      ! this subroutine performs an in-place permutation of an integer vector 
      ! ix according to the permutation array perm(*), i.e., on return, 
      ! the vector x satisfies,
      !
      !	ix(perm(j)) :== ix(j), j=1,2,.., n
      !
      !-----------------------------------------------------------------------
      ! on entry:
      !---------
      ! n 	= length of vector x.
      ! perm 	= integer array of length n containing the permutation  array.
      ! ix	= input vector
      !
      ! on return:
      !---------- 
      ! ix	= vector x permuted according to ix(perm(*)) :=  ix(*)
      !
      !----------------------------------------------------------------------c
      !           Y. Saad, Sep. 21 1989                                      c
      !----------------------------------------------------------------------c
      ! local variables
      integer tmp, tmp1
      integer :: init,ii,next,k,j
      !
      init       = 1
      tmp	       = ix(init)	
      ii         = perm(init)
      perm(init) = -perm(init)
      k          = 0
      loop6 : DO
         k=k+1
         tmp1   = ix(ii) 
         ix(ii) = tmp
         next   = perm(ii) 
         IF (next.LT.0) THEN
            DO 
               init=init+1
               IF (init.GT.n) EXIT loop6
               IF (perm(init).GE.0) EXIT
            END DO
            tmp	= ix(init)
            ii	= perm(init)
            perm(init)=-perm(init)
            CYCLE loop6
         END IF
         IF (k.GT.n) EXIT loop6
         tmp       = tmp1
         perm(ii)  = - perm(ii)
         ii        = next 
      END DO loop6
      !     
      DO j=1, n
         perm(j) = -perm(j)
      END DO
      !     
      return
      !-------------------end-of-ivperm--------------------------------------- 
      !-----------------------------------------------------------------------
    end subroutine ivperm


    subroutine pspltm(nrow,ncol,mode,ja,ia,title,ptitle,size,munt,nlines,lines,iunt)
      !-----------------------------------------------------------------------
      integer nrow,ncol,nlines,ptitle,mode,iunt, ja(*), ia(*), lines(nlines) 
      real size
      character title*(*), munt*2 
      !----------------------------------------------------------------------- 
      ! PSPLTM - PostScript PLoTer of a (sparse) Matrix
      ! This version by loris renggli (renggli@masg1.epfl.ch), Dec 1991
      ! and Youcef Saad 
      !------
      ! Loris RENGGLI, Swiss Federal Institute of Technology, Math. Dept
      ! CH-1015 Lausanne (Switzerland)  -- e-mail:  renggli@masg1.epfl.ch
      ! Modified by Youcef Saad -- June 24, 1992 to add a few features:
      ! separation lines + acceptance of MSR format.
      !-----------------------------------------------------------------------
      ! input arguments description :
      !
      ! nrow   = number of rows in matrix
      !
      ! ncol   = number of columns in matrix 
      !
      ! mode   = integer indicating whether the matrix is stored in 
      !           CSR mode (mode=0) or CSC mode (mode=1) or MSR mode (mode=2) 
      !
      ! ja     = column indices of nonzero elements when matrix is
      !          stored rowise. Row indices if stores column-wise.
      ! ia     = integer array of containing the pointers to the 
      !          beginning of the columns in arrays a, ja.
      !
      ! title  = character*(*). a title of arbitrary length to be printed 
      !          as a caption to the figure. Can be a blank character if no
      !          caption is desired.
      !
      ! ptitle = position of title; 0 under the drawing, else above
      !
      ! size   = size of the drawing  
      !
      ! munt   = units used for size : 'cm' or 'in'
      !
      ! nlines = number of separation lines to draw for showing a partionning
      !          of the matrix. enter zero if no partition lines are wanted.
      !
      ! lines  = integer array of length nlines containing the coordinates of 
      !          the desired partition lines . The partitioning is symmetric: 
      !          a horizontal line across the matrix will be drawn in 
      !          between rows lines(i) and lines(i)+1 for i=1, 2, ..., nlines
      !          an a vertical line will be similarly drawn between columns
      !          lines(i) and lines(i)+1 for i=1,2,...,nlines 
      !
      ! iunt   = logical unit number where to write the matrix into.
      !----------------------------------------------------------------------- 
      ! additional note: use of 'cm' assumes european format for paper size
      ! (21cm wide) and use of 'in' assumes american format (8.5in wide).
      ! The correct centering of the figure depends on the proper choice. Y.S.
      !-----------------------------------------------------------------------
      ! external 
!!$      integer LENSTR
!!$      external LENSTR
      ! local variables ---------------------------------------------------
      integer n,nr,nc,maxdim,istart,ilast,ii,k,ltit,m,kol,isep
      real lrmrgn,botmrgn,xtit,ytit,ytitof,fnstit,siz
      real xl,xr, yb,yt, scfct,u2dot,frlw,delt,paperx,conv,xx,yy
      logical square 
      ! change square to .true. if you prefer a square frame around
      ! a rectangular matrix
      real :: haf,zero
      haf=0.5
      zero=0.0
      conv=2.54
      square=.false.
      !-----------------------------------------------------------------------
      siz = size
      nr = nrow
      nc = ncol
      n = nc
      if (mode .eq. 0) n = nr
      !      nnz = ia(n+1) - ia(1) 
      maxdim = max(nrow, ncol)
      m = 1 + maxdim
      nc = nc+1
      nr = nr+1
      !
      ! units (cm or in) to dot conversion factor and paper size
      ! 
      if (munt.eq.'cm' .or. munt.eq.'CM') then
         u2dot = 72.0/conv
         paperx = 21.0
      else
         u2dot = 72.0
         paperx = 8.5*conv
         siz = siz*conv
      end if
      !
      ! left and right margins (drawing is centered)
      ! 
      lrmrgn = (paperx-siz)/2.0
      !
      ! bottom margin : 2 cm
      !
      botmrgn = 2.0
      ! scaling factor
      scfct = siz*u2dot/m
      ! matrix frame line witdh
      frlw = 0.25
      ! font size for title (cm)
      fnstit = 0.5
      ltit = LENSTR(title)
      ! position of title : centered horizontally
      !                     at 1.0 cm vertically over the drawing
      ytitof = 1.0
      xtit = paperx/2.0
      ytit = botmrgn+siz*nr/m + ytitof
      ! almost exact bounding box
      xl = lrmrgn*u2dot - scfct*frlw/2
      xr = (lrmrgn+siz)*u2dot + scfct*frlw/2
      yb = botmrgn*u2dot - scfct*frlw/2
      yt = (botmrgn+siz*nr/m)*u2dot + scfct*frlw/2
      if (ltit.gt.0) then
        yt = yt + (ytitof+fnstit*0.70)*u2dot
      end if
      ! add some room to bounding box
      delt = 10.0
      xl = xl-delt
      xr = xr+delt
      yb = yb-delt
      yt = yt+delt
      !
      ! correction for title under the drawing
      if (ptitle.eq.0 .and. ltit.gt.0) then
         ytit = botmrgn + fnstit*0.3
         botmrgn = botmrgn + ytitof + fnstit*0.7
      end if
      ! begin of output
      !
      write(iunt,10) '%!'
      write(iunt,10) '%%Creator: PSPLTM routine'
      write(iunt,12) '%%BoundingBox:',xl,yb,xr,yt
      write(iunt,10) '%%EndComments'
      write(iunt,10) '/cm {72 mul 2.54 div} def'
      write(iunt,10) '/mc {72 div 2.54 mul} def'
      write(iunt,10) '/pnum { 72 div 2.54 mul 20 string'
      write(iunt,10) 'cvs print ( ) print} def'
      write(iunt,10) '/Cshow {dup stringwidth pop -2 div 0 rmoveto show} def'
      !
      ! we leave margins etc. in cm so it is easy to modify them if
      ! needed by editing the output file
      write(iunt,10) 'gsave'
      if (ltit.gt.0) then
         write(iunt,*) '/Helvetica findfont ',fnstit,' cm scalefont setfont '
         write(iunt,*) xtit,' cm ',ytit,' cm moveto '
         write(iunt,'(3A)') '(',title(1:ltit),') Cshow'
      end if
      write(iunt,*) lrmrgn,' cm ',botmrgn,' cm translate'
      write(iunt,*) siz,' cm ',m,' div dup scale '
      !------- 
      ! draw a frame around the matrix
      write(iunt,*) frlw,' setlinewidth'
      write(iunt,10) 'newpath'
      write(iunt,11) 0, 0, ' moveto'
      if (square) then
         write(iunt,11) m,0,' lineto'
         write(iunt,11) m, m, ' lineto'
         write(iunt,11) 0,m,' lineto'
      else
         write(iunt,11) nc,0,' lineto'
         write(iunt,11) nc,nr,' lineto'
         write(iunt,11) 0,nr,' lineto'
      end if
      write(iunt,10) 'closepath stroke'
      !
      !     drawing the separation lines 
      ! 
      write(iunt,*)  ' 0.2 setlinewidth'
      do kol=1, nlines 
         isep = lines(kol) 
         !
         !     horizontal lines 
         !
         yy =  real(nrow-isep) + haf 
         xx = real(ncol+1) 
         write(iunt,13) zero, yy, ' moveto '
         write(iunt,13)  xx, yy, ' lineto stroke '
         !
         ! vertical lines 
         !
         xx = real(isep) + haf 
         yy = real(nrow+1)  
         write(iunt,13) xx, zero,' moveto '
         write(iunt,13) xx, yy, ' lineto stroke '             
      end do
      ! 
      !----------- plotting loop ---------------------------------------------
      !
      write(iunt,10) '0 0 1 setrgbcolor'
      write(iunt,10) '1 1 translate'
      write(iunt,10) '0.8 setlinewidth'
      write(iunt,10) '/p {moveto 0 -.40 rmoveto '
      write(iunt,10) '           0  .80 rlineto stroke} def'
      !     
      do ii=1, n
         istart = ia(ii)
         ilast  = ia(ii+1)-1 
         if (mode .eq. 1) then
            do k=istart, ilast
               write(iunt,11) ii-1, nrow-ja(k), ' p'
            end do
         else
            do k=istart, ilast
               write(iunt,11) ja(k)-1, nrow-ii, ' p'
            end do
            ! add diagonal element if MSR mode.
            if (mode .eq. 2) write(iunt,11) ii-1, nrow-ii, ' p' 
            !
         endif
      end do
      !-----------------------------------------------------------------------
      write(iunt,10) 'showpage'
      return
      !
10    format (A)
11    format (2(I6,1x),A)
12    format (A,4(1x,F9.2))
13    format (2(F9.2,1x),A)
      !-----------------------------------------------------------------------
    end subroutine pspltm


    integer function lenstr(s)
      !-----------------------------------------------------------------------
      ! return length of the string S
      !-----------------------------------------------------------------------
      character*(*) s
      integer len
      intrinsic len
      integer n
      !----------------------------------------------------------------------- 
      n = len(s)
10    continue
      if (s(n:n).eq.' ') then
         n = n-1
         if (n.gt.0) go to 10
      end if
      lenstr = n
      !
      return
      !--------end-of-pspltm--------------------------------------------------
      !-----------------------------------------------------------------------
    end function lenstr



  end module some_fun_saad

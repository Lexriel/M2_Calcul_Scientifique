module part4_module

  IMPLICIT NONE

CONTAINS

  SUBROUTINE compute_LU(N, h)

    integer, intent(in) :: N ! size
    integer :: i, j
    double precision, intent(in) :: h ! step
    double precision :: ln, dn ! temporary l_n and d_n defined in the question 4.2
    double precision, dimension(N, N) :: L, U


! Writing of L

    L = 0.
    L(2,1) = -h/3.
    L(3,2) = 1./4.
    L(4,3) = 2./7.

    do i=1,N
       L(i,i) = 1.
    end do

    ln = L(4,3)
    do i=5,N-1
       ln = 1./(4.-ln)
       L(i,i-1) = ln
    end do

    L(N, N-2) = -3.*ln/h
    L(N, N-1) = ( ln/(4.-ln) )*3./h

    open(unit=100,file='L.dat',status='unknown')

    do i=1,N
       do j=1,N
          write(100,*) L(i,j)
       end do
    end do


! Writing of U

    U = 0.
    U(1,1) = -3./h
    U(1,3) = -U(1,1)
    U(2,3) = 2.
    U(2,2) = 4.
    U(3,3) = 7./2.

    do i=3,N-1
       U(i,i+1) = 1.
    end do

    dn = U(3,3)

    do i=4,N-1
       dn = 4.-1./dn
       U(i,i) = dn
    end do

    dn = 1.-ln/dn ! does 1 - l(N)/d(N+1)
    U(N, N) = 3.*dn/h

    open(unit=101,file='U.dat',status='unknown')

    do i=1,N
       do j=1,N
          write(101,*) U(i,j)
       end do
    end do

    close(100)
    close(101)

  end subroutine compute_LU

end module part4_module

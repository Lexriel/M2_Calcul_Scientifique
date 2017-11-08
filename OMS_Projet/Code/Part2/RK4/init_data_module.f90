module init_data_module

implicit none

contains

subroutine init_data(A0)

double precision :: x0, y0, v_x0, v_y0, eps_x, eps_y
double precision, dimension(4), intent(in) :: A0

    eps_x = 2.36*10.**(-5.)
    eps_y = eps_x 

! **************************************************** !
!                                                      !
!   Writes in the file 'A0.dat' the initial data       !
!   A(0) = (/ a(0), da/dz(0), b(0), db/dz(0) /)        !
!   after correction so as A be S-periodic solutions   !
!   of the envelope equations.                         !
!                                                      !
! **************************************************** !

x0 = A0(1)/2.

y0 = A0(3)/2.

v_x0 = 1/2. * sqrt(eps_x**2/A0(1)**2. + A0(2)**2.)

v_y0 = 1/2. * sqrt(eps_y**2/A0(3)**2. + A0(4)**2.)


! **************************************************** !
!                                                      !
!   Writes in the file 'x0_y0_vx0_vy0.dat'             !
!   the data (/ x0, y0, vx0, vy0 /).                   !
!                                                      !
! **************************************************** !

open(unit=106, file='x0_y0_vx0_vy0.dat', status='unknown')
write(106,*) x0
write(106,*) y0
write(106,*) v_x0
write(106,*) v_y0

close(106)

END SUBROUTINE init_data

end module init_data_module

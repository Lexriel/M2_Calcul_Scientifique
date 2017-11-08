module beulerN_module

use broyden_module

implicit none

contains

    function odefun(t,y,N) result(z)
      implicit none
      integer, intent(in) :: N
      real, intent(in), dimension(N) :: y
      real, intent(in) :: t
      real, dimension(N) :: z
! Définition d'une ODE 
      z = (/ 3*y(1), y(2)**2, y(3) /)
    end function odefun


  subroutine beulerN(odefun, tspan ,y0 ,Nh)

    implicit none

    interface

       function odefun(t,x,N) result(z)
         integer,intent(in) :: N
         real, intent(in) :: t
         real, intent(in), dimension(N) :: x
         real, dimension(N) :: z
       end function odefun
    end interface

    real, intent(in), dimension(2) :: tspan
    real, intent(in), dimension(:) :: y0
    integer, intent(in) :: Nh
    integer :: N, i
    real, dimension(:), allocatable :: u
    real :: h, t
    
    allocate(u(N))

    N = size(y0)
    h = (tspan(2)-tspan(1)) / Nh
    t = tspan(1)
    u = y0

    open(unit=100, file="beulerN_data.dat", status='unknown')
    write(100,*) t, u
    
    do i=2,Nh
      t = t+h
      u = broyden(G, u, h)
      write(100,*) t, u
    end do

  deallocate(u)
  close(100)

  contains
    function G(x) result(Gx)
      implicit none
      real, intent(in), dimension(N) :: x
      real, dimension(N) :: Gx
      Gx = x-u-h*odefun(t,x,N)
    end function G

  end subroutine beulerN

end module beulerN_module

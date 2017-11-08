module part5_module

  use lapack95
  use blas95


  implicit none

contains

  function rho(x,y)
    double precision, intent(in) :: x, y
    double precision :: rho

    rho = 2.*x-1.+y

  ! If we do the part4, we will choose the values
  ! of rho we compute instead of this function,
  ! calling a file 'rho.dat' copied in the folder
  ! Part4 then pasted in the folder Part5.

  end function rho


  subroutine Solve_Laplacian(Nx,Ny,rho)

    interface
       double precision function rho(x,y)
         double precision, intent(in) :: x, y
       end function rho
    end interface

    integer, intent(in) :: Nx, Ny
    integer :: i,j,k
    double precision :: xmin, xmax, ymin, ymax
    double precision :: dx, dy
    double precision :: b, c, d
    double precision, dimension (Nx*Ny,Nx*Ny) :: A
    double precision, dimension(Nx*Ny) :: second_membre, u
    integer, dimension (Nx*Ny) :: ipiv

    xmin = 0.; xmax = 1.; ymin = 0.; ymax = 1.

    dx = (xmax - xmin)/Nx
    dy = (ymax - ymin)/Ny

    b = 2./(dx**2.) + 2./(dy**2.)
    c = -1./(dx**2.)
    d = -1./(dy**2.)


! ******************* Creation of A ******************* !

    do k=0,Ny-1
       do i=1,Nx
          do j=1,Nx
             if (i == j) then
                A(i + k*Nx, j + k*Nx) = b
             else if ( (i == j+1).or.(i == j-1) ) then
                A(i + k*Nx, j + k*Nx) = c
             else if ( ( (i == 1).and.(j==Nx) ).or.( (j == 1).and.(i==Nx) ) ) then
                A(i + k*Nx, j + k*Nx) = c
             else
                A(i + k*Nx, j + k*Nx) = 0.
             end if
          end do
       end do
    end do


    do k=0,Ny-2
       do i=1,Nx
          do j=1,Nx
             if (i == j ) then
                A(i + k*Nx, j + (k+1)*Nx) = d
                A(i + (k+1)*Nx, j + k*Nx) = d
             else
                A(i + k*Nx, j + (k+1)*Nx) = 0.
                A(i + (k+1)*Nx, j + k*Nx) = 0.
             end if
          end do
       end do
    end do


    do i=1,Nx
       do j=1,Nx
          if (i == j) then
             A(i + (Ny-1)*Nx, j) = d
             A(i, j + (Ny-1)*Nx) = d
          else
             A(i + (Ny-1)*Nx, j) = 0.
             A(i, j + (Ny-1)*Nx) = 0.
          end if
       end do
    end do


! ******************* Writing of A ******************* !
!                                                      !
!     writes A in the file 'A.dat' line per line       !
!                                                      !
! **************************************************** !

    open(unit=100,file='A.dat',status='unknown')

    do i=1,Nx*ny
       do j=1,Nx*Ny
          write(100,*) A(i,j)
       end do
    end do


! ****************** Creation of rho ****************** !

    do j=1,Ny
       do i=1,Nx
          second_membre(i + (j-1)*Nx) = rho(xmin + (i-1)*dx , ymin + (j-1)*dy)
       end do
    end do

! ************* Writing of second member ************* !
!                                                      !
!     writes A in the file 'A.dat' line per line       !
!                                                      !
! **************************************************** !

    open(unit=101,file='second_membre.dat',status='unknown')

    do i=1,Nx*ny
       write(101,*) second_membre(i)
    end do


! ************* Resolution of the system ************* !

    call getrf(A,ipiv)
    call getrs(A,ipiv, second_membre)

    u = second_membre


! ******************* Writing of u ******************* !

    open(unit=102,file='u.dat',status='unknown')

    do i=1,Nx*ny
       write(102,*) u(i)
    end do


    close(100)
    close(101)
    close(102)

  end SUBROUTINE Solve_Laplacian

end module part5_module

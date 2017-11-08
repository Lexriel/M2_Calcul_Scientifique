module spline

  include 'mpif.h'

CONTAINS

  !---------Source function to test the program----------!

  function f(x,y)

    double precision, intent(in) :: x, y
    double precision :: f
    f= x**2+y**2

  end function f

  !---------w function containing the w coefficients-----!

  function w(p)

    integer, intent(in) :: p
    double precision :: w

    if (p==1) then
       w = 15126./18817.
    else if (p==2) then
       w = -4053./18817.
    else if (p==3) then
       w = 1086./18817.
    else if (p==4) then
       w = -291./18817.
    else if (p==5) then
       w = 78./18817.
    else if (p==6) then
       w = -503./415608.
    else if (p==7) then
       w = 17./56451.
    else if (p==8) then
       w = -3./37634.
    else if (p==9) then
       w = 1./56451.
    else if (p==10) then
       w = -1./415608.
    else 
       w = 0
    end if

  end function w

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  SUBROUTINE Build_F_mat()

    IMPLICIT NONE

    integer :: I_proc, J_proc, i, j, k, l, Px, Py, Nx, Ny
    integer :: rank, P, code
    double precision :: xmin, xmax, ymin, ymax
    double precision :: x_I, y_J, dx, dy
    double precision, dimension (:,:), allocatable :: F_mat, Fx, Fy, Fxy
    double precision, dimension (:,:), allocatable :: x_pack, y_pack 
    double precision, dimension (:,:), allocatable :: up_recv, down_recv, left_recv, right_recv

    integer, dimension(MPI_STATUS_SIZE) :: statut

    integer :: up, down, left, right


    !------------Initialization of parameters-----------!

    xmin = 0.; xmax = 3.; ymin = 0.; ymax = 5.
    Px = 3; Py = 5; Nx = 10; Ny = 10

    dx = (xmax - xmin)/(Px*Nx)
    dy = (ymax - ymin)/(Py*Ny)

    allocate ( F_mat(Nx+23,Ny+23) )

    !--------------MPI-related parameters---------------!


    call MPI_INIT(code)

    call MPI_COMM_SIZE(MPI_COMM_WORLD,P,code)
    call MPI_COMM_RANK(MPI_COMM_WORLD,rank,code)



    !---Each process gets his (I_proc,J_proc) coordinates)--!
    !-------and computes base values on its subdomain-------!

    I_proc = rank / Py
    J_proc = mod(rank,Py)

    x_I = xmin + I_proc *(xmax-xmin)/Px
    y_J = ymin + J_proc *(ymax-ymin)/Py


    do i=1,Nx+1
       do j=1,Ny+1 
          F_mat(i+11,j+11) = f(x_I + (i-1)*dx , y_J + (j-1)*dy)
       end do
    end do


    !---------Each process determines its neighbours---------!

    if (J_proc == 0) then
       right =  rank + 1
       left = rank + (Py - 1)
    else if (J_proc == Py -1) then
       right = rank - (Py - 1)
       left = rank - 1
    else
       right = rank + 1
       left = rank - 1
    end if


    if (I_proc == 0) then
       up =  rank + (Px -1)*Py
       down = rank + Py
    else if (I_proc == Px -1) then
       up = rank - Py
       down = rank - (Px -1)*Py
    else
       up = rank - Py
       down = rank + Py
    end if



    !-------Send, recv & storing of North and South data-------!

    allocate ( x_pack(11,(Ny+1)) )
    allocate ( down_recv(11,(Ny+1)) )

    do j=1,Ny+1
       do i=1,11
          x_pack(i,j) = F_mat(i+11+1,j+11)
       end do
    end do

!!!

    call MPI_Send(x_pack,11*(Ny+1), MPI_DOUBLE_PRECISION, up, 0, MPI_COMM_WORLD, code)
    call MPI_Recv(down_recv,11*(Ny+1), MPI_DOUBLE_PRECISION, down, 0, MPI_COMM_WORLD, statut, code)


    do i=1,11
       do j=1,Ny+1
          F_mat(i+Nx+1+11,j+11) = down_recv(i,j)
       end do
    end do

    deallocate (down_recv)


    allocate (up_recv(11,(Ny+1)))

    do j=1,Ny+1
       do i=1,11
          x_pack(i,j) = F_mat(i+ Nx, j+11)
       end do
    end do

!!!

    call MPI_Send(x_pack,11*(Ny+1), MPI_DOUBLE_PRECISION, down, 0, MPI_COMM_WORLD, code)
    call MPI_Recv(up_recv,11*(Ny+1), MPI_DOUBLE_PRECISION, up, 0, MPI_COMM_WORLD, statut, code)

!!!

    do i=1,11
       do j=1,Ny+1
          F_mat(i, j+11) = up_recv(i,j)
       end do
    end do

    deallocate (x_pack)
    deallocate (up_recv)



    !------Computation of df/dx (except for the corners as we currently lack the data)---------------!
    !------Division by dx hasn't been done yet, we will wait for the corners-------------------------!

    allocate (Fx(2,Ny+3))

    Fx = 0

    do j=1,Ny+3
       do k=1,10
          Fx(1,j) = Fx(1,j) + w(k)*( F_Mat(11+k,j+10) - F_mat(11-k,j+10) )
          Fx(2,j) = Fx(2,j) + w(k)*( F_Mat(Nx+13+k,j+10) - F_mat(Nx+13-k,j+10) )
       end do
    end do



    !-----Send, recv & storing of East and West data------!

    allocate (y_pack((Nx+23),11))
    allocate (right_recv((Nx+23),11))

    do j=1,11
       do i=1,Nx+23
          y_pack(i,j) = F_mat(i,j+11+1)
       end do
    end do

!!!

    call MPI_Send(y_pack,11*(Nx+23), MPI_DOUBLE_PRECISION, left, 0, MPI_COMM_WORLD, code) 
    call MPI_Recv(right_recv,11*(Nx+23), MPI_DOUBLE_PRECISION, right, 0, MPI_COMM_WORLD, statut, code)

!!!

    do j=1,11
       do i=1,Nx+23
          F_mat(i,j+Ny+1+11) = right_recv(i,j)
       end do
    end do

    deallocate (right_recv)

    !--------------------------------!

    allocate (left_recv((Nx+23),11))

    do i=1,Nx+23
       do j=1,11
          y_pack(i,j) = F_mat(i, j+Ny)
       end do
    end do

    call MPI_Send(y_pack,11*(Nx+23), MPI_DOUBLE_PRECISION, right, 0, MPI_COMM_WORLD, code)
    call MPI_Recv(left_recv,11*(Nx+23), MPI_DOUBLE_PRECISION, left, 0, MPI_COMM_WORLD, statut, code)

    do j=1,11
       do i=1,Nx+23
          F_mat(i,j) = left_recv(i,j)
       end do
    end do

    deallocate (y_pack)
    deallocate (left_recv)



    !----Computation of df/dx in the corners----------------------!

    do k=1,10
       Fx(1,1) = Fx(1,1) + w(k)*( F_Mat(11+k,11) - F_mat(11-k,11) )
       Fx(2,1) = Fx(2,1) + w(k)*( F_Mat(Nx+13+k,11) - F_mat(Nx+13-k,11) )
       Fx(1,Ny+3) = Fx(1,Ny+3) + w(k)*( F_Mat(11+k,Ny+13) - F_mat(11-k,Ny+13) )
       Fx(2,Ny+3) = Fx(2,Ny+3) + w(k)*( F_Mat(Nx+13+k,Ny+13) - F_mat(Nx+13-k,Ny+13) )
    end do


    Fx = 1/dx * Fx   ! With this last line, all the df/dx have been computed !





    !------------------Computation of df/dy-----------------------!

    allocate (Fy(Nx+3,2))

    Fy = 0

    do i=1,Nx+3
       do l=1,10
          Fy(i,1) = Fy(i,1) + w(l)*( F_Mat(i+10,11+l) - F_mat(i+10,11-l) )
          Fy(i,2) = Fy(i,2) + w(l)*( F_Mat(i+10,Ny+13+l) - F_mat(i+10,Ny+13-l) )
       end do
    end do

    Fy= 1/dy * Fy



    !-----------------Computation of d2f/dxdy---------------------!

    allocate (Fxy(2,2))

    Fxy=0

    do k=1,10
       do l=1,10
          Fxy(1,1) = Fxy(1,1) + w(l)*w(k)*( F_Mat(11+k,11+l) - F_mat(11-k,11+l) &
               - F_Mat(11+k,11-l) + F_mat(11-k,11-l)  )

          Fxy(1,2) = Fxy(1,2) + w(l)*w(k)*( F_Mat(11+k,Ny+13+l) - F_mat(11-k,Ny+13+l) &
               - F_Mat(11+k,Ny+13-l) + F_mat(11-k,Ny+13-l)  )

          Fxy(2,1) = Fxy(2,1) + w(l)*w(k)*( F_Mat(Nx+13+k,11+l) - F_mat(Nx+13-k,11+l) &
               - F_Mat(Nx+13+k,11-l) + F_mat(Nx+13-k,11-l)  )

          Fxy(2,2) = Fxy(2,2) + w(l)*w(k)*( F_Mat(Nx+13+k,Ny+13+l) - F_mat(Nx+13-k,Ny+13+l) &
               - F_Mat(Nx+13+k,Ny+13-l) + F_mat(Nx+13-k,Ny+13-l)  )
       end do
    end do

    Fxy = 1/(dx*dy) * Fxy


    !************************* Tests *******************************!

    !Test for Fmat (base data only)

    if (rank == 7) then
       do i=1,Nx+1
          do j=1,Ny+1 
             print *, F_mat(i+11,j+11)
          end do
       end do
    end if

    !Test for df/dx

    !if (rank == 7) then
    !do i=1,2
    !	do j=1,Ny+3 
    !	print *, Fx(i,j)
    !	end do
    !end do
    !end if

    !Test for df/dy

    !if (rank == 7) then
    !do j=1,2
    !	do i=1,Nx+3
    !	print *, Fy(i,j)
    !	end do
    !end do
    !end if


    !Test for d2f/dxdy

    !if (rank == 7) then
    !do i=1,2
    !	do j=1,2 
    !	print *, Fxy(i,j)
    !	end do
    !end do
    !end if


    !Test for neighbours!

    if (rank == 7) then
       print *, "left",left
       print *, "right",right
       print *, "up",up
       print *, "down", down
    end if


    call MPI_FINALIZE(code)

    deallocate(F_mat)
    deallocate(Fx,Fy,Fxy)

  end SUBROUTINE Build_F_Mat

end module spline

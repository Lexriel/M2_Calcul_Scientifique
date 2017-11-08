program part4_program

  use part4_module
  use blas95
  use lapack95

  implicit none

  double precision :: dx,dy
  integer :: Nx, Ny, i, j
  double precision, dimension(:,:), allocatable :: Y1, Y2, Y3
  double precision, dimension(:,:), allocatable :: L1, U1, L2, U2
  double precision, dimension(:,:), allocatable :: Data_init

  Nx = 10; Ny = 10
  dx = 0.1; dy= 0.1

  allocate ( Data_init(Nx+3, Ny+3) )
  ! Is supposed to contain Data_init as described in the report,
  ! if we implement correctly Data_init, we should use a file
  ! containing Data_init, here we just choose an arbitrary
  ! Data_init

  Data_init = 1.



  !******************************* Solving linear systems ********************************!
  !                                                                                       !
  ! Step 1 : (Ny + 1) linear systems of size (Nx+3) with right-hand-side as columns of Y1 !
  ! Step 2 :  2 linear systems of size (Nx+3) with right-hand-side as columns of Y2       !
  ! Step 3 : (Nx + 3) linear systems of size (Ny+3) with right-hand-side as columns of Y3 !
  !                                                                                       !
  !***************************************************************************************! 


  !************************************** Step 1 *****************************************!

  ! Creation of matrices L1 and U1

  allocate (  L1(Nx+3,Nx+3) , U1(Nx+3,Nx+3)  )

  call compute_LU(Nx+3, dx)

  open(unit=100,file='L.dat',status='unknown')
  open(unit=101,file='U.dat',status='unknown')

  do i=1,Nx+3
     do j=1,Nx+3
        read(100,*) L1(i,j)
        read(101,*) U1(i,j)
     end do
  end do

  close(100)
  close(101)


  ! Creation of the related right-hand-side Y1

  allocate ( Y1(Nx+3,Ny+1) )

  do i=1,Nx+3
     Y1(i,:) = Data_init(i,2:Ny+2)
  end do

  ! We can solve the systems of step 1

  do j = 1,Ny+1
     call DTRSV('l','n','u',Nx+3,L1,Nx+3,Y1(:,j),1)
     call DTRSV('u','n','n',Nx+3,U1,Nx+3,Y1(:,j),1)
  end do
  ! these subroutines allow us to solve firstly 
  ! the equation L.y = b, then, when we know y,
  ! we solve U.x = y to get x. Thus, we have
  ! solved LU.x = b (with A=LU).


  !************************************** Step 2 *****************************************!

  ! Creation of the related right-hand-side Y2

  allocate( Y2 (Nx+3,2) )

  Y2(:,1) = Data_init(:,1)
  Y2(:,2) = Data_init(:,Ny+3)

  ! We can solve the 2 systems of size (Nx+3)

  do j=1,2
     call DTRSV('l','n','u',Nx+3,L1,Nx+3,Y2(:,j),1)
     call DTRSV('u','n','n',Nx+3,U1,Nx+3,Y2(:,j),1)
  end do

  deallocate(U1, L1, Data_init)


  !************************************** Step 3 *****************************************!

  ! Compute matrices L2 and U2, of size (Ny+3)

  allocate (  L2(Ny+3,Ny+3) , U2(Ny+3,Ny+3)  )

  call compute_LU(Ny+3, dy)

  open(unit=100,file='L.dat',status='unknown')
  open(unit=101,file='U.dat',status='unknown')

  do i=1,Ny+3
     do j=1,Ny+3
        read(100,*) L2(i,j)
        read(101,*) U2(i,j)
     end do
  end do

  close(100)
  close(101)

  ! Creation of the related right-hand-side Y3

  allocate ( Y3(Ny+3, Nx+3)  )

  Y3(1,:) = Y2(:,1)
  do i=2,Ny+2
     Y3(i,:) = Y1(:,i-1)
  end do
  Y3(Ny+3,:) = Y2(:,2)

  deallocate(Y1, Y2)

  ! Solves the last (Nx+3) systems of size (Ny+3) to get the eta_i,j

  do j= 1,Nx+3
     call DTRSV('l','n','u',Ny+3,L2,Ny+3,Y3(:,j),1)
     call DTRSV('u','n','n',Ny+3,U2,Ny+3,Y3(:,j),1)
  end do

  deallocate(L2, U2) 

  ! Writes eta_i,j coefficients in file 'eta.dat' (such as coefficients can be read line by line)

  open(unit=102,file='eta.dat',status='unknown')

  do j=1,Nx+3
     do i=1,Ny+3
        write(102,*) Y3(i,j)
     end do
  end do

  close(102)

  deallocate(Y3)

end program part4_program

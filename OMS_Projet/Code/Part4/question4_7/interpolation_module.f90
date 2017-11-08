module interpolation_module

use blas95
use lapack95
include 'mpif.h'

CONTAINS

IMPLICIT NONE

!*************************************************************************************!
!										      !
!		First we define compute LU and other things			      !
!		    we need for the whole interpolation				      !	
!										      !
!*************************************************************************************!



!*************** Subroutine computeLU to compute the eta_i,j later *******************!

SUBROUTINE compute_LU(taille, pas)

integer, intent(in) :: taille
integer :: i,j
double precision, intent(in) :: pas
double precision :: ln, dn
double precision, dimension(taille, taille) :: L, U


!!! Ecriture de L !!!

L = 0.
L(2,1) = -pas/3.
L(3,2) = 1./4.
L(4,3) = 2./7.

do i=1,taille
	L(i,i) = 1
end do


ln = L(4,3)
do i=5,taille-1
	ln = 1./(4.-ln)
	L(i,i-1) = ln
end do

L(taille, taille-2) = -3.*ln/pas
L(taille, taille-1) = ( ln/(4.-ln) )*3./pas

open(unit=100,file='L.dat',status='unknown')

do i=1,taille
  do j=1,taille
    write(100,*) L(i,j)
  end do
end do



!!! Ecriture de U !!!

U = 0.
U(1,1) =-3./pas
U(1,3) = -U(1,1)
U(2,3) = 2.
U(2,2) = 4.
U(3,3) = 7./2.

do i=3,taille-1
	U(i,i+1) = 1.
end do


dn = U(3,3)

do i=4,taille-1
	dn = 4.-1./dn
	U(i,i) = dn
end do

dn = 1.-ln/dn ! does 1 - l(N)/d(N+1)
U(taille, taille) = 3.*dn/pas

open(unit=101,file='U.dat',status='unknown')

do i=1,taille
  do j=1,taille
    write(101,*) U(i,j)
    print*, U(i,j)
  end do
end do

 close(100)
 close(101)

end subroutine compute_LU

!************* Source function to test the global program *************************!

function f(x,y)

double precision, intent(in) :: x,y
double precision :: f
f= x**2+y**2

end function f

!**************** w function containing the w coefficients ************************!

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
	w=0
end if

end function w

!*************************************************************************************!
!										      !
!		Now we're getting serious : here is the whole interpolation	      !
!		Part 1 : Filling Data_unit					      !
!		Part 2 : Using it for computing the eta_i,j			      !	
!										      !
!*************************************************************************************!


SUBROUTINE interpol()

IMPLICIT NONE

integer :: I_proc,J_proc,i,j,k,l,Px,Py,Nx,Ny
integer :: rank,P,code
integer :: up,down,left,right
double precision :: xmin,xmax,ymin,ymax
double precision :: x_I,y_J,dx,dy
double precision, dimension (:,:), allocatable :: F_mat, Data_init
double precision, dimension (:,:), allocatable :: x_pack, y_pack 
double precision, dimension (:,:), allocatable :: up_recv, down_recv, left_recv, right_recv


double precision, dimension(:,:), allocatable :: Y1, Y2, Y3
double precision, dimension(:,:), allocatable :: L1, U1, L2, U2

integer, dimension(MPI_STATUS_SIZE) :: statut


!------------Initialization of parameters-----------!

xmin=0.; xmax=3.; ymin=0.; ymax=5.
Px=3; Py=5; Nx=10; Ny=10

dx = (xmax - xmin)/(Px*Nx)
dy = (ymax - ymin)/(Py*Ny)

allocate (F_mat(Nx+23,Ny+23))

!--------------MPI-related parameters---------------!


 call MPI_INIT(code)

 call MPI_COMM_SIZE(MPI_COMM_WORLD,P,code)
 call MPI_COMM_RANK(MPI_COMM_WORLD,rank,code)

!***************************************************************!
!								!
!		Part 1 : Filling Data_init 			!
!								!
!***************************************************************!



!---Each process gets his (I_proc,J_proc) coordinates)--!
!-------and computes base values on its subdomain------!

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

allocate (x_pack(11,(Ny+1)))
allocate (down_recv(11,(Ny+1)))

do j=1,Ny+1
	do i=1,11
	    x_pack(i,j)=F_mat(i+11+1,j+11)
	end do
end do

!!!

 call MPI_Send(x_pack,11*(Ny+1), MPI_DOUBLE_PRECISION, up, 0, MPI_COMM_WORLD, code)
 call MPI_Recv(down_recv,11*(Ny+1), MPI_DOUBLE_PRECISION, down, 0, MPI_COMM_WORLD, statut, code)

!!!

do i=1,11
	do j=1,Ny+1
		F_mat(i+Nx+1+11,j+11) = down_recv(i,j)
	end do
end do

deallocate (down_recv)

!--------------------------------!

allocate (up_recv(11,(Ny+1)))

do j=1,Ny+1
	do i=1,11
	    x_pack(i,j)=F_mat(i+ Nx, j+11)
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




!-----Send, recv & storing of East and West data------!


allocate (y_pack((Nx+23),11))
allocate (right_recv((Nx+23),11))

do j=1,11
	do i=1,Nx+23
	    y_pack(i,j)=F_mat(i,j+11+1)
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
	    y_pack(i,j)=F_mat(i, j+Ny)
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



!---- I copy the "f" data in the inner part of"Data_init"----!

allocate ( Data_init(Nx+5,Ny+5) )

Data_init =0

do i=1,Nx+3
	do j=1,Ny+3
		Data_init(i+1,j+1) = F_mat(i + 10, j + 10)
	end do
end do


!------ I compute the derivatives with regards to x and directly store them in Data_init----!

do j=1,Ny+3
 	do k=1,10
		Data_init(1,j+1)= 1/dx * ( Data_init(1,j+1) + w(k)*( F_Mat(11+k,j+10) - F_mat(11-k,j+10) )  )
		Data_init(Nx+5,j+1)= 1/dx * ( Data_init(Nx+5,j+1) + w(k)*( F_Mat(Nx+13+k,j+10) - F_mat(Nx+13-k,j+10) )  )
	end do
end do

!------ I compute the derivatives with regards to y and directly store them in Data_init----!

do i=1,Nx+3
	do l=1,10
		Data_init(i+1,1)= 1/dy * ( Data_init(i+1,1) + w(l)*( F_Mat(i+10,11+l) - F_mat(i+10,11-l) ) )
		Data_init(i+1,Ny+5)= 1/dy * ( Data_init(i+1,Ny+5) + w(l)*( F_Mat(i+10,Ny+13+l) - F_mat(i+10,Ny+13-l) ) )
	end do
end do

!-------I compute d2f/dxdy and directly store them in Data_init ---------------------!

do k=1,10
	do l=1,10
			Data_init(1,1)= 1/(dx*dy) * ( Data_init(1,1) + w(l)*w(k)*( F_Mat(11+k,11+l) - F_mat(11-k,11+l) &
			- F_Mat(11+k,11-l) + F_mat(11-k,11-l)  )    )

			Data_init(1,Ny+5)= 1/(dx*dy) * ( Data_init(1,Ny+5) + w(l)*w(k)*( F_Mat(11+k,Ny+13+l) - F_mat(11-k,Ny+13+l) &
			- F_Mat(11+k,Ny+13-l) + F_mat(11-k,Ny+13-l)  )   )

			Data_init(Nx+5,1)= 1/(dx*dy) * ( Data_init(Nx+5,1) + w(l)*w(k)*( F_Mat(Nx+13+k,11+l) - F_mat(Nx+13-k,11+l) &
			- F_Mat(Nx+13+k,11-l) + F_mat(Nx+13-k,11-l)  )   )

			Data_init(Nx+5,Ny+5)= 1/(dx*dy) * ( Data_init(Nx+5,Ny+5) + w(l)*w(k)*( F_Mat(Nx+13+k,Ny+13+l) - F_mat(Nx+13-k,Ny+13+l) &
			- F_Mat(Nx+13+k,Ny+13-l) + F_mat(Nx+13-k,Ny+13-l)  )   )
	end do
end do


!************************** All we still need is Data_init, we don't need F_mat anymore ****************!

deallocate(F_mat)

!*******************************************************************************************************!
!													!
!	Part 2 : Using Data_init to compute the eta_i,j		 					!
!													!
!													!
!*******************************************************************************************************!



!******************************* Solving linear systems ********************************!
!											!
! Step 1 : (Ny + 3) linear systems of size (Nx+5) with right-hand-side as columns of Y1 !
! Step 2 :  2 linear systems of size (Nx+5) with right-hand-side as columns of Y2       !
! Step 3 : (Nx + 5) linear systems of size (Ny+5) with right-hand-side as columns of Y3 !
! 											!
!***************************************************************************************! 


!************************************** Step 1 *****************************************!


! Creation of matrices L1 and U1 !

allocate (  L1(Nx+5,Nx+5) , U1(Nx+5,Nx+5)  )

 call compute_LU(Nx+5, dx)

open(unit=100,file='L.dat',status='unknown')
open(unit=101,file='U.dat',status='unknown')

do i=1,Nx+5
  do j=1,Nx+5
    read(100,*) L1(i,j)
    read(101,*) U1(i,j)
  end do
end do

 close(100)
 close(101)


! Creation of the related right-hand-side Y1, reading "line by line" Data_init !

allocate ( Y1(Nx+5,Ny+3) )

do i=1,Nx+5
Y1(i,:)=Data_init(i,2:Ny+4)
end do

! We can solve the systems of step 1 !

do j = 1,Ny+3
 call DTRSV('l','n','u',Nx+5,L1,Nx+5,Y1(:,j),1)
 call DTRSV('u','n','n',Nx+5,U1,Nx+5,Y1(:,j),1)
end do


!************************************** Step 2 *****************************************!

! Creation of the related right-hand-side Y2 !

allocate(Y2(Nx+5,2) )

Y2(:,1) = Data_init(:,1)
Y2(:,2) = Data_init(:,Ny+5)

deallocate(Data_init)

! We can solve the 2 systems of size (Nx+5) !

do j=1,2
 call DTRSV('l','n','u',Nx+5,L1,Nx+5,Y2(:,j),1)
 call DTRSV('u','n','n',Nx+5,U1,Nx+5,Y2(:,j),1)
end do

deallocate(U1, L1)

!************************************** Step 3 *****************************************!

! Compute matrices L2 and U2, of size (Ny+5) !

allocate (  L2(Ny+5,Ny+5) , U2(Ny+5,Ny+5)  )

 call compute_LU(Ny+5, dy)

open(unit=100,file='L.dat',status='unknown')
open(unit=101,file='U.dat',status='unknown')

do i=1,Ny+5
  do j=1,Ny+5
    read(100,*) L2(i,j)
    read(101,*) U2(i,j)
  end do
end do

 close(100)
 close(101)

! Creation of the related right-hand-side Y3 !

allocate ( Y3(Ny+5, Nx+5)  )

Y3(1,:)=Y2(:,1)
do i=2,Ny+4
Y3(i,:)=Y1(:,i-1)
end do
Y3(Ny+5,:)=Y2(:,2)

deallocate(Y1, Y2)

! Solving the last (Nx+5) systems of size (Ny+5) to get the eta_i,j !

do j= 1,Nx+5
 call DTRSV('l','n','u',Ny+5,L2,Ny+5,Y3(:,j),1)
 call DTRSV('u','n','n',Ny+5,U2,Ny+5,Y3(:,j),1)
end do

deallocate(L2, U2) 

! Writing eta_i,j coefficients in file eta.dat (such as coefficients can be read line by line) !
! The work is done, that's all we needed to define the 2D-spline on the required domain	       !

open(unit=102,file='eta.dat',status='unknown')

do j=1,Nx+5
  do i=1,Ny+5
    write(102,*) Y3(i,j)
  end do
end do

 close(102)

deallocate(Y3)

!-----Woooo it's over \o/ --------!

 call MPI_FINALIZE(code)

end SUBROUTINE interpol

end module interpolation_module

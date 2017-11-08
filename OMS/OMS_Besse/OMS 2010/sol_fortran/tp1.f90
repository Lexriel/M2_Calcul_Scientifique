program tp1
  use fonctions
  implicit none

  real (kind=8) :: y0,eps,result
  real (kind=8), dimension(3) :: y0vec,resultvec
  real (kind=8), dimension(6) :: y0vec2
  integer, parameter :: Nh=100
  integer :: Nh2
  real (kind=8), dimension(Nh) :: t,u
  real (kind=8), allocatable, dimension(:) :: tvec
  real (kind=8), allocatable, dimension(:,:) :: uvec
  real (kind=8), dimension(2) :: tspan
  integer :: itermax,j

  y0=-2.d0
  result=Newton(f,df,y0,1.d-10,50)
  write(6,*) 'Le resultat de la methode de Newton est ',result

  y0=-2.d0
  result=Secant(f,y0,1.d-10,50)
  write(6,*) 'Le resultat de la methode de la secante est ',result

  y0vec=(/0.1d0,0.1d0,-0.1d0/)
  resultvec=NewtonV(fV,dfV,y0vec,1.d-10,50)
  write(6,*) 'Le resultat de Newton vectoriel est ',resultvec

  y0vec=(/0.1d0,0.1d0,-0.1d0/)
  resultvec=Broyden(fV,y0vec,1.d-10,50)
  write(6,*) 'Le resultat de Broyden est ',resultvec

  y0=0.d0
  tspan=(/0.d0,1.d0/)
  call feuler(odef,tspan,y0,Nh,t,u)
  write(6,*) 'Solution finale par Euler explicite', u(Nh)

  y0=0.d0
  tspan=(/0.d0,1.d0/)
  call beuler(odef,tspan,y0,Nh,t,u)
  write(6,*) 'Solution finale par Euler implicite', u(Nh)

  tspan=(/0.d0,25.d0/)
  y0vec2=(/0.d0,1.d0,0.d0,0.8d0,0.d0,1.2d0/)
  Nh2=100000
  allocate(tvec(Nh2),uvec(6,Nh2))
  call beulerV(fun_ex6 ,tspan ,y0vec2 ,Nh2 ,tvec, uvec )
  open (100,file='output.dat',status='unknown')
  do j=1,Nh2
     write(100,*) uvec(1:3,j)
  end do
  close(100)
  deallocate(tvec, uvec)

end program tp1

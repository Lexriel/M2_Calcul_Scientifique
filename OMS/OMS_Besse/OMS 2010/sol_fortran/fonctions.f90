module fonctions
contains
  function Newton(f,df,y0,eps,itermax) result(y)
    ! Newton function to solve nonlinear scalar equation
    ! f(y)=0
    ! f : function
    ! df : derivative
    ! y0 : initial guess
    ! eps : precision
    ! itermax : maximum number of iterations
    implicit none
    interface
       real(kind=8) function f(x)
         real(kind=8), intent(in) :: x
       end function f

       real(kind=8) function df(x)
         real(kind=8), intent(in) :: x
       end function df
    end interface

    real (kind=8), intent(in) :: y0,eps
    integer, intent(in) :: itermax
    
    real (kind=8) :: y,yk,ykp
    integer :: cpt
    logical :: test
    
    yk=y0
    cpt=1
    test=.true.
    
    do while (test)
       ykp=yk-f(yk)/df(yk)
       cpt=cpt+1
       yk=ykp
       test=(abs(f(yk))>eps).AND.(cpt<itermax)
    end do

    y=yk
  end function Newton
  

  function Secant(fun_f,y0,eps,itermax) result(y)
    ! Secant function to solve nonlinear scalar equation
    ! f(y)=0
    ! f : function
    ! df : derivative
    ! y0 : initial guess
    ! eps : precision
    ! itermax : maximum number of iterations
    implicit none
    interface
       real(kind=8) function fun_f(x)
         real(kind=8), intent(in) :: x
       end function fun_f
    end interface

    real (kind=8), intent(in) :: y0,eps
    integer, intent(in) :: itermax
    
    real (kind=8) :: y,yk,ykp,a,h
    integer :: cpt
    logical :: test


    yk=y0
    cpt=1
    test=.true.
    h=1.d-10
    a=(fun_f(y0+h)-fun_f(y0-h))/2.d0/h


    do while (test)
       ykp=yk-fun_f(ykp)/a
       a=(fun_f(ykp)-fun_f(yk))/(ykp-yk)
       cpt=cpt+1
       test=(abs(fun_f(ykp))>eps).AND.(cpt<itermax)
       yk=ykp
    end do
    
    y=ykp
    
  end function Secant

  real(kind=8) function f(x)
    ! function used to test Newton and Secant method
    implicit none
    real(kind=8), intent(in) :: x
    f=tanh(x)*cos(x**2.)+x+2.
  end function f
  
  real(kind=8) function df(x)
    ! derivative function used to test Newton and Secant method
    implicit none
    real(kind=8), intent(in) :: x
    df=(1.-tanh(x)**2.)*cos(x**2.)-2.*tanh(x)*sin(x**2.)*x+1.
  end function df
  

  function NewtonV(f,df,y0,eps,itermax) result(y)
    ! Newton function to solve nonlinear system of equations
    ! f(y)=0
    ! f : function
    ! df : jacobian matrix function
    ! y0 : initial guess
    ! eps : precision
    ! itermax : maximum number of iterations
    use lapack95
    use blas95
    implicit none
    interface
       function f(x,N) result(y)
         integer :: N
         real(kind=8), intent(in), dimension(N) :: x
         real(kind=8), dimension(N) :: y
       end function f

       function df(x,N) result(y)
         integer :: N
         real(kind=8), intent(in), dimension(N) :: x
         real(kind=8) , dimension(N,N) :: y
       end function df
    end interface

    real (kind=8), intent(in), dimension(:) :: y0
    real (kind=8), intent(in) :: eps
    integer, intent(in) :: itermax
    
    real (kind=8), dimension(size(y0)) :: y

    integer, dimension(:), allocatable :: ipiv
    real (kind=8), dimension(:), allocatable :: yk,ykp,res
    real (kind=8), dimension(:,:), allocatable :: A

    integer :: cpt
    logical :: test
    integer :: N

    N=size(y0)

    allocate(yk(N),ykp(N),res(N))
    allocate(A(N,N),ipiv(N))
    yk=y0
    cpt=1
    test=.true.
    
    do while (test)
       res=f(yk,N)
       a=df(yk,N)
       call getrf(a,ipiv)
       call getrs(a,ipiv,res)
       !res=df(yk)\f(yk)

       ykp=yk-res
       cpt=cpt+1
       yk=ykp
       test=(nrm2(f(yk,N))>eps).AND.(cpt<itermax)
    end do

    y=yk
    deallocate(yk,ykp,res,A,ipiv)
  end function NewtonV


function Broyden(f,y0,eps,itermax) result(y)
    ! Broyden function to solve nonlinear system of equations
    ! f(y)=0
    ! f : function
    ! y0 : initial guess
    ! eps : precision
    ! itermax : maximum number of iterations
    use lapack95
    use blas95
    implicit none
    interface
       function f(x,N) result(y)
         integer :: N
         real(kind=8), intent(in), dimension(N) :: x
         real(kind=8), dimension(N) :: y
       end function f
    end interface

    real (kind=8), intent(in), dimension(:) :: y0
    real (kind=8), intent(in) :: eps
    integer, intent(in) :: itermax
    
    real (kind=8), dimension(size(y0)) :: y

    integer, dimension(:), allocatable :: ipiv
    real (kind=8), dimension(:), allocatable :: yk,ykp,res,y0p,y0m,u,s
    real (kind=8), dimension(:,:), allocatable :: Ak,Atmp
    real (kind=8) :: h
    integer :: cpt,j,N
    logical :: test

    N=size(y0)

    allocate(yk(N),ykp(N),res(N),y0p(N),y0m(N),u(N),s(N))
    allocate(Ak(N,N),Atmp(N,N),ipiv(N))

    h=1.d-5

    do j=1,N
       y0p=y0 
       y0p(j)=y0p(j)+h
       y0m=y0 
       y0m(j)=y0m(j)-h
    
       Ak(:,j)=f(y0p,N)-f(y0m,N)
    end do
    Ak=Ak/2.d0/h

    yk=y0
    cpt=1
    test=.true.
    
    do while (test)
       res=f(yk,N)
       Atmp=Ak
       call getrf(Atmp,ipiv)
       call getrs(Atmp,ipiv,res)
       !    res=Ak\f(yk)
       ykp=yk-res
       u=f(ykp,N)-f(yk,N)
       s=ykp-yk
       call gemv(Ak,s,u,-1.d0,1.d0) ! u=-Ak*s+u
       Atmp=myprod(u,s,N)
       Atmp=Atmp/nrm2(s)**2.d0
       Ak=Ak+Atmp
       yk=ykp
       cpt=cpt+1
       test=(nrm2(f(yk,N))>eps).AND.(cpt<itermax)
    end do
  
    y=ykp
    deallocate(yk,ykp,y0p,y0m,res,Ak,Atmp,ipiv,u,s)
    contains
      function myprod(v1,v2,N) result(mat)
        implicit none
        integer :: N
        real (kind=8), dimension(N), intent(in) :: v1,v2
        real (kind=8), dimension(N,N) :: mat
        integer :: j,k
        do k=1,N
           do j=1,N
              mat(j,k)=v1(j)*v2(k)
           end do
        end do
      end function myprod
  end function Broyden

  function fV(x,N) result(y)
    ! function used to test Newton and Broyden method
    integer :: N
    real(kind=8), intent(in), dimension(N) :: x
    real(kind=8), dimension(N) :: y
    real(kind=8) :: pi
    
    pi=4.d0*atan(1.d0)
    y(1)=3.*x(1)-cos(x(2)*x(3))-0.5
    y(2)=x(1)**2.-81.*(x(2)+0.1)**2.+sin(x(3))+1.06
    y(3)=exp(-x(1)*x(2))+20*x(3)+(10*pi-3.)/3.
    
  end function fV
  
  function dfV(x,N) result(y)
    ! Jacobian function used to test Newton method
    integer :: N
    real(kind=8), intent(in), dimension(N) :: x
    real(kind=8) , dimension(N,N) :: y
    y=reshape((/3.d0,2.*x(1),-x(2)*exp(-x(1)*x(2)),&
         &x(3)*sin(x(2)*x(3)),-2.*81.*(x(2)+0.1) ,-x(1)*exp(-x(1)*x(2)),&
         & x(2)*sin(x(2)*x(3)),cos(x(3)),20.d0/),(/N,N/))
  end function dfV


  subroutine feuler(odefun ,tspan ,y0 ,Nh ,t ,u )
    ! FEULER Solves differential equations using the forward
    ! Euler method.
    ! [T,Y]= FEULER(ODEFUN ,TSPAN ,Y0 ,NH) with TSPAN=[T0 ,TF]
    ! integrates the system of differential equations ! y?=f(t,y) from time T0
    ! to TF with initial condition 
    ! Y0 using the forward Euler method on an equispaced 
    ! grid of NH intervals . 
    ! Function ODEFUN(T,Y) must return a vector , whose 
    ! elements hold the evaluation of f(t,y), of the 
    ! same dimension of Y. 
    ! Each row in the solution array Y corresponds to a 
    ! time returned in the column vector T. 
    ! [T,Y] = FEULER(ODEFUN ,TSPAN ,Y0 ,NH ,P1 ,P2 ,...) passes 
    ! the additional parameters P1 ,P2 ,... to the function 
    ! ODEFUN as ODEFUN(T,Y,P1 ,P2 ...).
    implicit none
    interface
       real(kind=8) function odefun(s,y)
         real(kind=8), intent(in) :: s,y
       end function odefun
    end interface
    real(kind=8), intent(in), dimension(2) :: tspan
    real(kind=8), intent(in) :: y0
    integer, intent(in) :: Nh
    real(kind=8), intent(out), dimension(Nh) :: t,u
    
    integer :: j
    real(kind=8) :: dt
    
    dt=(tspan(2)-tspan(1))/(Nh-1.d0)
    do j=1,Nh
       t(j)=tspan(1)+(j-1.d0)*dt
    end do
    
    u(1)=y0
    do j=2,Nh
       u(j)=u(j-1)+dt*odefun(t(j-1),u(j-1))
    end do
  end subroutine feuler

  subroutine beuler(odefun ,tspan ,y0 ,Nh ,t, u )
    ! BEULER Solves differential equations using the backward
    ! Euler method.
    ! [T,Y]= BEULER(ODEFUN ,TSPAN ,Y0 ,NH) with TSPAN=[T0 ,TF]
    ! integrates the system of differential equations ! y?=f(t,y) from time T0
    ! to TF with initial condition 
    ! Y0 using the borward Euler method on an equispaced 
    ! grid of NH intervals . 
    ! Function ODEFUN(T,Y) must return a vector , whose 
    ! elements hold the evaluation of f(t,y), of the 
    ! same dimension of Y. 
    ! Each row in the solution array Y corresponds to a 
    ! time returned in the column vector T. 
    ! [T,Y] = BEULER(ODEFUN ,TSPAN ,Y0 ,NH ,P1 ,P2 ,...) passes 
    ! the additional parameters P1 ,P2 ,... to the function 
    ! ODEFUN as ODEFUN(T,Y,P1 ,P2 ...).
    
    implicit none
    interface
       real(kind=8) function odefun(s,y)
         real(kind=8), intent(in) :: s,y
       end function odefun
    end interface
    real(kind=8), intent(in), dimension(2) :: tspan
    real(kind=8), intent(in) :: y0
    integer, intent(in) :: Nh
    real(kind=8), intent(out), dimension(Nh) :: t,u
    
    integer :: j
    real(kind=8) :: dt
    
    dt=(tspan(2)-tspan(1))/(Nh-1.d0)
    do j=1,Nh
       t(j)=tspan(1)+(j-1.d0)*dt
    end do
    
    u(1)=y0
    do j=2,Nh
       u(j)=Secant(f,u(j-1),1.d-8,50);
    end do
    
  contains
    function f(x) result(fx)
      implicit none
      real(kind=8), intent(in) :: x
      real(kind=8) :: fx
      fx=x-u(j-1)-dt*odefun(t(j),x)
    end function f
  end subroutine beuler

  function odef(t,y) result(ode)
    implicit none
    real(kind=8), intent(in) :: t,y
    real(kind=8) :: ode
    ode=cos(2.d0*y)
  end function odef



  subroutine feulerV(odefun ,tspan ,y0 ,Nh ,t ,u )
    implicit none
    interface
       real(kind=8) function odefun(s,y)
         real (kind=8), intent(in) :: s
         real (kind=8), intent(in), dimension(:) :: y
       end function odefun
    end interface
    real(kind=8), intent(in), dimension(2) :: tspan
    real(kind=8), intent(in), dimension(:) :: y0
    integer, intent(in) :: Nh
    real(kind=8), intent(out), dimension(Nh) :: t
    real(kind=8), intent(out), dimension(:,:) :: u
    
    integer :: j
    real(kind=8) :: dt
    
    dt=(tspan(2)-tspan(1))/(Nh-1.d0)
    do j=1,Nh
       t(j)=tspan(1)+(j-1.d0)*dt
    end do
    
    u(:,1)=y0
    do j=2,Nh
       u(:,j)=u(:,j-1)+dt*odefun(t(j-1),u(:,j-1))
    end do
  end subroutine feulerV

  subroutine beulerV(odefun ,tspan ,y0 ,Nh ,t, u )
    implicit none
    interface
       function odefun(s,y,dim) result(z)
         integer,intent(in) :: dim
         real (kind=8), intent(in) :: s
         real (kind=8), intent(in), dimension(dim) :: y
         real(kind=8), dimension(dim) :: z
       end function odefun
    end interface
    real(kind=8), intent(in), dimension(2) :: tspan
    real(kind=8), intent(in), dimension(:) :: y0
    integer, intent(in) :: Nh
    real(kind=8), intent(out), dimension(Nh) :: t
    real(kind=8), intent(out), dimension(:,:) :: u
    
    integer :: j
    real(kind=8) :: dt
    
    dt=(tspan(2)-tspan(1))/(Nh-1.d0)
    do j=1,Nh
       t(j)=tspan(1)+(j-1.d0)*dt
    end do
    
    u(:,1)=y0
    do j=2,Nh
       u(:,j)=Broyden(f,u(:,j-1),1.d-8,50)
    end do
    
  contains
    function f(x,dim) result(fx)
      implicit none
      integer :: dim
      real(kind=8), intent(in), dimension(dim) :: x
      real(kind=8), dimension(dim) :: fx
      fx=x-u(:,j-1)-dt*odefun(t(j),x,dim)
    end function f
  end subroutine beulerV

  function fun_ex6(t,XV,dim) result(V)
    use blas95
    implicit none
    integer, intent(in) :: dim
    real (kind=8), intent(in) :: t
    real (kind=8), intent(in), dimension(dim) :: XV
    real (kind=8), dimension(dim) :: V
    real (kind=8) :: x,y,z,vx,vy,vz,g,m,tmp1,tmp2,lambda
    real (kind=8), dimension(3) :: F,W,dphi

    x=XV(1);   y=XV(2);  z=XV(3);
    vx=XV(4); vy=XV(5); vz=XV(6);

    V(1)=vx; V(2)=vy; V(3)=vz;
    g=9.8d0
    m=1.d0
    F=(/0.d0,0.d0,-g*m/)
    W=(/vx,vy,vz/)
    dphi=(/2*x,2*y,2*z/)
    tmp1=dot(m*W,2.d0*W)
    tmp2=dot(dphi,F)
!  lambda=(m*W'*(H*W)+dphi'*F)/(nrm2(dphi)**2.d0)
    lambda=(tmp1+tmp2)/(nrm2(dphi)**2.d0)
  
    V(4)=(F(1)-lambda*dphi(1))/m
    V(5)=(F(2)-lambda*dphi(2))/m
    V(6)=(F(3)-lambda*dphi(3))/m
  end function fun_ex6

end module fonctions

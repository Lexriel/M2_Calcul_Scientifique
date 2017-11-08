%------------------------------------------------------
% This program solves the one-dimensionnal Helmholtz  -
% problem with general Robin boundary conditions      -
%------------------------------------------------------ 
%Uses GaussLobatto, PhysSpectGL, SpectPhysGL, DerSpect-
%------------------------------------------------------
clear all
format long e
%-----------------------
% Cut-off frequency n
%-----------------------
n=25;
%------------------------------------------------
% Choose the Helmholtz coefficient
%------------------------------------------------
H=1;
%-------------------------------------------------
% Choose the Robin Boundary condition coefficients
%-------------------------------------------------
alphap=0;
alpham=0;
betap=1;
betam=1;
%-------------------------------------------------------------------
% Preliminary computations
%-------------------------------------------------------------------
% Computation of the Gauss-Lobatto collocation points (ordered from
% -1 to 1
x=GaussLobatto(n);
% Computation of the first space derivative in the physical space
% Its name is derivaphys. It is the product of :
% the spectral/physics transform operator
% the derivation operator in the spectral space
% the physics/spectral transform operator
PhyspecGL=zeros(n+1);
SpecphyGL=zeros(n+1);
PhyspecGL=PhysSpectGL(n,PhyspecGL);
SpecphyGL=SpectPhysGL(n,SpecphyGL);
derivaspec=DerSpect(n);
derivaphys=SpecphyGL*derivaspec*PhyspecGL;
% Computation of the second derivative in the physical space
% Its name is derivaphys2, square of derivaphys
derivaphys2=derivaphys*derivaphys;
%------------------------------------------------------------------
% Definition of the test case¨
%------------------------------------------------------------------
k=1;
phi=0;
ue=cos(2*pi*k*x+phi);
derue=-2*pi*k*sin(2*pi*k*x+phi);
%------------------------------------------------------------------
% design of the complete Laplace Operator and source term
%------------------------------------------------------------------
% Second derivative operator in the bulk
Laplace=derivaphys2;
% Imposition of the Helmholtz coefficent
for i=1:n+1
    Laplace(i,i)=Laplace(i,i)-H*H;
end
% Imposition of the Robin Boundary conditions
Laplace(1,1:n+1)=betam*derivaphys(1,1:n+1);
Laplace(1,1)=Laplace(1,1)+alpham;
Laplace(n+1,1:n+1)=betap*derivaphys(n+1,1:n+1);
Laplace(n+1,n+1)=Laplace(n+1,n+1)+alphap;
%
%------------------------------------------------------------------
% source term definition
%------------------------------------------------------------------
sm=-4*pi^2*k^2*ue-H*H*ue; 
%no source in the bulk
sm(1)=alpham*ue(1)+betam*derue(1);
sm(n+1)=alphap*ue(n+1)+betap*derue(n+1);
%------------------------------------------------------------------
% Solving directly the linear problem
%------------------------------------------------------------------
u=inv(Laplace)*sm'
plot(x,u)
xlabel('x','Fontsize',16)
title('\it{Solution of the Helmholtz problem}','Fontsize',16)
%------------------------------------------------------------------
% Computation using the diagonalization of the internal operator
%------------------------------------------------------------------
% Computation of the interior operator and right hand side of the problem
P=[Laplace(1,1) Laplace(1,n+1);Laplace(n+1,1) Laplace(n+1,n+1)];
Lapint=Laplace(2:n ,2:n)- [Laplace(2:n,1) Laplace(2:n,n+1)]*inv(P)*[Laplace(1,2:n);Laplace(n+1,2:n)];
smint=sm(2:n)'-[Laplace(2:n,1) Laplace(2:n,n+1)]*inv(P)*[sm(1);sm(n+1)];
% Filtering of spurious modes : eigenvalues are all real and negative.
[vp,lambda]=eig(Lapint);
for i=1:n-1
    if(abs(lambda(i,i)) < 1.e-8 ) %Mesh-dependent criterion !
        lambda(i,i)=0;
    else 
        lambda(i,i)=1./lambda(i,i);
    end
end
% Solving the problem
uint=vp*lambda*inv(vp)*smint;
% Computing the boundary values
ucl=inv(P)*(-[Laplace(1,2:n);Laplace(n+1,2:n)]*uint+[sm(1);sm(n+1)]);
% Concatenation of the solution
udiag=[ucl(1); uint ;ucl(2)];
figure(2)
plot(x,udiag)
%------------------------------------------------------------------
% Error analysis
%------------------------------------------------------------------
%Plot of the error
 error=abs(u-ue');
 errordiag=abs(udiag-ue');
 figure(3)
 plot(x,error,'x',x,errordiag,'-x')
 xlabel('x','Fontsize',16)
 title('\it{Error of the resolution}','Fontsize',16)
 % Plot of the error Chebyshev spectrum
 specerror=abs(PhyspecGL*error);
 specerrordiag=abs(PhyspecGL*errordiag);
 figure(4)
 plot(x,specerror,'x',x,specerrordiag,'-x')
 xlabel('x','Fontsize',16)
 title('\it{Spectra of the errors}','Fontsize',16)
 


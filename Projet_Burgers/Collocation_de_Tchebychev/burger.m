%------------------------------------------------------
% This program solves the one-dimensionnal Burger     -
% problem with general Robin boundary conditions      -
%------------------------------------------------------ 
%Uses GaussLobatto, PhysSpectGL, SpectPhysGL, DerSpect-
%------------------------------------------------------
clear all
format long e
%-----------------------
% Pas de temps
%-----------------------
dt=0.001;
max=0;
%-----------------------
% Cut-off frequency n
%-----------------------
n=200;
%------------------------------------------------
% Choose the viscosity coefficient
%------------------------------------------------
nu=1/100/pi;
%nu=1;
%-------------------------------------------------
% Choose the Robin Boundary condition coefficients
%-------------------------------------------------
alphap=1;
alpham=1;
betap=0;
betam=0;
%-------------------------------------------------------------------
% Preliminary computations
%-------------------------------------------------------------------
% Computation of the Gauss-Lobatto collocation points (ordered from
% -1 to 1
x=GaussLobatto(n)';
% Computation of the first space derivative in the physical space
% Its name is derivaphys. It is the product of :
% the spectral/physics transform operator
% the derivation operator in the spectral space
% the physics/spectral transform operator
PhyspecGL=zeros(n+1);
SpecphyGL=zeros(n+1);
PhyspecGL=PhysSpectGL(n,PhyspecGL);
SpecphyGL=SpectPhysGL(n,SpecphyGL); 
derivaspec=Derspect(n); %Matrice de d�rivation dans l'espace spectral
derivaphys=SpecphyGL*derivaspec*PhyspecGL; % Matrice de d�rivation dans l'espace physique
% Computation of the second derivative in the physical space
% Its name is derivaphys2, square of derivaphys
derivaphys2=derivaphys*derivaphys;
%------------------------------------------------------------------
% First iteration
%-----------------------------------------------------------------
sol_init=sin(pi*(x+1)/2);
%plot(x,sol_init)
%xlabel('x','Fontsize',16)
%title('\it{Solution of the Helmholtz problem}','Fontsize',16)
%plot(x,sol_init)
%hold on
b=-dt*2*(derivaphys*sol_init).*sol_init+sol_init;
%b=-dt*2*derivaphys*sol_init+sol_init;
A=eye(n+1)-dt*4*nu*derivaphys2;
% Imposition of the Robin Boundary conditions
A(1,1:n+1)=betam*derivaphys(1,1:n+1);
A(1,1)=A(1,1)+alpham;
A(n+1,1:n+1)=betap*derivaphys(n+1,1:n+1);
A(n+1,n+1)=A(n+1,n+1)+alphap;
b(1)=0;
b(n+1)=0;
%u=A\b';
u=A\b;

%------------------------------------------------------------------
% Loop
%-----------------------------------------------------------------
for k=1:1000
    b=-dt*2*(derivaphys*u).*u+u;
    %b=-dt*2*derivaphys*u+u;
    %u=A\b';
    u=A\b;
    if mod(1000,k)==0
        %plot(x,u)
        %hold on
    end
    
    derive(k) = 2*(u(n+1) - u(n))/ ( 1 + cos (pi*(n-1)/n));
    if ( derive(k) < max )
        max = derive(k);
        temps_max= k*dt;
    end
    
end

vecteur_temps=linspace(-1,1,1000);
vecteur_temps=(vecteur_temps + 1)/2;

plot (vecteur_temps,derive);

max
temps_max


%---------------------------------------------------------------------
% Tests 
%---------------------------------------------------------------------
% plot(x,derivaphys*sol_init)
% hold on 
% plot(x,pi/2*cos(pi*(x+1)/2))


% %------------------------------------------------------------------
% % design of the complete Laplace Operator and source term
% %------------------------------------------------------------------
% % Second derivative operator in the bulk
% Laplace=derivaphys2;
% % Imposition of the Helmholtz coefficent
% for i=1:n+1
%     Laplace(i,i)=Laplace(i,i)-H*H;
% end
% % Imposition of the Robin Boundary conditions
% Laplace(1,1:n+1)=betam*derivaphys(1,1:n+1);
% Laplace(1,1)=Laplace(1,1)+alpham;
% Laplace(n+1,1:n+1)=betap*derivaphys(n+1,1:n+1);
% Laplace(n+1,n+1)=Laplace(n+1,n+1)+alphap;
% %
% %------------------------------------------------------------------
% % source term definition
% %------------------------------------------------------------------
% sm=-4*pi^2*k^2*ue-H*H*ue; 
% %no source in the bulk
% sm(1)=alpham*ue(1)+betam*derue(1);
% sm(n+1)=alphap*ue(n+1)+betap*derue(n+1);


% %------------------------------------------------------------------
% % Computation using the diagonalization of the internal operator
% %------------------------------------------------------------------
% % Computation of the interior operator and right hand side of the problem
% P=[Laplace(1,1) Laplace(1,n+1);Laplace(n+1,1) Laplace(n+1,n+1)];
% Lapint=Laplace(2:n ,2:n)- [Laplace(2:n,1) Laplace(2:n,n+1)]*inv(P)*[Laplace(1,2:n);Laplace(n+1,2:n)];
% smint=sm(2:n)'-[Laplace(2:n,1) Laplace(2:n,n+1)]*inv(P)*[sm(1);sm(n+1)];
% % Filtering of spurious modes : eigenvalues are all real and negative.
% [vp,lambda]=eig(Lapint);
% for i=1:n-1
%     if(abs(lambda(i,i)) < 1.e-8 ) %Mesh-dependent criterion !
%         lambda(i,i)=0;
%     else 
%         lambda(i,i)=1./lambda(i,i);
%     end
% end
% % Solving the problem
% uint=vp*lambda*inv(vp)*smint;
% % Computing the boundary values
% ucl=inv(P)*(-[Laplace(1,2:n);Laplace(n+1,2:n)]*uint+[sm(1);sm(n+1)]);
% % Concatenation of the solution
% udiag=[ucl(1); uint ;ucl(2)];
% figure(2)
% plot(x,udiag)
% %------------------------------------------------------------------
% % Error analysis
% %------------------------------------------------------------------
% %Plot of the error
%  error=abs(u-ue');
%  errordiag=abs(udiag-ue');
%  figure(3)
%  plot(x,error,'x',x,errordiag,'-x')
%  xlabel('x','Fontsize',16)
%  title('\it{Error of the resolution}','Fontsize',16)
%  % Plot of the error Chebyshev spectrum
%  specerror=abs(PhyspecGL*error);
%  specerrordiag=abs(PhyspecGL*errordiag);
%  figure(4)
%  plot(x,specerror,'x',x,specerrordiag,'-x')
%  xlabel('x','Fontsize',16)
%  title('\it{Spectra of the errors}','Fontsize',16)
 


%Solution to the Poisson equation with inhomogeneous Dirichlet BC

N=50; %Number of nodes
x=linspace(-3,3,N)'; %Discretization in x
y=x; %Discretization in y
[X,Y]=meshgrid(x,y); %Generation of meshgrid
R=sqrt(X.^2+Y.^2);
e=ones(N^2,1);
dx=x(2)-x(1);
CL=20; % Value of the BC

%Matrix of the linear system
A=spdiags([-e,-e,4*e,-e,-e],[-N,-1,0,1,N],N*N,N*N)/dx^2; 
A(N:N:N*N,N+1:N:N*N)=0; 
A(N+1:N:N*N,N:N:N*N)=0;

%Right hand side
%Source term
S=20*(R-1.5).^2.*(R-2).^2.*(R>1.5).*(R<2)-0.5*R.^2.*(R-1).^2.*(R<1);
q=reshape(S,N*N,1);
%Add the BC to the r.h.s
q(1:1:N)=q(1:1:N)+CL/dx^2; %Lower part of the domain
q(N:N:N*N)=q(N:N:N*N)+CL/dx^2; %Right part of the domain
q(1:N:N*N-N+1)=q(1:N:N*N-N+1)+CL/dx^2; %Left part of the domain
q(N*N-N+1:1:N*N)=q(N*N-N+1:1:N*N)+CL/dx^2; %Higher part of the domain

sol=A\q;

mesh(X,Y,reshape(sol,N,N));
axis tight;

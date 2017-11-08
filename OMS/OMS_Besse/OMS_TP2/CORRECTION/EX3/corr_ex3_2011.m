%Solution to the Poisson equation with inhomogeneous Dirichlet BC
% and non isotropic diffusion coefficient C=C(x,y)
% 
% The 1D equation is
%  - Dx (C Dx u) = f
% This equation can be read also as
%  -Dx C . Dx u -c Dxx u = f
% however a finite difference approximation may lead to a scheme which
% does not preserve the discrete maximum principle.
%
% We therefore prefer the following finite difference approx
%
%             u(x+h)-u(x)             u(x)-u(x-h)
%    c(x+h/2) ----------- - c(x-h/2)  -----------
%                  h                      h
% -  --------------------------------------------- = f(x)
%                         h
%
% We perform the same numerical scheme in 2D
clear all;
N=50; %Number of nodes
x=linspace(-3,3,N)'; %Discretization in x
y=x; %Discretization in y
[X,Y]=meshgrid(x,y); %Generation of meshgrid
R=sqrt(X.^2+Y.^2);
e=ones(N^2,1);
dx=x(2)-x(1);
CL=20; % Value of the BC

fc=@(x,y) 1-2*((x>-1).*(x<1).*(y>-1).*(y<1));

cpt=0;
for k=1:N
  for j=1:N
    cpt=cpt+1;
    cxmh(cpt,1)=fc(x(j)-dx/2,y(k));
    cxph(cpt,1)=fc(x(j)+dx/2,y(k));
    cymh(cpt,1)=fc(x(j),y(k)-dx/2);
    cyph(cpt,1)=fc(x(j),y(k)+dx/2);
  end
end

% $$$ A=spdiags([-cyph,-cxph,cymh+cxmh+cxph+cyph,-cxmh,-cymh],[-N,-1,0,1,N],N* ...
% $$$           N,N*N).'/dx^2;

A=spdiags([-[cyph(N+1:N*N);cyph(1:N)],-[cxph(2:N*N);cxph(1)],cymh+cxmh+cxph+cyph,-[cxmh(N*N);cxmh(1:N*N-1)],-[cymh(N*N-N+1:N*N);cymh(1:N*N-N)]],[-N,-1,0,1,N],N* ...
          N,N*N)/dx^2;


A(N:N:N*N,N+1:N:N*N)=0;
A(N+1:N:N*N,N:N:N*N)=0;

%Right hand side
%Source term
S=20*(R-1.5).^2.*(R-2).^2.*(R>1.5).*(R<2)-0.5*R.^2.*(R-1).^2.*(R<1);
%S=ones(N,N);
%S=zeros(N,N);
q=reshape(S,N*N,1);
%Add the BC to the r.h.s
q(1:1:N)=q(1:1:N)+CL/dx^2; %Lower part of the domain
q(N:N:N*N)=q(N:N:N*N)+CL/dx^2; %Right part of the domain
q(1:N:N*N-N+1)=q(1:N:N*N-N+1)+CL/dx^2; %Left part of the domain
q(N*N-N+1:1:N*N)=q(N*N-N+1:1:N*N)+CL/dx^2; %Higher part of the domain


sol=A\q;
mesh(X,Y,reshape(sol,N,N));
axis tight;
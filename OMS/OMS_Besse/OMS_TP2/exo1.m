function [u] = exo1(N)

x = linspace(-3,3,N+2)';
y = x;

xcalc = x(2:N+1);
ycalc = xcalc;

tspan = [-3,3];

[X,Y] = meshgrid(xcalc,ycalc);
[XD,YD] = meshgrid(x,y);

R = sqrt(X.^2+Y.^2);
h = (tspan(2) - tspan(1))/N;
% A = zeros(N*N,N*N);
% q = zeros(N*N,1);
% u = zeros(N*N,1);


% Définition de la matrice représentant le laplacien
e = ones(N*N,1);
A = spdiags([-e,-e,4*e,-e,-e], [-N,-1,0,1,N], N*N, N*N)/h^2;

% a:b:c signifie de a à c avec un pas de b
% Certains éléments de A sont nuls
    A(N:N:N*N,N+1:N:N*N) = 0;
    A(N+1:N:N*N,N:N:N*N) = 0;
    
% Définition de S
% (remplace habilement le if, elseif, else par l'utilisation de booléens)
    S = (20.*(R-1.5).^2).*(R-2).^2.*(R>1.5).*(R<2) - 0.5.*R.^2.*(R-1).^2.*(R>0).*(R<1);
%    S=zeros(N*N,1);
  
% chaque élément de S est indexé dans un vecteur s
    s = reshape(S, N*N, 1);
q = s;
q(1:N) = q(1:N) + 20/h^2;
q(N:N:N*N) = q(N:N:N*N) + 20/h^2;
q(1:N:(N-1)*N+1) = q(1:N:(N-1)*N+1) + 20/h^2;
q(N^2-N+1:N^2) = q(N^2-N+1:N^2) + 20/h^2;

%q(15*N:N:N*N-15*N) = q(15*N:N:N*N-15*N) + 20/h^2;
% q(30:35) = q(30:35) +20/h^2
u = A\q;

U = reshape(u, N, N);

Utot = 20*ones(N+2,N+2);
Utot(2:N+1,2:N+1) = U;


mesh(X,Y,U)

end
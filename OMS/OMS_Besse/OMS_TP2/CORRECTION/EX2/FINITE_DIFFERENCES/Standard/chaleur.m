% On suppose que le trou fait partie du maillage et on impose une source
% qui est la fonction indicatrice du trou multipliee par la valeur de la
% condition aux limites
% 
clear all;
% parametres de discretisation
[l,L,nx,ny,dx,dy,x,y]=param;

% matrice du laplacien
A=laplacien_trou(nx,ny,dx,dy);

% terme source
S = source_trou(nx,ny,dx,dy);

% conditions aux limites
CL=cl_trou(nx,ny,dx,dy);

% second membre du systeme lineaire
b = S + CL;


T = A\b;

[X,Y]=meshgrid(x,y);
temp=reshape(T,nx,ny);
mesh(X,Y,temp);
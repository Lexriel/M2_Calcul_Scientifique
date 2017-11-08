%-----------------------------------------------------------------
% calcul des parametres de discretisation du domaine de calcul
%-----------------------------------------------------------------

function[l,L,nx,ny,dx,dy,x,y]=param

% dimensions du domaine
% largeur 
l=1;
% longueur
L=1;


% points de discretisation en x
nx=20;
% pas en x 
dx=L/nx;


% points de discretisation en y
ny=20;
% pas en y 
dy=l/ny;                                                
                                                          

% coordonnees des points
x=0+dx/2:dx:L-dx/2;
y=0+dy/2:dy:l-dy/2;                              
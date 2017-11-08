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


T_matlab = A\b;


% calcul de temp(x,y) (transformation vecteur T -> matrice temp)
T=T_matlab;
temp=trans1d2d(T,nx,ny);

figure(3)
contour(x,y,temp',100)
colormap(jet)




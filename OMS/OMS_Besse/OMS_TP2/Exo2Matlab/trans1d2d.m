%------------------------------------------------------------------
% fonction qui transforme un vecteur de longueur imax*jmax en 
% une matrice de taille [1:imax,1:jmax]
%------------------------------------------------------------------

function[T]=trans1d2d(sol,imax,jmax)

T=zeros(imax,jmax);
k=1;
for j=1:jmax
    for i=1:imax
	T(i,j)=sol(k);
	k = k+1;
    end;
end;


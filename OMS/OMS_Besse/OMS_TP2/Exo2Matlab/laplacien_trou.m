%------------------------------------------------------------------
% fonction qui calcule la matrice du laplacien 2D pour la plaque trouee
%------------------------------------------------------------------


function[A]=laplacien_trou(imax,jmax,dx,dy)


% indices du bord du trou
itrou_min = round(imax/3);
itrou_max = 2*round(imax/3);
jtrou_min = round(jmax/3);  
jtrou_max = 2*round(jmax/3);


% remplissage de la matrice
k=1;
A=eye(imax*jmax);

a=+2/dx^2+2/dy^2;
b=-1/dx^2;
c=-1/dx^2;
d=-1/dy^2;
e=-1/dy^2;

for j=1:jmax
    for i=1:imax
      if ( i>itrou_min & i<itrou_max & j>jtrou_min & j<jtrou_max )
	% on ne met rien : ce bloc reste diagonal
      else
	% on remplit la matrice sauf si on est au bord (exterieur ou trou)
	A(k,k)=a;
        if i<imax
         if (i~=itrou_min | j>=jtrou_max | j<=jtrou_min)
           A(k,k+1)=b;
         end;
        end;
        if i>=2 
         if (i~=itrou_max | j>=jtrou_max | j<=jtrou_min)
           A(k,k-1)=c;
         end;
        end;
        if j<=jmax-1
         if (j~=jtrou_min | i>=itrou_max | i<=itrou_min)
           A(k,k+imax)=d;
         end;
        end;
        if j>=2
         if (j~=jtrou_max | i>=itrou_max | i<=itrou_min)
           A(k,k-imax)=e;
         end;
        end;
      end
        k=k+1;
    end;
end;

A=sparse(A);

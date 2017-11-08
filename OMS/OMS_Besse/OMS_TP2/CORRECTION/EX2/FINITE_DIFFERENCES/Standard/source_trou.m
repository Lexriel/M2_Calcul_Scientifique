%------------------------------------------------------------------
% fonction qui calcule le terme source pour la résolution de 
% l'équation de la chaleur stationnaire sur la plaque avec trou
%------------------------------------------------------------------


function[S]=source_trou(imax,jmax,dx,dy)


% indices du bord du trou
itrou_min = round(imax/3);
itrou_max = 2*round(imax/3);
jtrou_min = round(jmax/3);  
jtrou_max = 2*round(jmax/3);

% temperature du bord du trou
Ttrou = 100;


% remplissage du vecteur S
S=zeros(imax*jmax,1); 
k=1;
for j=1:jmax
    for i=1:imax
        if ( i>=itrou_min+1 & i<=itrou_max-1 & j>=jtrou_min+1 & j<=jtrou_max-1 )
	   S(k) = Ttrou;
        end;

        k=k+1;
    end;
end;

S=sparse(S);
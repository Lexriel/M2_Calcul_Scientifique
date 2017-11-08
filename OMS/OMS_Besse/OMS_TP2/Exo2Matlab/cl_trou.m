%------------------------------------------------------------------
% fonction qui calcule les termes dus aux conditions aux limites 
% pour la résolution de l'équation de la chaleur stationnaire sur 
% la plaque avec trou
%------------------------------------------------------------------


function[CL]=cl_trou(imax,jmax,dx,dy)


% indices du bord du trou
itrou_min = round(imax/3);
itrou_max = 2*round(imax/3);
jtrou_min = round(jmax/3);  
jtrou_max = 2*round(jmax/3);

% valeurs de la temperature aux bords exterieurs
Tvd=20;
Tvg=20;
Thh=20;
Thb=20;

% valeur de la temperature aux bords du trou
Ttrou = 100;


% remplissage du vecteur CL
a=+2/dx^2+2/dy^2;
b=-1/dx^2;
c=-1/dx^2;
d=-1/dy^2;
e=-1/dy^2;

CL=zeros(imax*jmax,1); 

% remplissage pour le bord exterieur
k=1;
for j=1:jmax
    for i=1:imax
        if i==imax 
           CL(k) = CL(k)-b*Tvd;
        end;
        if i==1 
           CL(k) = CL(k)-c*Tvg;
        end;
        if j==jmax 
           CL(k) = CL(k)-d*Thh;
        end;
        if j==1 
           CL(k) = CL(k)-e*Thb;
        end;

        k=k+1;
        
    end;
end;

% remplissage pour le bord du trou
k=1;
for j=1:jmax
    for i=1:imax
        if (i==itrou_min & j<jtrou_max & j>jtrou_min)
           CL(k) = -b*Ttrou;
        end;
        if (i==itrou_max & j<jtrou_max & j>jtrou_min)
           CL(k) = -c*Ttrou;
        end;
        if (j==jtrou_min & i<itrou_max & i>itrou_min)
          CL(k) = -d*Ttrou;
        end;
        if (j==jtrou_max & i<itrou_max & i>itrou_min)
           CL(k) = -e*Ttrou;
        end;

        k=k+1;
        
    end;
end;


CL=sparse(CL);













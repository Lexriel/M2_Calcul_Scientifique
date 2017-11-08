% Version assemblage sans connaitre la position des points au N,S,E,W
% Le maillage est calcule par Gmsh regulier en x,y (dx=dy)
% Les noeud des quadrangles peuvent etre ranges dans un ordre quelconque.
%
% Dans cette verion, les points du bords sont considere comme vraiment
% points du bord. Leur prise en compte dans le systeme lineaire est
% complete. Ils font partie du vecteur inconnu. Leur contribution dans la
% matrice du systeme lineaire se fait via l'identite.
%
clear all;
m = load_gmsh('extr.msh');
Zone_Phys_Bord1=1001;
Zone_Phys_Bord2=1002;
A=sparse(m.nbNod,m.nbNod);
rhs=zeros(m.nbNod,1);
CL_ext=20;
CL_int=100;
nodes=zeros(m.nbNod,1);
for j=1:m.nbNod
    nodes(j)=j;
end

for j=1:m.nbQuads
    p1=m.QUADS(j,1);
    p2=m.QUADS(j,2);
    p3=m.QUADS(j,3);
    p4=m.QUADS(j,4);
    
    L1=(m.POS(p4,1)-m.POS(p1,1))^2+(m.POS(p4,2)-m.POS(p1,2))^2;
    L2=(m.POS(p2,1)-m.POS(p1,1))^2+(m.POS(p2,2)-m.POS(p1,2))^2;
    A(p1,p1)=2/L1+2/L2;
    A(p1,p4)=-1/L1;
    A(p1,p2)=-1/L2;
    
    L1=(m.POS(p1,1)-m.POS(p2,1))^2+(m.POS(p1,2)-m.POS(p2,2))^2;
    L2=(m.POS(p3,1)-m.POS(p2,1))^2+(m.POS(p3,2)-m.POS(p2,2))^2;
    A(p2,p2)=2/L1+2/L2;
    A(p2,p1)=-1/L1;
    A(p2,p3)=-1/L2;

    L1=(m.POS(p2,1)-m.POS(p3,1))^2+(m.POS(p2,2)-m.POS(p3,2))^2;
    L2=(m.POS(p4,1)-m.POS(p3,1))^2+(m.POS(p4,2)-m.POS(p3,2))^2;
    A(p3,p3)=2/L1+2/L2;
    A(p3,p2)=-1/L1;
    A(p3,p4)=-1/L2;

    L1=(m.POS(p3,1)-m.POS(p4,1))^2+(m.POS(p3,2)-m.POS(p4,2))^2;
    L2=(m.POS(p1,1)-m.POS(p4,1))^2+(m.POS(p1,2)-m.POS(p4,2))^2;
    A(p4,p4)=2/L1+2/L2;
    A(p4,p3)=-1/L1;
    A(p4,p1)=-1/L2;

    % Calcul du second membre
    rhs(p1)=0;
    rhs(p2)=0;
    rhs(p3)=0;
    rhs(p4)=0;
    
end


noeud_bord=zeros(m.nbLines,2);
vu=zeros(m.nbNod,1);
k=0;
for j=1:m.nbLines
    p1=m.LINES(j,1);
    p2=m.LINES(j,2);
    type=m.LINES(j,3);    
    if not(vu(p1))
        k=k+1;
        noeud_bord(k,1)=p1;
        noeud_bord(k,2)=type;
        vu(p1)=k;
    end
    if not(vu(p2))
        k=k+1;
        noeud_bord(k,1)=p2;
        noeud_bord(k,2)=type;        
        vu(p2)=k;
    end
end

vec_cl=zeros(m.nbNod,1);
for j=1:m.nbLines
    p=noeud_bord(j,1);
    type=noeud_bord(j,2);
    if (type==Zone_Phys_Bord2)
       val_CL=CL_ext;
    else
       val_CL=CL_int;      
    end
    vec_cl(p)=val_CL;
    rhs=rhs-A*vec_cl;
    vec_cl(p)=0;
    A(p,:)=0;A(:,p)=0;
    A(p,p)=1;
    rhs(p)=val_CL;
end

sol=A\rhs;

% Trace version 1
figure(1);
clf;
x=m.POS(nodes,1);
y=m.POS(nodes,2);
plot3(x,y,sol,'.');
view(-30,30);
axis tight;


% Trace version 2
figure(2);
clf;xP=zeros(4,m.nbQuads);
yP=xP;
zP=xP;
for j=1:m.nbQuads
    for k=1:4
        noe=m.QUADS(j,k);
        xP(k,j)=m.POS(noe,1);
        yP(k,j)=m.POS(noe,2);
        zP(k,j)=sol(noe);
    end
end
patch(xP,yP,zP,zP);
view(-30,30);
axis tight;

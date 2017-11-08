clear all;

CL=20;
Zone_Phys_Bord=1001;

m=load_gmsh('ex1.msh');

A=sparse(m.nbNod,m.nbNod);
rhs=zeros(m.nbNod,1);

for l=1:m.nbTriangles
    %reperer les points du triangle
    k1=m.TRIANGLES(l,1); %correspond au point A1
    k2=m.TRIANGLES(l,2); %correspond au point A2
    k3=m.TRIANGLES(l,3); %correspond au point A3

    %relever les coordonnees des points
    x1=m.POS(k1,1); y1=m.POS(k1,2); %coordonnees du point A1
    x2=m.POS(k2,1); y2=m.POS(k2,2); %coordonnees du point A2
    x3=m.POS(k3,1); y3=m.POS(k3,2); %coordonnees du point A3

    %calcul de l'aire du triangle
    aire=det([x2-x1,x3-x1;y2-y1,y3-y1])/2;
    
    %creation de la matrice de rigidite locale
    Kloc=zeros(3,3);
    J=[y3-y1,y1-y2;x1-x3,x2-x1];
    lambda=[-1,1,0;-1,0,1];
    for j=1:3
        Kloc(1,j)=(J*lambda(1:2,j))'*(J*lambda(1:2,1));
        Kloc(2,j)=(J*lambda(1:2,j))'*(J*lambda(1:2,2));
        Kloc(3,j)=(J*lambda(1:2,j))'*(J*lambda(1:2,3));
    end
    Kloc=1/(4*aire)*Kloc;

    %remplissage de la matrice de rigidite
    A(k1,k1)=A(k1,k1)+Kloc(1,1);    A(k1,k2)=A(k1,k2)+Kloc(1,2);    A(k1,k3)=A(k1,k3)+Kloc(1,3);
    A(k2,k1)=A(k2,k1)+Kloc(2,1);    A(k2,k2)=A(k2,k2)+Kloc(2,2);    A(k2,k3)=A(k2,k3)+Kloc(2,3);
    A(k3,k1)=A(k3,k1)+Kloc(3,1);    A(k3,k2)=A(k3,k2)+Kloc(3,2);    A(k3,k3)=A(k3,k3)+Kloc(3,3);

    %creation de la matrice de masse locale
    Mloc=(aire/12)*[2,1,1;1,2,1;1,1,2];
    
    %remplissage du second membre
    rhs([k1;k2;k3])=rhs([k1;k2;k3])+Mloc*[fun_s(x1,y1);fun_s(x2,y2);fun_s(x3,y3)];
end
MAT=A;

% Prise en compte des conditions aux limites
% Determination des points du bords 
% Dans ls structure GMSH, les points du bord composent les lignes au bord
% mais ne sont pas forcement ranges dans un ordre pre-defini
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
    if (type==Zone_Phys_Bord)
        % Test utile si on avait plusieurs zones differentes au bord
       val_CL=CL;
    end
    vec_cl(p)=val_CL;
    % On multiplie la p-ieme colonne de la matrice par la valeur CL, et on
    % la soustrait au second membre. Le but est de laisser la matrice
    % symetrique
    rhs=rhs-A*vec_cl;
    vec_cl(p)=0;
    % La p-ieme ligne et la p-ieme colonne de la matrice sont remplacees
    % par une ligne  et une colonne de zeros.
    A(p,:)=0;A(:,p)=0;
    % Le terme App est remplacee par 1.
    A(p,p)=1;
    % La composante rhs(p) est remplacee par CL.
    rhs(p)=val_CL;
end

sol=A\rhs;

% Trace version 1
figure(1);
clf;
nodes=zeros(m.nbNod,1);
for j=1:m.nbNod
    nodes(j)=j;
end
x=m.POS(nodes,1);
y=m.POS(nodes,2);
plot3(x,y,sol,'.');
view(-30,30);
axis tight;

% Trace version 2
figure(2);
clf;xP=zeros(3,m.nbQuads);
yP=xP;
zP=xP;
for j=1:m.nbTriangles
    for k=1:3
        noe=m.TRIANGLES(j,k);
        xP(k,j)=m.POS(noe,1);
        yP(k,j)=m.POS(noe,2);
        zP(k,j)=sol(noe);
    end
end
patch(xP,yP,zP,zP);
view(-30,30);
axis tight;

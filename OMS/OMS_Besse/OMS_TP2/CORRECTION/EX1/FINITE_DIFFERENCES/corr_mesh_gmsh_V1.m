% Version assemblage par position des points au Nord, Sud, Est, Ouest
% Le maillage est calcule par Gmsh regulier en x,y
% Les noeuds dans chaque quadranle sont ranges dans l'ordre
% 1 = Bas-Gauche, 2 = Bas-Droite, 3 = Haut-Droite, 4 = Haut Gauche
%
% Les points du bord n'appartiennent pas au maillage. On soustrait au
% second membre la contribution des conditions aux limites.
%
%   4 -------- 3
%   |          |
%   |          |
%   |          |
%   |          |
%   1 -------- 2

clear all;
Zone_Phys_Bord=1001;
CL=20;

m = load_gmsh('ex1.msh');
north=zeros(m.nbNod,1);
east=north;
south=north;
west=north;
nodes=north;
% Determination pour chaque noeud des noeuds au Nord, Sud, Est, Ouest
for j=1:m.nbQuads
    noeud=m.QUADS(j,1);
    nodes(noeud)=noeud;
    north(noeud)=m.QUADS(j,4);
    east(noeud)=m.QUADS(j,2);
    
    noeud=m.QUADS(j,2);
    nodes(noeud)=noeud;
    north(noeud)=m.QUADS(j,3);
    west(noeud)=m.QUADS(j,1);
    
    noeud=m.QUADS(j,3);
    nodes(noeud)=noeud;
    south(noeud)=m.QUADS(j,2);
    west(noeud)=m.QUADS(j,4);
    
    
    noeud=m.QUADS(j,4);
    nodes(noeud)=noeud;
    east(noeud)=m.QUADS(j,3);
    south(noeud)=m.QUADS(j,1);
end

dx=m.POS(m.QUADS(1,2),1)-m.POS(m.QUADS(1,1),1);
A=sparse(m.nbNod,m.nbNod);
jbord=zeros(m.nbNod,1); 
% Variable indiquant le nombre de fois ou un noeud n'a pas de voisin
% inconnu
rhs=zeros(m.nbNod,1);

for j=1:m.nbNod
    A(j,j)=4/dx^2;
    
    x=m.POS(j,1);
    y=m.POS(j,2);
    R=sqrt(x^2+y^2);

    rhs(j)=20*(R-1.5).^2.*(R-2).^2.*(R>1.5).*(R<2)-0.5*R.^2.*(R-1).^2.*(R<1);

    if (north(j))
        A(j,north(j))=-1/dx^2;
    else
        jbord(j)=jbord(j)+1;
    end
    if (south(j))
        A(j,south(j))=-1/dx^2;
    else
        jbord(j)=jbord(j)+1;
    end
    if (west(j))
        A(j,west(j))=-1/dx^2;
    else
        jbord(j)=jbord(j)+1;
    end
    if (east(j))
        A(j,east(j))=-1/dx^2;
    else
        jbord(j)=jbord(j)+1;
    end
end

for j=1:m.nbLines
    noe=m.LINES(j,1);
    if (m.LINES(j,3)==Zone_Phys_Bord)
        rhs(noe)=jbord(noe)*CL*1/dx^2;
    end
end



sol=A\rhs;

% Trace version 1
figure(1);
clf;
x=m.POS(nodes,1);
y=m.POS(nodes,2);
plot3(x,y,sol,'.');
view(-30,30);


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


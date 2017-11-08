clear all;
m = load_gmsh('geometrie.msh');
north=zeros(m.nbNod,1);
east=north;
south=north;
west=north;
nodes=north;
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
rhs=zeros(m.nbNod,1);
CL_ext=20;
CL_int=100;
jbord=zeros(m.nbNod,1);

for j=1:m.nbNod
    A(j,j)=4/dx^2;
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

rhs=zeros(m.nbNod,1);
CL_ext=20;
CL_int=100;
for j=1:m.nbLines
    noe=m.LINES(j,1);
    if (m.LINES(j,3)==1002)
        rhs(noe)=jbord(noe)*CL_ext*1/dx^2;
    else
        rhs(noe)=jbord(noe)*CL_int*1/dx^2;
    end
end

sol=A\rhs;
x=m.POS(nodes,1);
y=m.POS(nodes,2);

figure(2);
clf;
%plot3(x,y,sol,'.')

xP=zeros(4,m.nbQuads);
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
clf;
patch(xP,yP,zP,zP);
view(-30,30);


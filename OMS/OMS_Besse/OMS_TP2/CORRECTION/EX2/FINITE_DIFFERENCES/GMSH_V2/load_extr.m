% Version assemblage sans connaitre la position des points au N,S,E,W
% Le maillage est calcule par Gmsh regulier en x,y (dx=dy)
% Les noeud des quadrangles peuvent etre ranges dans un ordre quelconque.
%
% Dans cette verion, les points du bords n'appartiennent pas au maillage
%
clear all;
m = load_gmsh('extr.msh');

A=sparse(m.nbNod,m.nbNod);
rhs=zeros(m.nbNod,1);
CL_ext=20;
CL_int=100;
nb_fois_vu=zeros(m.nbNod,1);
nodes=zeros(m.nbNod,1);
for j=1:m.nbNod
    nodes(j)=j;
end

for j=1:m.nbQuads
    p1=m.QUADS(j,1);
    p2=m.QUADS(j,2);
    p3=m.QUADS(j,3);
    p4=m.QUADS(j,4);
    nb_fois_vu(p1)=nb_fois_vu(p1)+1;
    nb_fois_vu(p2)=nb_fois_vu(p2)+1;
    nb_fois_vu(p3)=nb_fois_vu(p3)+1;
    nb_fois_vu(p4)=nb_fois_vu(p4)+1;
    
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
    L2=(m.POS(p3,1)-m.POS(p4,1))^2+(m.POS(p3,2)-m.POS(p4,2))^2;
    A(p4,p4)=2/L1+2/L2;
    A(p4,p3)=-1/L1;
    A(p4,p1)=-1/L2;
    
end

rhs=zeros(m.nbNod,1);
CL_ext=20;
CL_int=100;
for j=1:m.nbLines
    p1=m.LINES(j,1);
    p2=m.LINES(j,2);
    if (p1==1) disp('ici'); end;
    if (p2==1) disp('ici'); end;
    if (nb_fois_vu(p1)<=2)
        d=(m.POS(p2,1)-m.POS(p1,1))^2+(m.POS(p2,2)-m.POS(p1,2))^2;
        if (m.LINES(j,3)==1002)
            rhs(p1)=rhs(p1)+0.5*(3-nb_fois_vu(p1))*CL_ext*1/d;
        else
            rhs(p1)=rhs(p1)+0.5*(3-nb_fois_vu(p1))*CL_int*1/d;
        end
    end
    if (nb_fois_vu(p2)<=2)
        d=(m.POS(p2,1)-m.POS(p1,1))^2+(m.POS(p2,2)-m.POS(p1,2))^2;
        if (m.LINES(j,3)==1002)
            rhs(p2)=rhs(p2)+0.5*(3-nb_fois_vu(p2))*CL_ext*1/d;
        else
            rhs(p2)=rhs(p2)+0.5*(3-nb_fois_vu(p2))*CL_int*1/d;
        end
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
patch(xP,yP,zP,zP);


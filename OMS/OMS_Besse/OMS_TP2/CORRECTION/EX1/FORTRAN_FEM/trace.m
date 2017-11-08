% Trace la solution
load solution.dat;
m = load_gmsh('ex1.msh');
clf;

xP = zeros(3,m.nbQuads);
yP = xP;
zP = xP;
for j=1:m.nbTriangles
    for k=1:3
        noe = m.TRIANGLES(j,k);
        xP(k,j) = m.POS(noe,1);
        yP(k,j) = m.POS(noe,2);
        zP(k,j) = solution(noe);
    end
end
patch(xP,yP,zP,zP);
view(-30,30);
axis tight;

m = load_gmsh('essai.msh')
north = zeros(m.nbNod, 1);
south = north;
east = north;
west = north;
nodes = north;

for j = 1 : m.nbQuads
    noeud = m.QUADS(j, 1);
    nodes(noeud) = noeud;
    north(noeud) = m.QUADS(j, 4);
    east(noeud) = m.QUADS(j, 2);
    
    noeud = m.QUADS(j, 2);
    nodes(noeud) = noeud;
    north(noeud) = m.QUADS(j, 3);
    west(noeud) = m.QUADS(j, 1);
    
    noeud = m.QUADS(j, 3);
    nodes(noeud) = noeud;
    west(noeud) = m.QUADS(j, 4);
    south(noeud) = m.QUADS(j, 2);
    
    noeud = m.QUADS(j, 4);
    nodes(noeud) = noeud;
    south(noeud) = m.QUADS(j, 1);
    east(noeud) = m.QUADS(j, 3);
    
end

dx = m.POS(m.QUADS(1,2), 1) - m.POS(m.QUADS(1,1), 1);
A = sparse(m.nbNod, m.nbNod);
jbord = zeros(m.nbNod, 1);

! Pour le stencil : à chaque fois que l'un des éléments pointés n'est pas
! au bord, on affecte à notre habitude les éléments de la matrice A.
! Sinon, on ajoute 1 à jbord(j) (du fait d'être situé près du bord).
for j = 1 : nbNod
    A(j,j) = 4/dx^2;
    
    if ( north(j) ~= 0 )
        A(j, north(j)) = -1/dx^2;
    else
        jbord(j) = jbord(j) + 1;
    end

    if ( south(j) ~= 0 )
        A(j, south(j)) = -1/dx^2;
    else
        jbord(j) = jbord(j) + 1;
    end

    if ( west(j) ~= 0 )
        A(j, west(j)) = -1/dx^2;
    else
        jbord(j) = jbord(j) + 1;
    end

    if ( east(j) ~= 0 )
        A(j, east(j)) = -1/dx^2;
    else
        jbord(j) = jbord(j) + 1;
    end
end

! rhs = right hand side : membre droit de l'égalité.
rhs = zeros(m.nbNod, 1);
temp_ext = 20;
temp_int = 100;

! Si on est proche du bord, on doit insérer dans le membre de droite le
! terme connu.
for j = 1 : nbLines
    ! LINES(1, _) = [ noeud1 ; noeud2 ; num_ligne_du_bord ]
    noe = m.LINES(j,1); ! noe = noeud1 de la ligne j
    if ( m.LINES(j,3) == 47 )
        rhs(noe) = jbord(noe) * temp_ext * 1/dx^2;
    else if ( m.LINE(j,3) == 48 )
        rhs(noe) = jbord(noe) * temp_int * 1/dx^2;
    end
end

sol = A\rhs;
x = m.POS(nodes, 1);
y = m.POS(nodes, 2);

figure(2);
clf;
%plot3(x,y,sol, '.');

xP = zeros(4, m.nbQuads);
yP = xP;
zP = xP;

for j = 1 : m.nbQuads
    for k = 1 : 4
        noe = m.QUADS(j,k);
        xP(k,j) = m.POS(noe,1);
        yP(k,j) = m.POS(noe,2);
        zP(k,j) = sol(noe);
    end
end

patch(xP,yP,zP,zP);
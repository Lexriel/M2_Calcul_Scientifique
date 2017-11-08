MODULE exo1_ef_module

  use lapack95
  use blas95
  use mod_lec_fic.f90

  IMPLICIT NONE

  CONTAINS

  SUBROUTINE 

  type(msh) :: mesh
  TYPE(MatriceCSR) :: Mat2D
  integer :: CL, Zone_Phys_Bord, k1, k2, k3, i, j, k, l, m
  real, dimension(3,3) :: S1, S2, S3, M1
  real, dimension(2) :: P1, P2, P3, B1, B2
  real, dimension(2) :: x1, x2, x3, G1, G2, G3
  real, dimension(3) :: tri
  real, dimension(:,:), allocatable :: A, ind, val
  real, dimension(:), allocatable :: rhs, nbe
  real : detB, line


  call load_gmsh('ex1.msh', mesh)

  print *, "Nombre de noeud : ", mesh%nbNod
  print *, "Nombre de triangle : ", mesh%nbtriangles

CL = 20
Zone_Phys_Bord = 1001

S1 = 0.5*[1,-1,0; -1,1,0; 0,0,0]
S2 = 0.5*[2,-1,-1; -1,0,1; -1,1,0]
S3 = 0.5*[1,0,-1; 0,0,0; -1,0,1]

M1 = reshape ((/2,1,1, 1,2,1, 1,1,2/),(/3,3/))

allocate( ind(mesh%nbNod, 10) ) 

allocate( rhs(mesh%nbNod), nbe(mesh%nbNod) )

allocate( val(mesh%nbNod, 10) )


do l = 1, mesh%nbTriangles
  tri = mesh%TRIANGLES(l,1:3)
  P1 = mesh%POS(tri(1),1:2)
  P2 = mesh%POS(tri(2),1:2)
  P3 = mesh%POS(tri(3),1:2)

  k1 = mesh%TRIANGLES(l,1)
  k2 = mesh%TRIANGLES(l,2)
  k3 = mesh%TRIANGLES(l,3)

  x1 = mesh%POS(k1,1);  y1 = mesh%POS(k1,2);
  x2 = mesh%POS(k2,1);  y2 = mesh%POS(k2,2);
  x3 = mesh%POS(k3,1);  y3 = mesh%POS(k3,2);

  B1 = P2-P1 ;   B2 = P3-P1;
  detB = abs( B1(1)*B2(2) - B1(2)*B2(1) )
  G1 = ( B2(1)*B2(1) + B2(2)*B2(2) )/detB
  G2 =-( B1(1)*B2(1) + B1(2)*B2(2) )/detB
  G3 = ( B1(1)*B1(1) + B1(2)*B1(2) )/detB
 

!!!!!!!!!!!!! essai !!!!!!!!!!!!!!!!!!!!

  Mloc = (detB/24) * M1
  Aloc = G1*S1 + G2*S2 + G3*S3

  do m = 1, 3
    line = tri(m)    ! on se met sur la ligne du noeud courant du triangle courant
    rhs(line) = rhs(line) + Mloc * [fun_s(x1,y1); fun_s(x2,y2); fun_s(x3,y3)]   ! remplissage du second membre


    if (nbe(line) == 0) then
      ind(line,1:3) = tri
      nbe(line) = 3
      val(line, 1:3) = G1*S1(m,1:3) + G2*S2(m,1:3) + G3*S3(m,1:3)
    else

      out: do i = 1,3 ! j'appelle cette boucle 'out'
        do j = 1, nbe(line)
          if (ind(line, j) == tri(i)) then
            val(line, j) = val(line, j) + Aloc(m, i)
            cycle out ! dans ce cas, on passe à l'itération suivante de 'out : do'
          end if
        end do

        nbe(line) = nbe(line)+1
        ind(line, nbe(line)) = tri(i)
        val(line, nbe(line)) = Aloc(m,i)

      end do out
    end if
  end do



!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!







 
  !Aloc = G1*S1 + G2*S2 + G3*S3
  
  !A(k1,k1) = A(k1,k1) + Aloc(1,1);    A(k1,k2) = A(k1,k2) + Aloc(1,2);    A(k1,k3) = A(k1,k3) + Aloc(1,3);
  !A(k2,k1) = A(k2,k1) + Aloc(2,1);    A(k2,k2) = A(k2,k2) + Aloc(2,2);    A(k2,k3) = A(k2,k3) + Aloc(2,3);
  !A(k3,k1) = A(k3,k1) + Aloc(3,1);    A(k3,k2) = A(k3,k2) + Aloc(3,2);    A(k3,k3) = A(k3,k3) + Aloc(3,3);
  
!  %creation de la matrice de masse locale
!  Mloc = (detB/24) * M1
!  %remplissage du second membre
!  rhs([k1;k2;k3]) = rhs([k1;k2;k3]) + Mloc * [fun_s(x1,y1); fun_s(x2,y2); fun_s(x3,y3)]
end do

  
% Prise en compte des conditions aux limites
% Determination des points du bords 
% Dans ls structure GMSH, les points du bord composent les lignes au bord
% mais ne sont pas forcement ranges dans un ordre pre-defini

noeud_bord = zeros(m.nbLines,2)
vu = zeros(m.nbNod,1)
k=0
do j = 1, m.nbLines
    p1 = m.LINES(j,1)
    p2 = m.LINES(j,2)
    type = m.LINES(j,3)    
    if not(vu(p1))
        k = k+1
        noeud_bord(k,1) = p1
        noeud_bord(k,2) = type
        vu(p1) = k
    end if
    if not(vu(p2))
        k = k+1
        noeud_bord(k,1) = p2
        noeud_bord(k,2) = type        
        vu(p2) = k
    end if
end do

vec_cl = zeros(m.nbNod,1)
do j = 1, m.nbLines
    p = noeud_bord(j,1)
    type = noeud_bord(j,2)
    if (type == Zone_Phys_Bord)
        % Test utile si on avait plusieurs zones differentes au bord
       val_CL = CL
    end if
    vec_cl(p) = val_CL
    % On multiplie la p-ieme colonne de la matrice par la valeur CL, et on
    % la soustrait au second membre. Le but est de laisser la matrice
    % symetrique
    rhs = rhs - A*vec_cl
    vec_cl(p) = 0
    % La p-ieme ligne et la p-ieme colonne de la matrice sont remplacees
    % par une ligne  et une colonne de zeros.
    A(p,:) = 0; A(:,p) = 0;
    % Le terme App est remplacee par 1.
    A(p,p) = 1
    % La composante rhs(p) est remplacee par CL.
    rhs(p) = val_CL
end do

sol = A\rhs

% Trace version 1
figure(1);
clf;
nodes = zeros(m.nbNod,1)
do j = 1, m.nbNod
    nodes(j) = j
end do
x = mes%POS(nodes,1)
y = mesh.POS(nodes,2)
plot3(x,y,sol,'.');
view(-30,30);
axis tight;

% Trace version 2
figure(2);
clf;xP=zeros(3,m.nbQuads);
yP = xP;
zP = xP;
do j = 1, mesh%nbTriangles
    do k = 1, 3
        noe = mesh%TRIANGLES(j,k)
        xP(k,j) = mesh%POS(noe,1)
        yP(k,j) = mesh%POS(noe,2)
        zP(k,j) = sol(noe)
    end do
end do
patch(xP,yP,zP,zP);
view(-30,30);
axis tight;
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


  END SUBROUTINE

END MODULE

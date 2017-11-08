// Definition des points

Point(1) = {0,0,0,0.1};
Point(2) = {1,0,0,0.1};
Point(3) = {1,1,0,0.1};
Point(4) = {0,1,0,0.1};
Point(5) = {0.25, 0.25, 0, 0.1};
Point(6) = {0.75, 0.25, 0, 0.1};
Point(7) = {0.75, 0.75, 0, 0.1};
Point(8) = {0.25, 0.75, 0, 0.1};
Point(9) = {0., 0.25, 0, 0.1};
Point(10) = {1, 0.25, 0, 0.1};
Point(11) = {1., 0.75, 0, 0.1};
Point(12) = {0, 0.75, 0, 0.1};
Point(13) = {0.25, 0, 0, 0.1};
Point(14) = {0.75, 0., 0, 0.1};
Point(15) = {0.75, 1, 0, 0.1};
Point(16) = {0.25, 1, 0, 0.1};

//  4---16--------15-----3
//  |                    |
//  12   8---------7     11
//  |	 |         |     |
//  |	 |         |     |
//  |    |         |     |
//  |    |         |     |
//  |	 |         |     |
//  |	 |         |     |
//  9	 5---------6     10
//  |                    |
//  1 ---13--------14----2


// Definition des lignes
Line(1) = {1, 13};
Line(2) = {13, 14};
Line(3) = {14,2};
Line(4) = {2,10};
Line(5) = {10,11};
Line(6) = {11,3};
Line(7) = {3,15};
Line(8) = {15,16};
Line(9) = {16,4};
Line(10) = {4,12};
Line(11) = {12,9};
Line(12) = {9,1};
Line(13) = {5,6};
Line(14) = {6,7};
Line(15) = {7,8};
Line(16) = {8,5};
Line(17) = {13,5};
Line(18) = {14,6};
Line(19) = {6,10};
Line(20) = {7,11};
Line(21) = {7,15};
Line(22) = {8,16};
Line(23) = {8,12};
Line(24) = {5,9};

// Definitions des zones physiques avec attribution d'un identifiant
Physical Line(1001) = {13, 14, 15, 16};
Physical Line(1002) = {6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5};
Physical Surface(1003) = {1, 2, 3, 4, 5, 6, 7, 8};

// Definition des regions permettant la creation des surfaces
// !!!!!! ATTENTION !!!!!
// On veut mailler les quadrangles dans la direction
//   4 --------- 3
//   |           |
//   |           |
//   1-----------2
// Pour cela, toutes les zones rectangulaires doivent etre definies
// avec cette orientation en commencant par le coin inferieur gauche

Line Loop(1) = {1,17,24,12};
Line Loop(2) = {2,18,-13,-17};
Line Loop(3) = {3,4,-19,-18}; 
Line Loop(4) = {19,5,-20,-14}; 
Line Loop(5) = {20,6,7,-21}; 
Line Loop(6) = {-15,21,8,-22};
Line Loop(7) = {-23,22,9,10};
Line Loop(8) = {-24,-16,23,11};

// Definition des surfaces
Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};
Plane Surface(6) = {6};
Plane Surface(7) = {7};
Plane Surface(8) = {8};

// Tell Gmsh how many cells you want per edge
// Transfinite Line{1,24,23,9} = 30;
// Transfinite Line{3,19,20,7} = 30;
// Transfinite Line{10,22,21,6} = 30;
// Transfinite Line{12,17,18,4} = 30;
// Transfinite Line{11,16,14,5} = 59;
// Transfinite Line{2,13,15,8} = 59;

// Transfinite Line{1,24,23,9} = 20;
// Transfinite Line{3,19,20,7} = 20;
// Transfinite Line{10,22,21,6} = 20;
// Transfinite Line{12,17,18,4} = 20;
// Transfinite Line{11,16,14,5} = 39;
// Transfinite Line{2,13,15,8} = 39;

Transfinite Line{1,24,23,9} = 4;
Transfinite Line{3,19,20,7} = 4;
Transfinite Line{10,22,21,6} = 4;
Transfinite Line{12,17,18,4} = 4;
Transfinite Line{11,16,14,5} = 7;
Transfinite Line{2,13,15,8} = 7;

// Tell Gmsh what the corner points are(going clockwise or counter-clockwise):
// !!!!!! ATTENTION !!!!! COMME POUR LA DEF DES REGIONS
// On veut mailler les quadrangles dans la direction
//   4 --------- 3
//   |           |
//   |           |
//   1-----------2
// Pour cela, toutes les zones rectangulaires doivent etre definies
// avec cette orientation en commencant par le coin inferieur gauche
Transfinite Surface{1} = {1,13,5,9};
Transfinite Surface{2} = {13,14,6,5};
Transfinite Surface{3} = {14,2,10,6};
Transfinite Surface{4} = {6,10,11,7};
Transfinite Surface{5} = {7,11,3,15};
Transfinite Surface{6} = {8,7,15,16};
Transfinite Surface{7} = {12,8,16,4};
Transfinite Surface{8} = {9,5,8,12};

// Recombine the triangles into quads:
// Combine des triangles pour former des quadrilateres
Recombine Surface{1};
Recombine Surface{2};
Recombine Surface{3};
Recombine Surface{4};
Recombine Surface{5};
Recombine Surface{6};
Recombine Surface{7};
Recombine Surface{8};

// Havent tested this yet, but doesnt seem to hurt:
Mesh.Smoothing = 100;


lc=0.1;
N=50;
// Definition des points
Point(1) = {-3,-3,0,lc};
Point(2) = {3,-3,0,lc};
Point(3) = {3,3,0,lc};
Point(4) = {-3,3,0,lc};

// Definition des lignes
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3 ,4};
Line(4) = {4, 1};

// Definition du domaine
Line Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};


// Definitions des zones physiques avec attribution d'un identifiant
Physical Line(1001) = {1, 2, 3, 4};
Physical Surface(1003) = {1};

// Tell Gmsh how many cells you want per edge
Transfinite Line{1,2,3,4} = N;

// Tell Gmsh what the corner points are(going clockwise or counter-clockwise):
Transfinite Surface{1} = {1,2,3,4};

// Recombine the triangles into quads:
Recombine Surface{1};

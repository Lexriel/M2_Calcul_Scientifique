N=50;
Point(1) = {-3,-3,0,0.1};
// Cree une ligne a partir du point 1 en l'extrudant dans la direction d=(6,0) et en 
// prevoyant pour le maillage N points repartis de maniere reguliere dans la direction d
Extrude {6,0,0} {Point{1};Layers {{N},{1}};}
// Cree une surface a partir de la ligne en l'extrudant dans la direction d=(0,6) et en 
// prevoyant pour le maillage N points repartis de maniere reguliere dans la direction d
Extrude {0,6,0} {Line{1}; Layers{ {N}, {1}};Recombine;}

// Definitions des zones physiques avec attribution d'un identifiant
Physical Line(1001) = {1, 2, 3, 4};
Physical Surface(1003) = {5};


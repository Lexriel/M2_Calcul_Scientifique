// Version 1
N1=10;
N2=20;
Point(1)={0,0,0,2};
Extrude {0.25,0,0} {Point{1};Layers {{N1},{1}};}
Extrude {0,0.25,0} {Line{1}; Layers{ {N1}, {1}};Recombine;}
Extrude {0.5,0,0} {Line{4}; Layers{ {N2}, {1}};Recombine;}
Extrude {0.25,0,0} {Line{6}; Layers{ {N1}, {1}};Recombine;}
Extrude {0,.5,0} {Line{12}; Layers{ {N2}, {1}};Recombine;}
Extrude {0,0.25,0} {Line{14}; Layers{ {N1}, {1}};Recombine;}
Extrude {-0.5,0,0} {Line{19}; Layers{ {N2}, {1}};Recombine;}
Extrude {-0.25,0,0} {Line{22}; Layers{ {N1}, {1}};Recombine;}
Extrude {0,-0.5,0} {Line{27}; Layers{ {N2}, {1}};Recombine;}

Physical Line(1001) = {8, 15,23,31};
Physical Line(1002) = {1,7,11,10,16,20,18,24,28,26,32,3};
Physical Surface(1003) = {5,9,13,17,21,25,29,33};


// Version 1
//Point(1)={0,0,0,2};
//Extrude {1,0,0} {Point{1};Layers {{4},{1}};}
//Extrude {0,3,0} {Line{1}; Layers{ {4,4,4}, {0.333,0.666,1}};Recombine;}
//Extrude {0,1,0} {Point{2};Layers {{4},{1}};}
//Extrude {2,0,0} {Line{6}; Layers{ {4,4}, {0.5,1}};Recombine;}
//Extrude {-1,0,0} {Point{7};Layers {{4},{1}};}
//Extrude {0,2,0} {Line{11}; Layers{ {4,4}, {0.5,1}};Recombine;}
//Extrude {0,-1,0} {Point{10};Layers {{4},{1}};}
//Extrude {-1,0,0} {Line{16}; Layers{ {4}, {1}};Recombine;}
//
//Line(50)={5,8};
//Line(51)={8,11};
//
//
//Physical Line(1001) = {1,8,7,13,12,18,2,3};
//Physical Line(1002) = {50,51,19,4};
//Physical Surface(1003) = {5,10,15,20};




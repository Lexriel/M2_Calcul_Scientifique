#ATP TOPOLOGY for Swiss-PdbViewer 3.5
 based on GROMOS96 parameters kindly provided by Wilfred van Gunsteren.  
 Ref: W.F. van Gunsteren et al. (1996) in Biomolecular simulation:
      the GROMOS96 manual and user guide. Vdf Hochschulverlag ETHZ
      (http://igc.ethz.ch/gromos)
-----------------------------------------------------------------------
//NAME    TYPE      CHARGE-B1  AT   GRO-43A1  SIMPEL  
 N9       NR        -0.2000     8   -0.2000    0.0000
 C4       C          0.2000    11    0.2000    0.0000
 N3       NR        -0.3600     8   -0.3600    0.0000
 C2       CR1        0.3600    16    0.3600    0.0000
 N1       NR        -0.3600     8   -0.3600    0.0000
 C6       C          0.3600    11    0.3600    0.0000
 N6       NT        -0.8300     6   -0.8300    0.0000
 H61      H          0.4150    18    0.4150    0.0000
 H62      H          0.4150    18    0.4150    0.0000
 C5       C          0.0000    11    0.0000    0.0000
 N7       NR        -0.3600     8   -0.3600    0.0000
 C8       CR1        0.3600    16    0.3600    0.0000
 C1*      CH1        0.2000    12    0.2000    0.0000
 O4*      OA        -0.3600     3   -0.3600    0.0000
 C4*      CH1        0.1600    12    0.1600    0.0000
 C2*      CH1        0.1500    12    0.1500    0.0000
 O2*      OA        -0.5480     3   -0.5480    0.0000
 H2*      H          0.3980    18    0.3980    0.0000
 C3*      CH1        0.1500    12    0.1500    0.0000
 O3*      OA        -0.5480     3   -0.5480    0.0000
 H3*      H          0.3980    18    0.3980    0.0000
 C5*      CH2        0.0000    13    0.0000    0.0000
 O5*      OA        -0.3600     3   -0.3600    0.0000
 PA       P,SI       1.1550    27    0.7050   -1.0000
 O1A      OM        -0.3600     2   -0.6350    0.0000
 O2A      OM        -0.3600     2   -0.6350    0.0000
 O3A      OA        -0.3600     3   -0.3600    0.0000
 PB       P,SI       1.1550    27    0.7050   -1.0000
 O1B      OM        -0.3600     2   -0.6350    0.0000
 O2B      OM        -0.3600     2   -0.6350    0.0000
 O3B      OA        -0.3600     3   -0.3600    0.0000
 PG       P,SI       1.0800    27    0.6300   -1.0000
 O1G      OM        -0.3600     2   -0.6350    0.0000
 O2G      OM        -0.3600     2   -0.6350    0.0000
 O3G      OA        -0.5480     3   -0.5480    0.0000
 H3G      H          0.3980    18    0.3980    0.0000
//BOND
---------
 N9       C4           9
 N9       C8           9
 N9       C1*         21
 C4       N3          11
 C4       C5          15
 N3       C2           6
 C2       N1           6
 N1       C6          11
 C6       N6           8
 C6       C5          15
 N6       H61          2
 N6       H62          2
 C5       N7           9
 N7       C8           9
 C1*      O4*         19
 C1*      C2*         25
 O4*      C4*         19
 C4*      C3*         25
 C4*      C5*         25
 C2*      O2*         19
 C2*      C3*         25
 O2*      H2*          1
 C3*      O3*         19
 O3*      H3*          1
 C5*      O5*         19
 O5*      PA          27
 PA       O1A         23
 PA       O2A         23
 PA       O3A         27
 O3A      PB          27
 PB       O1B         23
 PB       O2B         23
 PB       O3B         27
 O3B      PG          27
 PG       O1G         23
 PG       O2G         23
 PG       O3G         27
 O3G      H3G          1
//ANGLE
-----------------------
 C4       N9       C8           6
 C4       N9       C1*         36
 C8       N9       C1*         36
 N9       C4       N3          38
 N9       C4       C5           6
 N3       C4       C5          26
 C4       N3       C2          26
 N3       C2       N1          26
 C2       N1       C6          26
 N1       C6       N6          26
 N1       C6       C5          26
 N6       C6       C5          26
 C6       N6       H61         22
 C6       N6       H62         22
 H61      N6       H62         23
 C4       C5       C6          26
 C4       C5       N7           6
 C6       C5       N7          38
 C5       N7       C8           6
 N9       C8       N7           6
 N9       C1*      O4*          8
 N9       C1*      C2*          8
 O4*      C1*      C2*          8
 C1*      O4*      C4*          9
 O4*      C4*      C3*          8
 O4*      C4*      C5*          8
 C3*      C4*      C5*          7
 C1*      C2*      O2*          8
 C1*      C2*      C3*          7
 O2*      C2*      C3*          8
 C2*      O2*      H2*         11
 C4*      C3*      C2*          7
 C4*      C3*      O3*          8
 C2*      C3*      O3*          8
 C3*      O3*      H3*         11
 C4*      C5*      O5*          8
 C5*      O5*      PA          25
 O5*      PA       O1A         13
 O5*      PA       O2A         13
 O5*      PA       O3A          4
 O1A      PA       O2A         28
 O1A      PA       O3A         13
 O2A      PA       O3A         13
 PA       O3A      PB          25
 O3A      PB       O1B         13
 O3A      PB       O2B         13
 O3A      PB       O3B          4
 O1B      PB       O2B         28
 O1B      PB       O3B         13
 O2B      PB       O3B         13
 PB       O3B      PG          25
 O3B      PG       O1G         13
 O3B      PG       O2G         13
 O3B      PG       O3G          4
 O1G      PG       O2G         28
 O1G      PG       O3G         13
 O2G      PG       O3G         13
 PG       O3G      H3G         11
//IMPROPER
-----------------------
 N9       C8       C4       C1*          1
 C4       N9       C8       N7           1
 C8       N9       C4       C5           1
 N9       C8       N7       C5           1
 N9       C4       C5       N7           1
 C8       N7       C5       C4           1
 C4       N9       N3       C5           1
 C5       C6       N7       C4           1
 N3       C4       C5       C6           1
 C4       C5       C6       N1           1
 C5       C4       N3       C2           1
 C5       C6       N1       C2           1
 C4       N3       C2       N1           1
 N3       C2       N1       C6           1
 C6       C5       N1       N6           1
 N6       H61      H62      C6           1
 C1*      N9       O4*      C2*          2
 C4*      O4*      C5*      C3*          2
 C2*      O2*      C3*      C1*          2
 C3*      C2*      O3*      C4*          2
//TORSION
-----------------------
 C5       C6       N6       H61          4
 O4*      C1*      N9       C4           6
 C1*      C2*      O2*      H2*         12
 C4*      C3*      O3*      H3*         12
 C4*      O4*      C1*      C2*         14
 C3*      C4*      O4*      C1*         14
 C3*      C2*      C1*      O4*         17
 C3*      C2*      C1*      O4*          7
 O2*      C2*      C1*      N9           7
 O2*      C2*      C1*      O4*          8
 C4*      C3*      C2*      C1*         17
 O3*      C3*      C2*      C1*          7
 C4*      C3*      C2*      O2*          7
 O3*      C3*      C2*      O2*          8
 C5*      C4*      C3*      C2*         17
 C5*      C4*      C3*      O3*          7
 O4*      C4*      C3*      C2*          7
 O4*      C4*      C3*      O3*          8
 O5*      C5*      C4*      C3*         17
 O5*      C5*      C4*      C3*          7
 O5*      C5*      C4*      O4*          8
 C4*      C5*      O5*      PA          14
 C5*      O5*      PA       O3A         11
 C5*      O5*      PA       O3A          9
 O5*      PA       O3A      PB          11
 O5*      PA       O3A      PB           9
 PA       O3A      PB       O3B         11
 PA       O3A      PB       O3B          9
 O3A      PB       O3B      PG          11
 O3A      PB       O3B      PG           9
 PB       O3B      PG       O3G         11
 PB       O3B      PG       O3G          9
 O3B      PG       O3G      H3G         11
 O3B      PG       O3G      H3G          9

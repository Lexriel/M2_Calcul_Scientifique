#GLU TOPOLOGY for Swiss-PdbViewer 3.5
 based on GROMOS96 parameters kindly provided by Wilfred van Gunsteren.  
 Ref: W.F. van Gunsteren et al. (1996) in Biomolecular simulation:
      the GROMOS96 manual and user guide. Vdf Hochschulverlag ETHZ
      (http://igc.ethz.ch/gromos)
-----------------------------------------------------------------------
//NAME    TYPE      CHARGE-B1  AT   GRO-43A1  SIMPEL  
 N        N         -0.2800     5   -0.2800    0.0000 
 H        H          0.2800    18    0.2800    0.0000 
 CA       CH1        0.0000    12    0.0000    0.0000 
 CB       CH2        0.0000    13    0.0000    0.0000 
 CG       CH2        0.0000    13    0.0000    0.0000 
 CD       C          0.7200    11    0.2700    0.0000 
 OE1      OM        -0.3600     2   -0.6350   -0.5000 
 OE2      OM        -0.3600     2   -0.6350   -0.5000 
 C        C          0.3800    11    0.3800    0.0000 
 O        O         -0.3800     1   -0.3800    0.0000 
//BOND
---------
 N        H            2
 N        CA          20
 CA       C           26
 C        O            4
 C       +N            9
 CA       CB          26
 CB       CG          26
 CG       CD          26
 CD       OE1          5
 CD       OE2          5
//ANGLE
-----------------------
-C        N        H           31
 H        N        CA          17
-C        N        CA          30
 N        CA       C           12
 CA       C       +N           18
 CA       C        O           29
 O        C       +N           32
 N        CA       CB          12
 C        CA       CB          12
 CA       CB       CG          14
 CB       CG       CD          14
 CG       CD       OE1         21
 CG       CD       OE2         21
 OE1      CD       OE2         37
//IMPROPER
-----------------------
 N       -C        CA       H            1
 C        CA      +N        O            1
 CA       N        C        CB           2
 CD       OE1      OE2      CG           1
//TORSION
-----------------------
-CA      -C        N        CA           4
-C        N        CA       C           19
 N        CA       C       +N           20
 N        CA       CB       CG          17
 CA       CB       CG       CD          17
 CB       CG       CD       OE2         20

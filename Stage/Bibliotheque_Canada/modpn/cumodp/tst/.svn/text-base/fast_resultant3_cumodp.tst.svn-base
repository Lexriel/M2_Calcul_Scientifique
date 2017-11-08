#Test function:
#- MultivariateResultant 

#test 500
with(RegularChains):
with(FastArithmeticTools):
with(modpn):
with(TestTools):

## Setting the polynomial ring with a prime characteristic
p := 469762049:
vars := [x, y, z]:
v := vars[1]:
R := PolynomialRing(p, vars):

dx := 8;
dy := dx:
dz := dx:
f := randpoly(vars, dense, degree = dx) mod p:
g := randpoly(vars, dense, degree = dy) mod p:
fm := PolynomialConvertIn(R, f):
gm := PolynomialConvertIn(R, g):

EnableCUDA(true):
Try[testnoerror](10, MultivariateResultant(R, fm, gm, p), 'assign'='r1'):
r2 := PolynomialConvertOut(r1):

EnableCUDA(false):
Try[testnoerror](20, MultivariateResultant(R, fm, gm, p), 'assign'='r3'):
r4 := PolynomialConvertOut(r3):

Try[testnoerror](30, Resultant(f, g, v) mod p, 'assign'='r0'):
Try(40, evalb(`mod`(Expand(r2 - r0), p) = 0), true);
Try(50, evalb(`mod`(Expand(r4 - r0), p) = 0), true);


#end test;

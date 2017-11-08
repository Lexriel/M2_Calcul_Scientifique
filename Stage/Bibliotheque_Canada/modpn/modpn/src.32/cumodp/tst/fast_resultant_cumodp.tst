#Test function:
#- MultivariateResultant 

#test 350
with(RegularChains):
with(FastArithmeticTools):
with(modpn):
with(TestTools):

## Setting the polynomial ring with a prime characteristic
p := 469762049;
vars := [a, b]:
v := vars[1]:
R := PolynomialRing(p, vars):

for i from 1 to 3 do 
    dx := i * 10:
    dy := i * 10:
    f := randpoly(vars, dense, degree = dx) mod p:
    g := randpoly(vars, dense, degree = dy) mod p:
    fm := PolynomialConvertIn(R, f):
    gm := PolynomialConvertIn(R, g):
    ## compute resultant using modpn C
    Try[testnoerror](i*10, MultivariateResultant(R, fm, gm, p), 'assign'='r1'):
    r2 := PolynomialConvertOut(r1):
    ## Maple default
    Try[testnoerror](i*10 + 1, `mod`(Resultant(f, g, v), p) , 'assign'='r3'):
    ## Comparing the results
    Try(i * 10 + 2, evalb(`mod`(Expand(r3 - r2), p) = 0), true);
end do:

p := 469762049;
vars := [a, b, c]:
v := vars[1]:
R := PolynomialRing(p, vars):

for i from 1 to 2 do 
    dx := 8 + i:
    dy := 8 + i:
    f := randpoly(vars, dense, degree = dx) mod p:
    g := randpoly(vars, dense, degree = dy) mod p:
    fm := PolynomialConvertIn(R, f):
    gm := PolynomialConvertIn(R, g):
    ## compute resultant using modpn C
    Try[testnoerror](i*40, MultivariateResultant(R, fm, gm, p), 'assign'='r1'):
    r2 := PolynomialConvertOut(r1):
    ## Maple default
    Try[testnoerror](i*40 + 1, `mod`(Resultant(f, g, v), p), 'assign'='r3'):
    ## Comparing the results
    Try(i * 10 + 2, evalb(`mod`(Expand(r3 - r2), p) = 0), true);
end do:


#end test;

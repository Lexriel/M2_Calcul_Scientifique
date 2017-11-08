with(RegularChains):
with(FastArithmeticTools):
with(modpn):

p := 469762049:
vars := [a, b, c]:
v := vars[1]:
R := PolynomialRing(p, vars):

f := randpoly(vars, dense, degree = 10) mod p:
g := randpoly(vars, dense, degree = 10) mod p:
fm := PolynomialConvertIn(R, f):
gm := PolynomialConvertIn(R, g):
## compute resultant using modpn C
t1 := time():
r1 := MultivariateResultant(R, fm, gm, p):
t2 := time() - t1:
r2 := PolynomialConvertOut(r1):
## Maple default
t1 := time():
r3 := `mod`(Resultant(f, g, v), p):
t2 := time() - t1:
print("maple timing ", t2):
## Comparing the results
print(evalb(`mod`(Expand(r3 - r2), p) = 0)):

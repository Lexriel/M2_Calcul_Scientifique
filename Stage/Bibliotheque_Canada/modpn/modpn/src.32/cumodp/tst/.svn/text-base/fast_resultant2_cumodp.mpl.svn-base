
with(RegularChains):
with(FastArithmeticTools):
with(modpn):

## Setting the polynomial ring with a prime characteristic
p := 469762049:
vars := [a, b]:
v := vars[1]:
R := PolynomialRing(p, vars):

dx := 80:
dy := 60:
f := randpoly(vars, dense, degree = dx) mod p:
g := randpoly(vars, dense, degree = dy) mod p:
fm := PolynomialConvertIn(R, f):
gm := PolynomialConvertIn(R, g):

EnableCUDA(true):
t1 := time():
r1 := MultivariateResultant(R, fm, gm, p):
t1 := time() - t1:
r2 := PolynomialConvertOut(r1):
print("Is CUDA enabled", IsCUDAEnabled());
print("timing", t1):

EnableCUDA(false):
t2 := time():
r3 := MultivariateResultant(R, fm, gm, p):
t2 := time() - t2:
r4 := PolynomialConvertOut(r3):
print("Is CUDA enabled", IsCUDAEnabled());
print("timing", t2):

if dx <= 50 then
r0 := Resultant(f, g, v) mod p:
end if;

print("speedup", t2 / t1);
print("result is correct", evalb(`mod`(r2 - r4, p)=0));
if dx <= 50 then
print("result is correct", evalb(`mod`(r0 - r2, p)=0));
end if:

with(RegularChains):
with(FastArithmeticTools):
with(modpn):

## Setting the polynomial ring with a prime characteristic
p := 469762049:
vars := [x, y, z]:
v := vars[1]:
R := PolynomialRing(p, vars):

dx := 13;
dy := dx:
dz := dx:
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
print("cuda vs C, speedup", t2 / t1);
print("result is correct", evalb(`mod`(r2 - r4, p)=0));

if dx <= 8 then
    t0 := time():
    r0 := Resultant(f, g, v) mod p:
    t0 := time() - t0:
    print("cuda vs maple, speedup", t0 / t1);
    print("result is correct", evalb((Expand(r2 - r0) mod p) = 0));
end if:

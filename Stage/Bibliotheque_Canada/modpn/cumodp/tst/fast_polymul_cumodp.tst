#Test function:
#- TFTFFTMul

#test 150
with(RegularChains):
with(FastArithmeticTools):
with(modpn):
with(TestTools):

p := 469762049:
vars := [a, b, c, d]:
R := PolynomialRing(p, vars):
f := randpoly(vars, dense, degree = 10) mod p:
g := randpoly(vars, dense, degree = 10) mod p:

fm := PolynomialConvertIn(R, f):
gm := PolynomialConvertIn(R, g):
InFm := PartialDegsAssert(fm):
InGm := PartialDegsAssert(gm):
repFm := CRepAssert(InFm, 1):
repGm := CRepAssert(InGm, 1):

## compute the product
Try[testnoerror](10, TFTFFTMul(R, repFm, repGm, 1, p), 'assign'='hm'):
hmodpn := PolynomialConvertOut(hm):
hmaple := Expand(f * g) mod p:
hdiff := Expand(hmaple - hmodpn) mod p:

## Comparing the results
Try(20, evalb(hdiff = 0), true);

#end test;


#Test function:
#- BivariateSolve

#test 500
with(modpn):
with(TestTools):

## y > x in a modpn poly. ring
p := 962592769;
R := PolynomialRing(p, [y, x]): 
F := randpoly([x, y], degree=60, dense) mod p:
G := randpoly([x, y], degree=60, dense) mod p:
## solve the bivariate system
EnableCUDA(false);
Try[testnoerror](10, BivariateSolve(F, G, R), 'assign'='dec1'):
EnableCUDA(true);
Try[testnoerror](20, BivariateSolve(F, G, R), 'assign'='dec2'):
Try(30, evalb(length(dec1)=length(dec2)), true);
#end test;

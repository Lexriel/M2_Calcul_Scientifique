## Compute the saturate ideal of a bivariate regular chain
sat_ideal2 := proc(T, R) 
local p, K, y;
    p := R:-prime;   
    y := R:-VarList[1];
    K := PolynomialIdeals:-PolynomialIdeal({op(T)}, characteristic=p):
    if (nops(T) = 1) and degree(T[1], y) > 0 then
        K := PolynomialIdeals:-Saturate(K, lcoeff(T[1], y));
    end if;
    return K;
end proc:

intersect_list_ideals := proc(L) 
local n, K1, K2;
    n := nops(L);
    if (n=1) then return L[1]; end if;
    K1 := intersect_list_ideals(L[1..floor(n/2)]);
    K2 := intersect_list_ideals(L[floor(n/2)+1..n]);
    return PolynomialIdeals:-Intersect(K1, K2);
end proc:

with(modpn):
p := 962592769;
## y > x in a modpn poly. ring
R := PolynomialRing(p, [y, x]); 
F := randpoly([x, y], degree=60, dense) mod p:
G := randpoly([x, y], degree=60, dense) mod p:

EnableCUDA(false);
t1 := time():
dec1 := BivariateSolve(F, G, R):
t1 := time() - t1; 
EnableCUDA(true);
t2 := time():
dec2 := BivariateSolve(F, G, R):
t2 := time() - t2;

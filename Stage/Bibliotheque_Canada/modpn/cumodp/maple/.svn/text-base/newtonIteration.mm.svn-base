newtonIteration := proc(f, i, p)
	local g, j:
	g := 1:
	for j from 1 by 1 to i do 
		
		g := Rem( (2*g - f*g*g) , x^(2^j), x) mod p:
	od:
	return g:
end proc:

p := 7:
#for k from 2 by 1 to 10 do
	d := 2^4:
	A :=  RandomTools[Generate](polynom(integer(range = -p..p), x, degree = d)):
	const := (Rem(A,x,x) mod p):
	A := A - const:
	A := A +1:

	B := RandomTools[Generate](polynom(integer(range = -p..p), x, degree = d)):
	const := (Rem(B,x,x) mod p):
	B := B - const:
	B := B +1:

	C := Expand(A*B)mod p:

	i := ceil(log[2](d)):

	Ainv := newtonIteration(A, i, p);
	AinvExtend := Rem( (2*Ainv - A*Ainv*Ainv) , x^(2^(i+1)), x) mod p;

	Binv := newtonIteration(B, i, p);
	BinvExtend := Rem( (2*Binv - B*Binv*Binv) , x^(2^(i+1)), x) mod p;

	Cinv := newtonIteration(C, i+1, p);
	AinvBinvExt := Rem(AinvExtend*BinvExtend, x^(2^(i+1)),x)mod p;

	if Cinv <> AinvBinvExt 
		then print(false)
		else print(true)
	end if;
#od:








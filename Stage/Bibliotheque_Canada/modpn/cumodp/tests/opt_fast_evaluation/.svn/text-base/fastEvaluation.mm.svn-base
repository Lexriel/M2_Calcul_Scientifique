newtonIteration := proc(f, i, p)
	local g, j:
	g := 1:
	for j from 1 by 1 to i do 		
		g := Rem( (2*g - f*g*g) , x^(2^j), x) mod p:
	od:
	return g:
end proc:

printPolyFile := proc(p, x, d, fd)
	local j:
	for j from 0 by 1 to d do 		
		fprintf(fd, "%d ", coeff(p, x, j)  ):
	od:
end proc:


fd := fopen("KP.dat", READ):
X := readdata(fd, [integer, integer]):
fclose(fd):


w := X[1][1]:
p := X[1][2]:

k := w+1:
plainMulLimit := 8:


Points[]:
fd := fopen("Points.dat", WRITE):
for j from 1 by 1 to 2^(k-1) do 
	Points[j] := modp(rand(),p):
	fprintf(fd, "%d ", Points[j]):
od:
fclose(fd):


F[][]:
F[1][1] :=  RandomTools[Generate](polynom( integer(range = -0..(p-1))  , x, degree =  2^(k-1)-1)):
#F[1][1] :=  RandomTools[Generate](polynom( integer(range = 0..(p-1))  , x, degree =  2^(k-1)-1)):


Value[]:
fd := fopen("value.dat", WRITE):
for j from 1 by 1 to 2^(k-1) do 
	Value[j] :=  Eval(F[1][1], x= Points[j] )mod p:
	fprintf(fd, "%d ", Value[j]):
od:
fclose(fd):


M[][]:
InV[][]:

fdM := fopen("PolyM.dat", WRITE):
fdMinv := fopen("PolyMinv.dat", WRITE):


for j from 1 by 1 to 2^(k-1) do 
	M[1][j] := x - Points[j]:
	Inv[1][j] :=  1:
	printPolyFile(M[1][j], x, 1  , fdM):
	#printPolyFile(Inv[1][j], x, 0  , fdMinv):
	fprintf(fdM, "\n"):
 	#fprintf(fdMinv, "\n"):	 				
od:


for j from 2 by 1 to (k-1) do
	for l from 1 by 1 to 2^(k-j) do  
		M[j][l] := Expand( M[j-1][2*l-1]* M[j-1][2*l] ) mod p:
		Inv[j][l] := eval(M[j][l], x = 1/x):

		Inv[j][l] := expand(Inv[j][l]*x^(2^(j-1))):
		Inv[j][l] := newtonIteration(Inv[j][l], j-1 ,p):
		
		printPolyFile(M[j][l], x, 2^(j-1), fdM):
		if (j-1) >= plainMulLimit then 
			printPolyFile(Inv[j][l], x, 2^(j-1)-1  , fdMinv):
			fprintf(fdMinv, "\n"):
		end if;

		fprintf(fdM, "\n"): 

	od:
od:
fclose(fdM):
fclose(fdMinv):



fd := fopen("PolyF.dat", WRITE):
printPolyFile(F[1][1], x, 2^(k-1)-1  , fd):
for j from 2 by 1 to k do
	for l from 1 by 1 to 2^(j-1) do
		F[j][l] := Rem(F[j-1][ceil(l*0.5)], M[k-j+1][l], x) mod p:
		fprintf(fd, "\n"):
		printPolyFile(F[j][l], x, 2^(k-j)-1  , fd):
	od:
od:

fclose(fd):


for j from 1 by 1 to 2^(k-1) do 
	if Value[j] <> F[k][j] then print(false)
	#else print(true)
	end if;
od:


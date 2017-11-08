printPolyFile := proc(p, x, d, fd)
	local j:
	for j from 0 by 1 to d do 		
		fprintf(fd, "%d ", coeff(p, x, j)  ):
	od:
end proc:



fd := fopen("PNM.dat", READ):
X := readdata(fd, [integer, integer, integer]):
fclose(fd):


p := X[1][1]:
n := X[1][2]:
m := X[1][3]:





fd := fopen("Poly1.dat", WRITE):
A:
A :=  RandomTools[Generate](polynom( integer(range = -0..(p-1))  , x, degree =  (n-1) )):
printPolyFile(A, x, (n-1), fd):
fclose(fd):

fd := fopen("Poly2.dat", WRITE):
B:
B :=  RandomTools[Generate](polynom( integer(range = -0..(p-1))  , x, degree =  (m-1) )):
printPolyFile(B, x, (m-1), fd):
fclose(fd):

fd := fopen("Rem.dat", WRITE):
R:
R := Rem(A, B, x) mod p:
printPolyFile(R, x, (m-2), fd):
fclose(fd):


fd := fopen("Quo.dat", WRITE):
Q:
Q := Quo(A, B, x) mod p:
printPolyFile(Q, x, (n-m), fd):
fclose(fd):




printPolyFile := proc(p, x, d, fd)
	local j:
	for j from 0 by 1 to d do 		
		fprintf(fd, "%d ", coeff(p, x, j)  ):
	od:
end proc:


fd := fopen("PNM.dat", READ):
X := readdata(fd, [integer, integer, integer, integer]):
fclose(fd):


p := X[1][1]:
n := X[1][2]:
m := X[1][3]:
h := X[1][4]:

R :=  RandomTools[Generate](polynom( integer(range = -0..(p-1))  , x, degree =  (h-1) )):

fd := fopen("Poly1.dat", WRITE):
Atemp:
Atemp :=  RandomTools[Generate](polynom( integer(range = -0..(p-1))  , x, degree =  (n-h) )):
A:
A := Expand( Atemp*R ) mod p:

printPolyFile(A, x, (n-1), fd):
fclose(fd):

fd := fopen("Poly2.dat", WRITE):
Btemp:
Btemp :=  RandomTools[Generate](polynom( integer(range = -0..(p-1))  , x, degree =  (m-h) )):
B:
B :=  Expand( Btemp*R ) mod p:
printPolyFile(B, x, (m-1), fd):
fclose(fd):

fd := fopen("GCD.dat", WRITE):
Q:
Q := Gcd(A, B) mod p:
printPolyFile(Q, x, m-1, fd):
fclose(fd):








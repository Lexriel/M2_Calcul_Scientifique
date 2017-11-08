#include<iostream>
#include <ctime>
#include<cmath>
#include<fstream>
#include "cudaGcd.h"
#include "cumodp_simple.h"
#include "defines.h"
#include "rdr_poly.h"
#include "inlines.h"
#include "printing.h"
#include "cudautils.h"
#include "types.h"



using namespace std;

sfixn add_modCPU(sfixn a, sfixn b, sfixn P) {
    sfixn r = a + b;
    r -= P;
    r += (r >> BASE_1) & P;
    return r;
}

sfixn sub_modCPU(sfixn a, sfixn b, int P) {
    sfixn r = a - b;
    r += (r >> BASE_1) & P;
    return r;
}



sfixn neg_modCPU(sfixn a, int P) {
    sfixn r = - a;
    r += (r >> BASE_1) & P;
    return r;
}


void egcdCPU(sfixn x, sfixn y, sfixn *ao, sfixn *bo, sfixn *vo, int P) {
    sfixn t, A, B, C, D, u, v, q;

    u = y; v = x;
    A = 1; B = 0;
    C = 0; D = 1;

    do {
        q = u / v;
        t = u;
        u = v;
        v = t - q * v;
        t = A;
        A = B;
        B = t - q * B;
        t = C;
        C = D;
        D = t - q * D;
    } while (v != 0);

    *ao = A;
    *bo = C;
    *vo = u;
}




sfixn inv_modCPU(sfixn n, int P) {
    sfixn a, b, v;
    egcdCPU(n,  P, &a, &b, &v, P);
    if (b < 0) b += P;
    return b % P;
}




sfixn mul_modCPU(sfixn a, sfixn b, int P) {
    double ninv = 1 / (double)P;
    sfixn q  = (sfixn) ((((double) a) * ((double) b)) * ninv);
    sfixn res = a * b - q * P;
    res += (res >> BASE_1) & P;
    res -= 	P;
    res += (res >> BASE_1) & P;
    return res;
}


sfixn pow_modCPU(sfixn a, sfixn ee, int P) {
    sfixn x, y;
    usfixn e;

    if (ee == 0) return 1;
    if (ee == 1) return a;
    if (ee < 0) e = - ((usfixn) ee); else e = ee;

    x = 1;
    y = a;
    while (e) {
        if (e & 1) x = mul_modCPU(x, y,P);
        y = mul_modCPU(y, y,P);
        e = e >> 1;
    }

    if (ee < 0) x = inv_modCPU(x,P);
    return x;
}

		


int main(int argc, char *argv[])
{
	int n = 10, m = 5;// h = 3, 
	int i, j, k;
	sfixn p = 7;

	if (argc > 1) p = atoi(argv[1]);
        if (argc > 2) n = atoi(argv[2]);
	if (argc > 3) m = atoi(argv[3]);
       // if (argc > 4) h = atoi(argv[4]);
	
	if(m > n)
	{
		i = m;
		m = n;
		n = i;
	}
	
	/*
	ofstream ofs1( "PNM.dat", ofstream::out  );	
	ofs1<<p<<" "<<n<<" "<<m<<" "<<h;
        ofs1.close();	
	*/

	//i = system("maple -q gcd.mm");
	///*
	sfixn *A = new int[n];
	sfixn *B = new int[m];
	sfixn *G = new int[m];
	
	for(i = 0; i < n; ++i)	
		A[i] =  rand()% p;
	for(i = 0; i < m; ++i)	
		B[i] =  rand()% p;


	
	for(i = 0; i < m; ++i)
		G[i] = 0;
	/*
	ifstream ifs1( "Poly1.dat", ifstream::in  ); 
	for(i = 0; i < n; ++i)	
		ifs1>>A[i];
	ifs1.close();

	ifstream ifs2( "Poly2.dat", ifstream::in  ); 
	for(i = 0; i < m; ++i)	
		ifs2>>B[i];
	ifs2.close();		
	
	*/
	//cout<<endl;
	//for(i = 0; i < n; ++i)
	//	cout<<A[i]<<" ";
	//cout<<endl;


	//for(i = 0; i < m; ++i)
	//	cout<<B[i]<<" ";
	//cout<<endl;

	float outerTime = gcdPrimeField(A, B, G, n, m, p);
	cout<<n<<" "<<m<<" "<<outerTime/1000.0<<endl;
	
	//cout<<endl;
	/*
	for(i = m-1; i >= 0; --i)
	{
		if(G[i] != 0)
			break;
	}
	if(i <= 0)
	{
		G[0] = 1;
	}
	else
	{
		k = inv_modCPU( G[i], p);	
		for(j = 0; j <= i; ++j)
			G[j] = mul_modCPU(G[j], k, p);
		
	}

	ofstream ofs2( "GCDcuda.dat", ofstream::out  );	
	for(i = 0; i < m; ++i)
		ofs2<<G[i]<<" ";
        ofs2.close();	

	i = system("diff -b GCD.dat GCDcuda.dat");

	*/	
	
	//cout<<endl;
	
	delete [] A;
	delete [] B;
	delete [] G;
	//*/
	return 0;
}






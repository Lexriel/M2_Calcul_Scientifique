#include<iostream>
#include <ctime>
#include<cmath>
#include<fstream>
#include "cudaDiv.h"
#include "cumodp_simple.h"
#include "defines.h"
#include "rdr_poly.h"
#include "inlines.h"
#include "printing.h"
#include "cudautils.h"
#include "types.h"


using namespace std;



int main(int argc, char *argv[])
{
	int n = 10, m = 5, i;
	

	sfixn p = 7;
	/*
	ifstream ifs( "PNM.dat", ifstream::in  ); 
	ifs>>p;
	ifs>>n;
	ifs>>m;
	ifs.close();
	*/
	 if(argc > 1) n = atoi(argv[1]);
	 if(argc > 2) m = atoi(argv[2]);
	 if(argc > 3) p = atoi(argv[3]);

	

	sfixn *A = new sfixn[n];
	sfixn *B = new sfixn[m];

	for(i = 0; i < n; ++i)	
		A[i] =  rand()% p;
	for(i = 0; i < m; ++i)	
		B[i] =  rand()% p;
	
	/*
	ifstream ifs1( "Poly1.dat", ifstream::in  ); 
	
	ifs1.close();

	ifstream ifs2( "Poly2.dat", ifstream::in  ); 
	
	ifs2.close();
	*/

	sfixn *Q = new sfixn [ n-m +1 ];
	sfixn *R = new sfixn [ m - 1 ];
	float outerTime = divPrimeField(A, B, R, Q, n, m, p);
	/*

	ofstream ofsQ( "QuoGPU.dat", ofstream::out  );
	for(i = 0;  i < (n - m + 1); ++i)
		ofsQ<<Q[i]<<" ";
        ofsQ.close();


	ofstream ofs( "RemGPU.dat", ofstream::out  );
	for(i = 0;  i < (m - 1); ++i)
		ofs<<R[i]<<" ";
        ofs.close();
	*/
	cout<<n<<" "<<m<<" "<<outerTime/1000.0<<endl;

	
	
	
	delete [] A;
	delete [] B;
	delete [] Q;
	delete [] R;
	return 0;
}






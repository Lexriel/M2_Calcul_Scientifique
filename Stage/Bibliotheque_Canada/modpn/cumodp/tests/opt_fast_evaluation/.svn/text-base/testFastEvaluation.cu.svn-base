#include<iostream>
#include <ctime>
#include<cmath>
#include<fstream>
#include "fastPolyEvaluation.h"
#include "cumodp_simple.h"
#include "defines.h"
#include "rdr_poly.h"
#include "inlines.h"
#include "printing.h"
#include "cudautils.h"
#include "types.h"

using namespace std;

	
void getFpoly(sfixn *F, int numPoints)
{
	ifstream ifs( "PolyF.dat", ifstream::in  ); 
	for(int i = 0; i < numPoints; ++i)
		ifs>>F[i];
	ifs.close();
}

void getPoints(sfixn *M,  int numPoints)
{
	ifstream ifs( "Points.dat", ifstream::in  ); 	
	for(int i = 0; i < numPoints; ++i)
		ifs>>M[i];	
	ifs.close();	
}


void printSubInvTree(struct PolyEvalSteps check, int k)
{
	ofstream ofs( "PolyMinvMgpu.dat", ofstream::out  );
	int j, length, no;
	int i = plainMulLimit;
	length = 1 << (i);
	no = 1 << (k-i-1);
	sfixn *T = new sfixn[length*no ];
	for(; i < k; ++i )
	{
		length = 1 << (i);
		no = 1 << (k-i-1);

		cudaMemcpy(T, check.InvMl[i], sizeof(sfixn)*no*length, cudaMemcpyDeviceToHost);
		for(j = 0; j < length*no ; ++j)
                {
                        ofs<<T[j]<<" ";
                        if(j%length == (length-1))
                        	ofs<<endl;                        
                }
		cudaMemcpy(T, check.InvMr[i], sizeof(sfixn)*no*length, cudaMemcpyDeviceToHost);
		for(j = 0; j < length*no ; ++j)
                {
                        ofs<<T[j]<<" ";
                        if(j%length == (length-1))
                        	ofs<<endl;                        
                }
	}
	delete [] T;
	ofs.close();	
}

void printSubProductTree(struct PolyEvalSteps check, int k)
{
	ofstream ofs( "PolyMgpu.dat", ofstream::out  );


	int i, j, l, length, no;
	length = 2;
	no = 1 << (k-1);		
	sfixn *T = new sfixn[length*no ];
	for(i = 0; i < plainMulLimit; ++i )
	{
		cudaMemcpy(T, check.Ml[i], sizeof(sfixn)*no*length, cudaMemcpyDeviceToHost);
		for(j = 0; j < no; ++j)
		{
			for(l = 0; l < length; ++l)
			{
				ofs<<T[j*length + l]<<" ";
			}
			ofs<<endl;
		}

		cudaMemcpy(T, check.Mr[i], sizeof(sfixn)*no*length, cudaMemcpyDeviceToHost);
		for(j = 0; j < no; ++j)
		{
			for(l = 0; l < length; ++l)
			{
				ofs<<T[j*length + l]<<" ";
			}
			ofs<<endl;
		}	
		
		no = no/2;
		length = 2*length - 1; 
		
	}
	length = 1 << (i);
	no = 1 << (k-i-1);
	for(; i < k; ++i )
	{
		length = 1 << (i);
		no = 1 << (k-i-1);
		cudaMemcpy(T, check.Ml[i], sizeof(sfixn)*no*length, cudaMemcpyDeviceToHost);
		for(j = 0; j < no; ++j)
		{
			for(l = 0; l < length; ++l)
			{
				ofs<<T[j*length + l]<<" ";
			}
			ofs<<"1"<<endl;
		}
		cudaMemcpy(T, check.Mr[i], sizeof(sfixn)*no*length, cudaMemcpyDeviceToHost);
		for(j = 0; j < no; ++j)
		{
			for(l = 0; l < length; ++l)
			{
				ofs<<T[j*length + l]<<" ";
			}
			ofs<<"1"<<endl;
		}	
	}	
	delete [] T;
	ofs.close();	
}
	


int main(int argc, char *argv[])
{
	/*
	We are working on prime filed (mod p).
	We have 2^k number of points to evaluate.
	*/
	int k = 10;
	sfixn p = 469762049;
	int verify = 1;
	if (argc > 1) k = atoi(argv[1]);
        if (argc > 2) p = atoi(argv[2]);
        if (argc > 3) verify = atoi(argv[2]);		
	int numPoints = 1 << k;

	// Writing the value of k and p into a file	
	ofstream ofs1( "KP.dat", ofstream::out  );	
	ofs1<<k<<" "<<p;
        ofs1.close();	

	/*
	First call maple to create 
		i)   random 2^k points x_1, x_2,..x_{2^k}. Each x_i in field of prime p
		ii)  a random polynomial F of length 2^k over finite field of prime p. 
		iii) subproduct tree with  (X-x_1)..(X-x_{2^k})
		iv)  subinverse tree of the subproduct tree
		v)   Fast evaluation
	*/
	int i = system("maple -q fastEvaluation.mm");
	
	sfixn *points = new sfixn[numPoints];
	sfixn *F = new sfixn[numPoints];
	getPoints(points, numPoints); // getting the values of x for which we need to evaluate F.
	getFpoly(F, numPoints); // getting the polynomial F to be evaluated.	
	
	time_t start,end;
	time (&start);
	struct PolyEvalSteps  check = fastEvaluation(F, points, k, p, verify);	
	// the last parameter is a flag. 
	// 1 means we need the output at each level of subproduct tree, subinverse tree and evaluation.
	// 0 means we will need only the evaluation.
	// points will have the evaluation value for F.
	time (&end);
	double dif = difftime (end,start);
	cout<<"It took "<<dif<<" seconds"<<endl;

	
	

	if(verify == 1)
	{
		printSubProductTree(check,  k);
		i = system("diff -b PolyM.dat PolyMgpu.dat");
		printSubInvTree(check,  k);
		i = system("diff -b PolyMinv.dat PolyMinvMgpu.dat");
	}
	delete [] points;
	delete [] F;
	return 0;
}


/*
int main(int argc, char *argv[])
{
	
	int i;	
	
	
        sfixn *M2;
        sfixn *F;


	if( mapleCheck == 0)
	{
		 numPoints = 1 << (k-1);
	         M1 = new sfixn[numPoints];
        	 M2 = new sfixn[numPoints];
		 F  = new sfixn[numPoints];
		 

        	numPoints = numPoints/2;
        	for(i = 0; i < numPoints; ++i)
        	{
                	//ifs>>M1[2*i];
                	M1[2*i] =  -(rand()% p);
                	M1[2*i+1] = 1;
        	}
        	for(i = 0; i < numPoints; ++i)
        	{
                	//ifs>>M2[2*i];
                	M2[2*i] = -(rand()% p);
                	M2[2*i+1] = 1;
        	}
		numPoints = 1 << (k-1);
		for(i = 0; i < numPoints; ++i)
			F[i] = (rand()% p);
			
	}
        else
	{
		getPrimeLevel(&p, &k);
		numPoints = 1 << (k-1);
                M1 = new sfixn[numPoints];
                M2 = new sfixn[numPoints];
		F  = new sfixn[numPoints];

		getPoints(M1, M2, numPoints);
		getFpoly(F, numPoints);
	
	}

	
	struct status left, right;

	 left =  fastEvaluation(k, p, M1, F, checkResult);

	right =  fastEvaluation(k, p, M2, F, checkResult);
	
	if(checkResult % 2 == 1)
		printSubProductTree(left, right, k);

	if(checkResult  == 2 ||checkResult  == 3 ||checkResult >= 6)
                printInverseTree(left, right, k);


	for(i = 0; i < (k-1); ++i)
	{
		delete [] left.M[i];
		delete [] right.M[i];
	}
        for(i = plainMulLimit; i < (k-1); ++i)
        {
                delete [] left.InvM[i];
                delete [] right.InvM[i];
        }


	delete [] M1;
	delete [] M2;
	delete [] F;
	

	return 0;
}


void printInverseTree(struct status left, struct status right, int k)
{
	ofstream ofs( "../maple/PolyInvMcuda.dat", ofstream::out  );
        int i, j, length, no;
        sfixn *T;
        for(i = plainMulLimit; i < (k-1); ++i )
        {
                length = 1 << (i);
                no = 1 << (k-i-2);

                T = left.InvM[i];
                for(j = 0; j < length*no ; ++j)
                {
                        ofs<<T[j]<<" ";
                        if(j%length == (length-1))
                        	ofs<<endl;                        
                }
                T = right.InvM[i];
                for(j = 0; j <length*no ; ++j)
                {
                        ofs<<T[j]<<" ";
                        if(j%length == (length-1))
                        	ofs<<endl;                        
                }
        }
        ofs.close();
}


void printSubProductTree(struct status left, struct status right, int k)
{

	ofstream ofs( "../maple/subproductCuda.dat", ofstream::out  );
	int i, j, length, no;
	length = 2;
	no = 1 << (k-2);		
	sfixn *T;
	for(i = 0; i < (k-1); ++i )
	{
		length = 1 << (i);
		no = 1 << (k-i-2);
		if(i < plainMulLimit) ++length;

		T = left.M[i];
		for(j = 0; j < length*no ; ++j)
		{
			ofs<<T[j]<<" ";
			if(j%length == (length-1)) 
			{
				if(i >= plainMulLimit) ofs<<1<<endl;
				else ofs<<endl;
			}
		}
		T = right.M[i];
		for(j = 0; j <length*no ; ++j)
		{
			ofs<<T[j]<<" ";
			if(j%length == (length-1)) 
			{
				if(i >= plainMulLimit) ofs<<1<<endl;
				else ofs<<endl;
			}
		}				
	}		
	ofs.close();	
}

void printGPUArray(sfixn *A, int l)
{
	sfixn *B = new sfixn [l];
	int i;
	cudaMemcpy(B, A, sizeof(sfixn)*l, cudaMemcpyDeviceToHost);
	cout<<endl;
	for(i =0; i < l; ++i)
		cout<<B[i]<<" ";
	cout<<endl;
	delete [] B;
}



*/





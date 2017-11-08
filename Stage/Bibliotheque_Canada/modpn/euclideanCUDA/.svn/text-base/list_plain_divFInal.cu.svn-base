/*******************************************************************
* This code is intended for for computing remainder i.e f rem m    *
* The degree of m is T*W-1 that is 511 and the degree of f is 1020
********************************************************************/

#include<iostream>
#include <ctime>
#include<cmath>

#define BASE_1 31

const int T = 128;
/***********************************************************
* one thread is responsible for computing "W" coefficients *
***********************************************************/
const int W = 4; 



typedef int            sfixn;
typedef unsigned int   usfixn;

using namespace std;


__device__ __host__ __inline__ 
void egcd(sfixn x, sfixn y, sfixn *ao, sfixn *bo, sfixn *vo, int P) {
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



__device__ __host__ __inline__ 
sfixn inv_mod(sfixn n, int P) {
    sfixn a, b, v;
    egcd(n,  P, &a, &b, &v, P);
    if (b < 0) b += P;
    return b % P;
}

__device__ __host__ __inline__ 
sfixn mul_mod(sfixn a, sfixn b, int P) {
    double ninv = 1 / (double)P;
    sfixn q  = (sfixn) ((((double) a) * ((double) b)) * ninv);
    sfixn res = a * b - q * P;
    res += (res >> BASE_1) & P;
    res -= 	P;
    res += (res >> BASE_1) & P;
    return res;
}

__device__ __host__ __inline__ 
sfixn sub_mod(sfixn a, sfixn b, int P) {
    sfixn r = a - b;
    r += (r >> BASE_1) & P;
    return r;
}


__global__ void list_divCUDA(sfixn *M, sfixn *F, int start, int length, int threadsPerDiv, int DivPerBlock, int polyNum, int P)
{
	__shared__ sfixn sM[T*W];
	__shared__ sfixn sF[T*2*W];
	__shared__ sfixn invM[T];
	__shared__ sfixn oneF[T];
	__shared__ int mID[T];
	__shared__ int fID[T];
	__shared__ int startF[T];
	
	int i, j, k, l, s, t, polyID;
	polyID = ((threadIdx.x/threadsPerDiv) + blockIdx.x*DivPerBlock);
	if( polyID < polyNum )
	{
		if( (threadIdx.x %threadsPerDiv) == (threadsPerDiv-1))
		{
			i = threadIdx.x;				
			mID[i] = (i/ threadsPerDiv) + (blockIdx.x*DivPerBlock);
			fID[i] = mID[i]/ 2;
	
			mID[i] = start + (mID[i]+1)*length -1;
			fID[i] = (fID[i]+1)*(2*length-2) - 1;
			invM[i] = inv_mod(M[mID[i]], P);			
		}
		else
		{
			i = threadIdx.x/threadsPerDiv;
			i = (i*threadsPerDiv) + threadsPerDiv -1;
		}
		__syncthreads();
		
		j = threadIdx.x;
		k = i - j;
		t = W-1;
		l = mID[i] - k*W;
		s = l - W;
		for(; l > s && l >= 0; --l, --t ) 
			sM[j*W + t] = M[l];	
	
		l = fID[i] - (k*W*2);
		s = l - 2*W;
		t = 2*W-1;	
		for(; l > s && l >= 0; --l, --t ) 
			sF[j*2*W +t ] = F[l];

		if(i == j)
		{
			fID[i] = 2*i*W + 2*W -1;
			mID[i] = fID[i] - length + 1;
			while( sF[ fID[ i ] ] == 0 &&  fID[ i ] > mID[i] ) --fID[ i ];

			oneF[i] = mul_mod(invM[i], sF[fID[i]], P);			
		}
		
		__syncthreads();


		//int temp1 =0 , temp2 = 4;
		while(fID[ i ] > mID[i]) //while(temp1 < temp2)
		{
			//--temp2;

			l = fID[ i ] - k*W;
			s = l - W;
			if( (fID[ i ] - length) > s )	s = fID[ i ] - length;
			t = i*W + W -1 -k*W;
			for(; l > s; --l, --t)
				sF[l] = sub_mod(sF[l], mul_mod(sM[t], oneF[i], P), P);


			if(i == j)
			{
				--fID[ i ];
				while( sF[ fID[ i ] ] == 0 &&  fID[ i ] > mID[i] ) --fID[ i ];
				oneF[i] = mul_mod(invM[i], sF[fID[i]], P);
			
			}
			__syncthreads();
		}

		if(i == j)
		{
			fID[i] = mID[i] - length +1;
			startF[i] =  ( (polyID/2) +1)*(2*length-2) - 1;
			
			if(polyID%2 == 0)
				startF[i] = startF[i] - length +1;
			
		}
		__syncthreads();

	


		

		l = mID[ i ] - k*W;
		t = l - W;
		s =  startF[i] - k*W;
		if(t < fID[i]) t = fID[i];
		for(; l >  t; --l, --s)
			F[s] = sF[l];
		
			
	}		
}

/*
//check to copy one elemnt before destroying
		//	if(j != T-1)
	//	k = sF[j+1][2*W-1]

//

	if(j<10){
		F[j*18 ] = j;
		F[j*18 + 1] = i;
		F[j*18 + 2] = invM[i];
		F[j*18 + 3] = oneF[i];
		F[j*18 + 4] = mID[i];
		F[j*18 + 5] = fID[i];
		F[j*18 + 6] = sM[j*W];
		F[j*18 + 7] = sM[j*W+ 1];
		F[j*18 + 8] = sM[j*W+ 2];
		F[j*18 + 9] = sM[j*W+ 3];
		F[j*18 + 10] = sF[j*2*W];
		F[j*18 + 11] = sF[j*2*W+ 1];
		F[j*18 + 12] = sF[j*2*W+ 2];
		F[j*18 + 13] = sF[j*2*W+ 3];
		F[j*18 + 14] = sF[j*2*W+ 4];
		F[j*18 + 15] = sF[j*2*W+ 5];
		F[j*18 + 16] = sF[j*2*W+ 6];
		F[j*18 + 17] = sF[j*2*W+ 7];
		
		}


if(k%2 == 0)
	{
		for(l = 1; l < W; ++l)
			sF[j][l-1] = sub_mod(sF[j][l],mul_mod(sM[j][l], oneF[i], P), P);
	}
	else
	{
	}
	for(l = 0; l < length-1; ++l)				
	{
		for(t = W-2, s = 2*W-2; t >=0; --t, --s)
			sF[s+1] = sub_mod(sM[t], mul_mod(sF[s], oneF[i], P), P);
		
		if(j == i)
			oneF[i] =  mul_mod(invM[i], sF[2*W-1], P);	
		__syncthreads();
	}

		if((k+1)*W > length)
		{
			l = (k+1)*W -length;
			for(s = 0; s < l; ++s)
				sM[j* +s] = 0;
		} 

	if( ((k+1)*2*W) > (2*length-2))
		{
			l = (k+1)*2*W -(2*length-2);
			for(s = 0; s < l; ++s)
				sF[j][s] = 0;
		}


*/


void divCPU(sfixn *M, sfixn *F, int n, int length, int start, int P)
{
	cout<<"r := Rem(";

	int t;
	int k = ((n/2) + 1)*(2*length-2) -1;
	int l = k - (2*length -2);
	k = k+1;
	l = l+1;
	
	cout<<F[l];
	l=l+1;
	t =1;
	for(; l<k; ++l,++t)
		cout<<"+"<<F[l]<<"*x^"<<t;
	cout<<",";


	int i = start + (n+1)*length -1;
	int j = i - length;
	j = j +1;
	i = i+1;

	
	cout<<M[j];
	j = j +1;

	t=1;
	for(; j < i; ++j, ++t)
		cout<<"+"<<M[j]<<"*x^"<<t;
	
	
	
	cout<<",x) mod "<<P<<";"<<endl;

}

void list_div(sfixn *M, sfixn *F, int n, int m, int  start_offset, int length_poly, int poly_on_layer, int p )
{

	/**************************************************
	* Number of threads responsible for one division  *
	**************************************************/	
	int threadsForAdiv = (int)ceil((double)length_poly / (double)W);

	/******************************************
	* Number of divisions in one thread block *
	******************************************/
	int divInThreadBlock = (int)floor((double)T/(double)threadsForAdiv);

	/****************************
	* Number of blocks required *
	****************************/
	int blockNo = (int)ceil((double)poly_on_layer/(double) divInThreadBlock );


	cout<<"threadsForAdiv: "<<threadsForAdiv<<" divInThreadBlock: "<<divInThreadBlock<<" blockNo: " <<blockNo<<endl;

	sfixn *Mgpu, *Fgpu;

	cudaMalloc((void **)&Mgpu, sizeof(sfixn)*n);
	cudaMalloc((void **)&Fgpu, sizeof(sfixn)*m);
        cudaMemcpy(Mgpu, M, sizeof(int)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(Fgpu, F, sizeof(int)*m, cudaMemcpyHostToDevice);
	
	int Ftemp[1000];
	list_divCUDA<<<blockNo, T>>>(Mgpu, Fgpu, start_offset, length_poly, threadsForAdiv, divInThreadBlock, poly_on_layer, p);	

	cudaMemcpy(Ftemp, Fgpu, sizeof(int)*1000, cudaMemcpyDeviceToHost);
	cout<<endl<<"F: "<<endl;
	int i, j, k = 0; 	
       for(j=0; j < poly_on_layer; ++j) // for(j=0; j < threadsForAdiv; ++j)
	{
		cout<<endl;
		for(i = 0; i < (length_poly - 1); ++i)//for(i = 0; i < 18; ++i)
			cout<<Ftemp[k++]<<" ";		
		
	}
	cout<<endl;
	
	for(int j=0; j < poly_on_layer; ++j)
	{
		divCPU(M, F, j, length_poly, start_offset,p);
	}
		
	

	cudaFree(Mgpu);
        cudaFree(Fgpu);
}



int main(int argc, char *argv[])
{
	int   p= 7;
	int start_offset = 0, length_poly = 4, poly_on_layer = 4;
	

	if (argc > 1) start_offset = atoi(argv[1]);
	if (argc > 2) length_poly = atoi(argv[2]);
        if (argc > 3) poly_on_layer = atoi(argv[3]);
	if (argc > 4) p = atoi(argv[4]);
	

	/**************************************
	* poly_on_layer MUST BE DIVISIBLE BY 2 *
	***************************************/
	int n = 1000, m = 1000, i;
	
	sfixn *M = new int[n];
	sfixn *F = new int[m];
	
	cout<<endl;
	for(i = 0; i < n; ++i)
	{	
		M[i] = rand()% p;
		if(i % length_poly == 0) cout<<endl;
		if( (i % length_poly) == (length_poly-1) ) 
		{
			if(M[i] == 0)
			{
				while(M[i] ==0)
					M[i] = rand()% p;
			
			}
		}
		cout<<M[i]<<" ";
	}
	cout<<endl;		
	for(i = 0; i < m; ++i)	
	{
		F[i] = rand()% p;
		if(i % (2*length_poly -2) == 0) cout<<endl;
		cout<<F[i]<<" ";
	}
	cout<<"done"<<endl;
		
	list_div(M ,F, n, m, start_offset, length_poly, poly_on_layer, p );
	
		

	delete [] M;
	delete [] F;
	
	
	return 0;
}
	
	

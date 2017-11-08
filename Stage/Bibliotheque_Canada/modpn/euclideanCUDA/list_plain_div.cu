#include<iostream>
#include <ctime>
#include<cmath>

#define BASE_1 31

const int T = 128;
/***********************************************************************
* one thread is responsible for computing "W" coefficients *
***********************************************************************/
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


__global__ void list_divCUDA(sfixn *M, sfixn *F, int start, int length, int threadsPerDiv, int DivPerBlock, int P)
{
	__shared__ sfixn sM[T][W];
	__shared__ sfixn sF[T][2*W];
	__shared__ sfixn invM[T];
	__shared__ sfixn oneF[T];
	__shared__ int mID[T];
	__shared__ int fID[T];
	
	int i, j, k, l, s, t;
	
	if( (threadIdx.x %threadsPerDiv) == (threadsPerDiv-1))
	{
		i = threadIdx.x;				
		mID[i] = (i/ threadsPerDiv) + (blockIdx.x*DivPerBlock);
		fID[i] = mID[i]/ 2;

		mID[i] = start + (mID[i]+1)*length -1;
		fID[i] = (fID[i]+1)*(2*length-2) - 1;
		invM[i] = inv_mod(M[mID[i]], P);
		oneF[i] = mul_mod(invM[i], F[fID[i]], P);
	}
	else
	{
		i = threadIdx.x;
		while( (i % threadsPerDiv) != (threadsPerDiv-1) ) ++i;		
	}
	__syncthreads();
	
	j = threadIdx.x;
	k = j - i;
	t = W-1;
	l = mID[i] - k*W;
	s = l - W;
	for(; l < s && l >= 0; --l, --t ) 
		sM[j][t] = M[ l];

	l = fID[i] - (k*W*2);
	s = l - 2*W;
	t = 2*W-1;	
	for(; l < s && l >= 0; --l, --t ) 
		sF[j][t ] = F[ l];
	//check to copy one elemnt before destroying
	//	if(j != T-1)
//	k = sF[j+1][2*W-1]
	__syncthreads();

	for(l = 0; l < length-2; ++l)				
	{
		for(t = W-2, s = 2*W-2; t >0; --t, --s)
			sF[s+1] = sub_mod(sM[t], mul_mod(sF[s], oneF[i], P), P);
		
		if(j == i)
			oneF[i] =  mul_mod(invM[i], sF[2*W-1], P);	
		__syncthreads();
	}
		
}

void list_div(sfixn *M, sfixn *F, int n, int m, int  start_offset, int length_poly, int poly_on_layer, int p )
{
	int length_polyF = 2*length_poly -1;
	int poly_on_layerF = poly_on_layer/2;

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

	cout<<"length_polyF: "<<length_polyF<<" poly_on_layerF: "<<poly_on_layerF<<endl;
	cout<<"threadsForAdiv: "<<threadsForAdiv<<" divInThreadBlock: "<<divInThreadBlock<<" blockNo: " <<blockNo<<endl;

	sfixn *Mgpu, *Fgpu;

	cudaMalloc((void **)&Mgpu, sizeof(sfixn)*n);
	cudaMalloc((void **)&Fgpu, sizeof(sfixn)*m);
        cudaMemcpy(Mgpu, M, sizeof(int)*n, cudaMemcpyHostToDevice);
	cudaMemcpy(Fgpu, F, sizeof(int)*m, cudaMemcpyHostToDevice);




	cudaFree(Mgpu);
        cudaFree(Fgpu);
}



int main(int argc, char *argv[])
{
	int BLOCK = 8,  p= 7;
	int start_offset = 0, length_poly = 4, poly_on_layer = 4;
	int length_polyF = 7, poly_on_layerF = 2;

	if (argc > 1) start_offset = atoi(argv[1]);
	if (argc > 2) length_poly = atoi(argv[2]);
        if (argc > 3) poly_on_layer = atoi(argv[3]);
	if (argc > 4) p = atoi(argv[4]);
	if (argc > 5) BLOCK = atoi(argv[5]);
	

	

	/**************************************
	* poly_on_layer MUST BE DIVISIBLE BY 2 *
	***************************************/
	int n =	65536, m = 65536, i;
	
	sfixn *M = new int[n];
	sfixn *F = new int[m];
	
	for(i = 0; i < n; ++i)	
		M[i] = rand()% p;
		
	for(i = 0; i < m; ++i)	
		F[i] = rand()% p;
	
	list_div(M ,F, n, m, start_offset, length_poly, poly_on_layer, p );
	
		

	delete [] M;
	delete [] F;
	
	
	return 0;
}
	
	

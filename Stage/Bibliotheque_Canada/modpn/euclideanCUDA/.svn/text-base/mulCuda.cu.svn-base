#include<iostream>
#include <ctime>
#include<cmath>

using namespace std;

const int BASE_1 = 31;
const int BLOCK = 128;
const int Tx = 16;
const int T = Tx*BLOCK+BLOCK-1;


typedef int            sfixn;
typedef unsigned int   usfixn;




sfixn add_modCPU(sfixn a, sfixn b, sfixn P) {
    sfixn r = a + b;
    r -= P;
    r += (r >> BASE_1) & P;
    return r;
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




__device__ __host__ __inline__ 
sfixn add_mod(sfixn a, sfixn b, sfixn P) {
    sfixn r = a + b;
    r -= P;
    r += (r >> BASE_1) & P;
    return r;
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

__global__ void initA_C(sfixn *A, sfixn *C, int n, int c)
{
	int i = (blockIdx.x*blockDim.x + threadIdx.x)*BLOCK;
	if(i < c)
	{
		int j, k, l, p;	
		p = n + BLOCK -1;
		
		if(i == 0)
		{
			k = BLOCK -1;
			for(j = 0; j < k; ++j)
				A[j] = 0;

			j = i + BLOCK;
			l = 0;			
			for(; l < j && k < p; ++k, ++l)
			{
				A[k] = C[l];
				C[l] = 0;			
			}

			k = n + 2*(BLOCK -1);
			j = n + BLOCK -1;
			for(; j < k; ++j)
				A[j] = 0;			
		}
		else
		{
			j = i + BLOCK;
			k = i + BLOCK -1;

			if(k < p)
			{
				for(; i < j && k < p; ++i, ++k)
				{
					A[k] = C[i];
					C[i] = 0;
				}
			}
			else
			{
				
				for(; i < j && i < c; ++i)
					C[i] = 0;
			}
		}		
		
	}
}


__global__ void mul(sfixn *A, sfixn *B, sfixn *C, int n, int m, int c, int P)
{
	__shared__ sfixn sA[T];
	__shared__ sfixn sB[BLOCK];
	sfixn Res[BLOCK];

	int i = (blockIdx.x*blockDim.x + threadIdx.x)*BLOCK;
	int j = i + BLOCK;
	int k = threadIdx.x*BLOCK;
	int l = k + BLOCK;
	
	for(; i < j && i < n; ++i, ++k) 
		sA[k] = A[i];
	for(; k < l ; ++k)
		sA[k] = 0;

	if(threadIdx.x == 0)
	{		
		for(i = 0; i < BLOCK && i < m; ++i)	sB[i] = B[i];
		for(; i < BLOCK; ++i) sB[i] = 0;
		j = (blockIdx.x*blockDim.x + Tx)*BLOCK;
		for(i = Tx*BLOCK; i < T && j < n; ++i, ++j )	sA[i] = A[j];
		for(; i < T; ++i) sA[i] = 0;
	}
	__syncthreads();

	l = threadIdx.x*BLOCK + BLOCK -1;
	int s = (blockIdx.x*blockDim.x + threadIdx.x)*BLOCK;
	for(i = 0; i < BLOCK; ++i) 
	{		
		Res[i] = 0;	
		k = l + i;		
		for(j = 0; j <BLOCK; ++j, --k)
		{

			Res[i] = add_mod(Res[i] , mul_mod(sA[k], sB[j], P), P);
			//Res[i] += sA[k]*sB[j];
		}
		C[s + i] = Res[i];
	}	
}


void mulCUDA( sfixn *A,  sfixn *B, int n, int m, int Prime)
{
	int i;
	sfixn *Agpu, *Bgpu, *Cgpu;

	if(m > n)	
	{
		i = m;
		m = n;
		n = i;
		
		Agpu = A;
		A = B;
		B = Agpu;	
	}
	
	cudaMalloc((void **)&Agpu, ( sizeof(sfixn)*(n + 2*(BLOCK-1)))  );
	cudaMalloc((void **)&Bgpu, sizeof(sfixn)*m);
	cudaMalloc((void **)&Cgpu, sizeof(sfixn)*(m+n-1));

        cudaMemcpy(Bgpu, B, sizeof(sfixn)*m, cudaMemcpyHostToDevice);
        cudaMemcpy(Cgpu, A, sizeof(sfixn)*n, cudaMemcpyHostToDevice);

        int total_blocksX = (int)ceil((double)(m+n-1) / (double)(BLOCK*Tx));
	//
	//cout<<"total_blocksX: "<<total_blocksX<<endl;
	//
	dim3 threadsPerBlock(Tx); 
	dim3 numBlocks(total_blocksX);
	
	initA_C<<<numBlocks, threadsPerBlock>>>(Agpu, Cgpu,  n, n+m-1);	
	int n1 = n + (BLOCK-1);
	total_blocksX = (int)ceil((double)(n1) / (double)(BLOCK*Tx));
	numBlocks.x = total_blocksX;
	//numBlocks.y = (int)ceil((double)m /(double) BLOCK);

	n1 += (BLOCK-1);
	mul<<<numBlocks, threadsPerBlock>>>(Agpu, Bgpu, Cgpu,  n1, m, n+m-1, Prime);	


	
		//*
		int j;
		int *temp1 = new  sfixn[n+m-1];
		cudaMemcpy(temp1, Cgpu, sizeof(int)*(m+n-1), cudaMemcpyDeviceToHost);
		int *temp2 = new  sfixn[n+m-1];
		for(i = 0; i < n+m-1; ++i)
			temp2[i] = 0;
		for(i = 0; i < n; ++i)
			for(j = 0; j < m; ++j)
				temp2[i+j] = add_mod(temp2[i+j] , mul_mod(A[i], B[j], Prime), Prime);
		for(i = 0; i < n+m-1; ++i)
		{
			if(temp1[i] != temp2[i])
			{
				cout<<"ERROR "<<i<<" "<<temp1[i]<<" "<<temp2[i]<<endl;
			}	
		}

	/*
		cudaMemcpy(temp, Agpu, (sizeof(sfixn)*(n + 2*(BLOCK-1))), cudaMemcpyDeviceToHost);

		cout<<endl<<"A: "<<endl;	
       		for(j=0; j < n +2*(BLOCK-1); ++j)
			cout<<temp[j]<<" ";
		cout<<endl;	

		cudaMemcpy(temp, Bgpu, sizeof(int)*m, cudaMemcpyDeviceToHost);
		cout<<endl<<"B: "<<endl;	 	
        	for(j=0; j < m; ++j)
			cout<<temp[j]<<" ";

		cout<<endl<<"C: "<<endl;
		cudaMemcpy(temp, Cgpu, sizeof(int)*(m+n-1), cudaMemcpyDeviceToHost);
		for(j=0; j < m+n-1 ; ++j)
			cout<<temp[j]<<" ";
		cout<<endl;	
		
		*/

	cudaFree(Agpu);
        cudaFree(Bgpu);
        cudaFree(Cgpu);
	delete [] temp1;
	delete [] temp2;

	//return GMtemp;	

}



int main(int argc, char *argv[])
{
	int n = 10, m = 10, p = 7, i;
	if (argc > 1) n = atoi(argv[1]);
	if (argc > 2) m = atoi(argv[2]);
        if (argc > 3) p = atoi(argv[3]);
	
	
	
	sfixn *A = new  sfixn[n];
	sfixn *B = new  sfixn[m];
	//sfixn *mul =  NULL;//new int[n+m];
	

	for(i = 0; i < n; ++i)	
	{
		A[i] = rand()% p;
		//cout<<A[i]<<" ";
	}
	cout<<endl;
		
	for(i = 0; i < m; ++i)
	{	
		B[i] = rand()% p;
		//cout<<B[i]<<" ";
	}
	cout<<endl;
	
	mulCUDA(A ,B, n, m, p);
	

	delete [] A;
	delete [] B;
	//delete [] mul;
	
	return 0;
}
	



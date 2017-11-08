#include<iostream>
#include <ctime>
#include<cmath>

using namespace std;

const int BASE_1 = 31;
const int BLOCK = 512;
const int Tx = 4;
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

__global__ void zeroC(sfixn *C, int n)
{
	int i = (blockIdx.x*blockDim.x + threadIdx.x)*BLOCK;
	if(i < n)
	{
		int j = i + BLOCK;
		for(; i < j && i < n; ++i)
			C[i] = 0;	
	}
}


__global__ void copyA(sfixn *A, sfixn *C, int n)
{
	int i = (blockIdx.x*blockDim.x + threadIdx.x)*BLOCK;
	if(i < n)
	{
		int j,k;
		if(i == 0)
		{
			j = BLOCK -1;
			for(; i < j; ++i)
				A[i] = 0; 
			i = n + BLOCK -1;
			j = i + BLOCK -1;
			for(; i < j; ++i)
				A[i] = 0;
			i = 0;
		}
		j = i + BLOCK;
		k = i + (BLOCK -1);
		for(; i < j && i < n; ++i, ++k)
			A[k] = C[i];		
	}					
}

//addFinal<<<numBlocks, threadsPerBlock>>>(Cgpu, CgpuFinal, aRowSize, m+n-1, i, Prime);			
				
__global__ void addFinal(sfixn *Cgpu, sfixn *CgpuFinal, int c, int n, int w, int P)
{
	int i = (blockIdx.x*blockDim.x + threadIdx.x)*BLOCK;
	if(i < c)
	{
		
		int k = i + w*BLOCK;
		i = i + c*w;
		int j = i + BLOCK;
		
		for(; i < j && k < n; ++i, ++k )
			CgpuFinal[k] =  add_mod( Cgpu[i], CgpuFinal[k], P);
		
	}
}		

			
__global__ void add(sfixn *Cgpu, sfixn *CgpuFinal, int c, int n, int w, int i, int j, int k, int P)
{
	int virtualBlockID = blockIdx.x % w;
	int rowNo = (blockIdx.x/ w);
	
	int firstRow = i + rowNo*k;
	
	
	int limitStart = (virtualBlockID*blockDim.x + threadIdx.x)*BLOCK;
	int limitEnd, r1start, fStart, secondRow;	
	r1start = (firstRow*c) +  limitStart;
	limitEnd = limitStart + BLOCK;
	
	if(limitStart < j*BLOCK)
	{		
		if(limitEnd > j*BLOCK)	limitEnd = j*BLOCK;
		if(limitEnd > c) limitEnd = c;		
		fStart = limitStart + firstRow*BLOCK; 
		for(; limitStart < limitEnd; ++limitStart, ++r1start, ++fStart )
			CgpuFinal[fStart] = add_mod(Cgpu[r1start], CgpuFinal[fStart], P);				
	}		
	else
	{
		if(limitEnd > c) limitEnd = c;
		secondRow = firstRow + j;
		int r2start = secondRow*c + limitStart - j*BLOCK;
		for(; limitStart < limitEnd; ++r1start, ++r2start,  ++limitStart)
			Cgpu[r2start] = add_mod(Cgpu[r1start], Cgpu[r2start], P);
	}
}
	
__global__ void mul(sfixn *A, sfixn *B, sfixn *C, int n, int m, int c, int P, int unitBlocks)
{
	__shared__ sfixn sA[T];
	__shared__ sfixn sB[BLOCK];
	sfixn Res[BLOCK];

	int virtualBlockID = blockIdx.x % unitBlocks;

	int i = (virtualBlockID*blockDim.x + threadIdx.x)*BLOCK;	
	int j = i + BLOCK;
	int k = threadIdx.x*BLOCK;
	int l = k + BLOCK;
	
	int whereInB = (blockIdx.x/ unitBlocks);
	
	for(; i < j && i < n; ++i, ++k) 
		sA[k] = A[i];
	for(; k < l ; ++k)
		sA[k] = 0;

	if(threadIdx.x == 0)
	{
		i = whereInB*BLOCK;
		j = i + BLOCK;		
		k = 0;
		for(; i < j && i < m; ++i, ++k)	sB[k] = B[i];
		for(; k < BLOCK; ++k) sB[k] = 0;
		j = (virtualBlockID*blockDim.x + Tx)*BLOCK;
		for(i = Tx*BLOCK; i < T && j < n; ++i, ++j )	sA[i] = A[j];
		for(; i < T; ++i) sA[i] = 0;
	}
	__syncthreads();

	l = threadIdx.x*BLOCK + BLOCK -1;
	whereInB = whereInB *c;
	int s = (virtualBlockID*blockDim.x + threadIdx.x)*BLOCK;	

	for(i = 0; i < BLOCK && s < c; ++i, ++s) 
	{		
		Res[i] = 0;	
		k = l + i;		
		for(j = 0; j <BLOCK; ++j, --k)
		{
			Res[i] = add_mod(Res[i] , mul_mod(sA[k], sB[j], P), P);			
		}
		C[s + whereInB] = Res[i];
	}		
}


void mulCUDA( sfixn *A,  sfixn *B, int n, int m, int Prime)
{
	int i, j, k, l, w;
	sfixn *Agpu, *Bgpu, *Cgpu, *CgpuFinal;

	if(m > n)	
	{
		i = m;
		m = n;
		n = i;
		
		Agpu = A;
		A = B;
		B = Agpu;	
	}
	int aRowSize = n + BLOCK -1;
	int RowNo = (int)ceil((double)m/(double)BLOCK);
	int RowNo2Pow = (int)ceil(log2((double)RowNo));
	int rowNoPerfect = pow(2, (double)RowNo2Pow);
	//cout<<endl<<endl<<RowNo<<" "<<RowNo2Pow<<" "<<rowNoPerfect<<endl;
	


	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
        //cudaEventCreate(&stop);
	//cudaEventRecord(start, 0);


	cudaMalloc((void **)&Agpu, ( sizeof(sfixn)*(n + 2*(BLOCK-1)))  );
	cudaMalloc((void **)&Bgpu, sizeof(sfixn)*m);
	cudaMalloc((void **)&Cgpu, sizeof(sfixn)*(aRowSize*rowNoPerfect));

        cudaMemcpy(Bgpu, B, sizeof(sfixn)*m, cudaMemcpyHostToDevice);
        cudaMemcpy(Cgpu, A, sizeof(sfixn)*n, cudaMemcpyHostToDevice);
	
	int total_blocksX = (int)ceil((double)(n) / (double)(BLOCK*Tx));
	dim3 threadsPerBlock(Tx); 
	dim3 numBlocks(total_blocksX);
	copyA<<<numBlocks, threadsPerBlock>>>(Agpu, Cgpu,  n);	

	numBlocks.x = (int)ceil( (double)(aRowSize*rowNoPerfect)/(double)(BLOCK*Tx) );	
	zeroC<<<numBlocks, threadsPerBlock>>>(Cgpu, aRowSize*rowNoPerfect);

	int n1 = n + (BLOCK-1);
	total_blocksX = (int)ceil((double)(n1) / (double)(BLOCK*Tx));
	numBlocks.x = total_blocksX* RowNo;

	n1 += (BLOCK-1);
	
	mul<<<numBlocks, threadsPerBlock>>>(Agpu, Bgpu, Cgpu,  n1, m, aRowSize, Prime, total_blocksX);	
	
	//cout<<endl<<n1<<" "<<numBlocks.x<<" "<<numBlocks.y<<" "<<total_blocksX<<endl;
	
	i =(n + m -1);
	cudaMalloc((void **)&CgpuFinal, sizeof(sfixn)*i );
	numBlocks.x = (int)ceil((double)(i) / (double)(BLOCK*Tx));
	zeroC<<<numBlocks, threadsPerBlock>>>(CgpuFinal, i);
	
	i = 0;
	j = 1;
	k = 2;
	w = (int)ceil((double)(aRowSize) / (double)(BLOCK*Tx));
	for(l = 0; l < RowNo2Pow; ++l)
	{		 
		numBlocks.x = w*pow(2, (double)RowNo2Pow-l-1);
		//cout<<endl<<i<<" "<<j<<" "<<k<<" "<<w<<" "<<numBlocks.x<<endl;	
		add<<<numBlocks, threadsPerBlock>>>(Cgpu, CgpuFinal, aRowSize, m+n-1, w, i, j, k, Prime);			

		i = i + j;
		j = 2*j;
		k = 2*k;
					
	}	
	//cout<<endl<<i<<endl;
	numBlocks.x = w;
	addFinal<<<numBlocks, threadsPerBlock>>>(Cgpu, CgpuFinal, aRowSize, m+n-1, i, Prime);			
	/*
	int *temp1 = new  sfixn[aRowSize*rowNoPerfect];
	cudaMemcpy(temp1, Cgpu, sizeof(int)*(aRowSize*rowNoPerfect), cudaMemcpyDeviceToHost);
	cout<<endl;
	for(i = 0; i < aRowSize*rowNoPerfect; ++i)
	{
		if(i % aRowSize ==0) cout<<endl;
		cout<<temp1[i]<<" ";
	}
	delete [] temp1;
	cout<<endl;
	*/


	//cudaEventRecord(stop, 0);
        //cudaEventSynchronize(stop);
        //float outerTime;
        //cudaEventElapsedTime(&outerTime, start, stop);
	//cout<<outerTime/1000.0<<endl;



	///*
	int *temp2 = new  sfixn[m+n-1];
	cudaMemcpy(temp2, CgpuFinal, sizeof(int)*(m+n-1), cudaMemcpyDeviceToHost);
	/**************************************************************
	/* MULTIPLICATION RESULT IS IN temp2 *************************
	*/


	delete [] temp2;
	//for(i = 0; i < m+n-1; ++i)
		//cout<<temp2[i]<<" ";
	//cout<<endl;
	
	//*/
	
	/*

	int *temp3 = new  sfixn[n+m-1];
	for(i = 0; i < n+m-1; ++i)
		temp3[i] = 0;
	for(i = 0; i < n; ++i)
		for(j = 0; j < m; ++j)
			temp3[i+j] = add_mod(temp3[i+j] , mul_mod(A[i], B[j], Prime), Prime);
	for(i = 0; i < n+m-1; ++i)
	{
		if(temp3[i] != temp2[i])
		{
			cout<<"ERROR "<<i<<" "<<temp3[i]<<" "<<temp2[i]<<endl;
		}	
	}


delete [] temp3;
	*/




       cudaFree(Agpu);
        cudaFree(Bgpu);
	
        cudaFree(Cgpu);
	cudaFree(CgpuFinal);
	

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
	//cout<<endl;
		
	for(i = 0; i < m; ++i)
	{	
		B[i] = rand()% p;
		//cout<<B[i]<<" ";
	}
	//cout<<endl;
	
	mulCUDA(A ,B, n, m, p);
	

	delete [] A;
	delete [] B;
	//delete [] mul;
	
	return 0;
}
	



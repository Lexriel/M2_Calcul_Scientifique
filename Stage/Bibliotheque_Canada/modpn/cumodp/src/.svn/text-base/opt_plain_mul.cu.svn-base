#include "../include/opt_plain_mul.h"

using namespace std;


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


/////////////////////////////////////////////////////////////////////
//BEGIN:opt_plain_mul_tst
/////////////////////////////////////////////////////////////////////
void opt_plain_mul(int n, int m, int Prime)
{
	sfixn *A = new  sfixn[n];
	sfixn *B = new  sfixn[m];
	
	sfixn i, p;
	p = Prime;

	for(i = 0; i < n; ++i)	
	{
		srand ( time(NULL));
		A[i] = rand()% p;
	}

	for(i = 0; i < m; ++i)
	{	
		srand ( time(NULL));
		B[i] = rand()% p;
	}
	
	printf("A := "); print_poly(n-1, A, 'x'); printf(";\n");
	printf("B := "); print_poly(m-1, B, 'x'); printf(";\n");
	int j, k, l, w;
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


	 ////////////////////////////////////////////////////////////////////////
        //This part is for benchmarking
        ////////////////////////////////////////////////////////////////////////
        cudaEvent_t start, stop;
        float time;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,0);
        ////////////////////////////////////////////////////////////////////////




	
	mul<<<numBlocks, threadsPerBlock>>>(Agpu, Bgpu, Cgpu,  n1, m, aRowSize, Prime, total_blocksX);	
	
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
		add<<<numBlocks, threadsPerBlock>>>(Cgpu, CgpuFinal, aRowSize, m+n-1, w, i, j, k, Prime);			

		i = i + j;
		j = 2*j;
		k = 2*k;
					
	}	
	
	numBlocks.x = w;
	addFinal<<<numBlocks, threadsPerBlock>>>(Cgpu, CgpuFinal, aRowSize, m+n-1, i, Prime);			

	////////////////////////////////////////////////////////////////////////
	//This part is for benchmarking
	////////////////////////////////////////////////////////////////////////
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);
	cudaEventDestroy(stop);
	
	printf("TIME USED: %f\n",time);

	double multiplications = (n*m);
	double additions = m*(m-1) + ( (n+m-1 -(2*m))*m);
	cout<<"Multiplications: "<<multiplications<<endl;
	cout<<"additions: "<<additions<<endl;

	int *temp2 = new  sfixn[m+n-1];
	cudaMemcpy(temp2, CgpuFinal, sizeof(int)*(m+n-1), cudaMemcpyDeviceToHost);

	printf("GResult := "); print_poly(n+m-2, temp2, 'x'); printf(";\n");
	
	delete [] temp2;

    cudaFree(Agpu);
    cudaFree(Bgpu);
	
    cudaFree(Cgpu);
	cudaFree(CgpuFinal);
	
	delete [] A;
	delete [] B;
	
}


/////////////////////////////////////////////////////////////////////
//END:opt_plain_mul_tst
/////////////////////////////////////////////////////////////////////
/*
int main(int argc, char *argv[])
{
	int n = 10, m = 10, p = 7, i;
	if (argc > 1) n = atoi(argv[1]);
	if (argc > 2) m = atoi(argv[2]);
    if (argc > 3) p = atoi(argv[3]);
	opt_plain_mul(n, m, p);

	return 0;
}
	
*/


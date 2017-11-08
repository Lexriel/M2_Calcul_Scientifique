#include<iostream>
#include <ctime>
#include<cmath>

#define BASE_1 31
#define BLOCK 8
#define THno 256

typedef int            sfixn;
typedef unsigned int   usfixn;

using namespace std;

__device__ __host__ __inline__ 
sfixn add_mod(sfixn a, sfixn b, sfixn P) {
    sfixn r = a + b;
    r -= P;
    r += (r >> BASE_1) & P;
    return r;
}


__device__ __host__ __inline__ 
sfixn sub_mod(sfixn a, sfixn b, int P) {
    sfixn r = a - b;
    r += (r >> BASE_1) & P;
    return r;
}


__device__ __host__ __inline__ 
sfixn neg_mod(sfixn a, int P) {
    sfixn r = - a;
    r += (r >> BASE_1) & P;
    return r;
}

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
sfixn pow_mod(sfixn a, sfixn ee, int P) {
    sfixn x, y;
    usfixn e;

    if (ee == 0) return 1;
    if (ee == 1) return a;
    if (ee < 0) e = - ((usfixn) ee); else e = ee;

    x = 1;
    y = a;
    while (e) {
        if (e & 1) x = mul_mod(x, y,P);
        y = mul_mod(y, y,P);
        e = e >> 1;
    }

    if (ee < 0) x = inv_mod(x,P);
    return x;
}


__global__ void nReduceM(int *A,  int *B,  int *status)
{
	if(status[4] == 0)
	{
		--status[2];
		while( status[2] >= 0 && B[ status[2] ] == 0)
			--status[2];
		if(status[2] < 0 )
			status[4] = 2;	
		else if( status[2] < status[1] )
			status[4] = 1;			
	}
	else if(status[4] == 1)
	{
		--status[1];
		while( status[1] >= 0 && A[ status[1] ] == 0)
			--status[1];
		if(status[1] < 0)
			status[4] = 3;	
		else if( status[2] >  status[1] )
			status[4] = 0;	
	}		
}


__global__ void stateCompute(int *A,  int *B,  int *status)
{
	if(status[4] == 0)
	{
		status[3] = status[2] - status[1];
		status[0] = mul_mod( inv_mod(A[ status[1]  ], status[5]), B[ status[2] ],  status[5])  ;  ;
	}
	else if(status[4] == 1)
	{
		status[3]= status[1] - status[2];
		status[0] =   mul_mod( inv_mod(B[ status[2]  ], status[5] ), A[ status[1] ],  status[5])  ;
	}
}
///*
__global__ void gcdGPU(int *A, int *B, int *status)
{

        int k = (blockIdx.x*blockDim.x + threadIdx.x)*BLOCK;

	if(status[4] == 0 && k < status[1])
	{
		int i = k;
		k = k+ BLOCK;
		if(k>= status[1]) k = status[1];
		for(; i < k; ++i)
			B[i+ status[3]] = sub_mod(B[i+ status[3]], mul_mod( status[0], A[i],  status[5]), status[5]);		
	}
	else if(status[4] == 1 && k < status[2])
	{
		int i = k;
		k = k+ BLOCK;
		if(k>= status[2]) k = status[2];
		for(; i < k; ++i)
			A[i+ status[3]] = sub_mod(A[i+ status[3]], mul_mod( status[0], B[i],  status[5]), status[5]);		
	}
}//*/


/*
__global__ void gcdGPU(int *A, int *B, int *status)
{
	int j= threadIdx.x;
        int k = (blockIdx.x*blockDim.x + j)*BLOCK;
	__shared__ int s_status[6];
	if(j < 6)
		s_status[j] = status[j] ;
	__syncthreads;

	int limit = s_status[1];;
	if(s_status[4] == 1 )  limit = s_status[2];

	if( k < limit && s_status[4] <= 1)
	{
		__shared__ int s_dataA[BLOCK*THno], s_dataB[BLOCK*THno];
		int i = k;
		k = k+ BLOCK;	
		int iCopy = i;
		j = j*BLOCK;
		if(k>= limit)	k = limit;
		if(s_status[4] == 0 )
		{			
			for(; i < k; ++i, ++j)
			{
				s_dataA[j] = A[i];
				s_dataB[j] = B[i+ s_status[3]];
			}
			j = threadIdx.x*BLOCK;	
			for(; iCopy < k; ++iCopy, ++j)
				B[iCopy + s_status[3]] = sub_mod(s_dataB[j], mul_mod( s_status[0], s_dataA[j],  s_status[5]), s_status[5]);				
		}
		else if(s_status[4] == 1 )
		{		
			for(; i < k; ++i, ++j)
			{
				s_dataA[j] = A[i+s_status[3]];
				s_dataB[j] = B[i];			
			}			
			j = threadIdx.x*BLOCK;
			for(; iCopy < k; ++iCopy, ++j)
				A[iCopy + s_status[3]] = sub_mod(s_dataA[j], mul_mod( s_status[0], s_dataB[j],  s_status[5]), s_status[5]);		
		}					
	}
}
*/


/*
state[5]:=
0-> multiply with the inverse of leading and the other leading in finite field
1-> n
2-> m
3-> |n-m|
4-> status 0 means B > A
    status 1 means A > B
    status 2 means gcd = A
    status 3 means gcd = B
5-> Prime for prime field
*/

void gcdCPU(int *A, int *B, int n, int m, int P)
{
	
	if(n==1 || m == 1)
		return;
	int *Agpu, *Bgpu, *stateGpu, state[6]= { mul_mod( inv_mod(B[m-1], P), A[n-1],  P), n-1, m-1, n-m, 1, P};

	cudaMalloc((void **)&Agpu, sizeof(int)*n);
        cudaMemcpy(Agpu, A, sizeof(int)*n, cudaMemcpyHostToDevice);
	delete [] A;
	
	cudaMalloc((void **)&Bgpu, sizeof(int)*m);
        cudaMemcpy(Bgpu, B, sizeof(int)*m, cudaMemcpyHostToDevice);
	delete [] B;

	cudaMalloc((void **)&stateGpu, sizeof(int)*6);
        cudaMemcpy(stateGpu, state, sizeof(int)*6, cudaMemcpyHostToDevice);

	int i, itNu, currentM, diff;
	diff = n-m;
	itNu = m+n+1;	
	currentM = m;

	dim3 threadsPerBlock(THno);
        dim3 numBlocksN;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
        cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//*
	int statusCPU[1000], GCD[1000], T[6],j;
	//*/
	
	
	for(i = 1; i <= itNu; ++i)
	{
		/*
		cout<<"i: "<<i<<" diff: "<<diff<<" currentM :"<<currentM<<endl;
		*/
		numBlocksN=(ceil((double)( currentM )/(double)(THno*BLOCK)));

        	gcdGPU<<<numBlocksN, threadsPerBlock>>>(Agpu, Bgpu, stateGpu);		
		nReduceM<<<1, 1>>>(Agpu, Bgpu, stateGpu);
		stateCompute<<<1, 1>>>(Agpu, Bgpu, stateGpu);

		if(diff <= 1)
		{
			diff = 1 - diff;
			currentM = currentM - diff;
		}
		else
			--diff;	
		//*
		if(i==itNu){
		//cout<<"i: "<<i<<" diff: "<<diff<<" currentM :"<<currentM<<endl;
       		cudaMemcpy(statusCPU, stateGpu, sizeof(int)*6, cudaMemcpyDeviceToHost);

		//cout<<endl; 	
       		//for(j=0; j < 6; ++j)
		//	cout<<statusCPU[j]<<" ";
		//cout<<endl;
	

		cudaMemcpy(GCD, Agpu, sizeof(int)*statusCPU[1]+1, cudaMemcpyDeviceToHost);
		cudaMemcpy(T, Bgpu, sizeof(int)*statusCPU[2]+1, cudaMemcpyDeviceToHost);
	
		//cout<<endl<<"A"<<endl;
	 	if(statusCPU[4] ==2)
		{
        	for(j=0; j <= statusCPU[1]; ++j)
			cout<<GCD[j]<<" ";
		
		cout<<endl;//<<"B: "<<endl;
		}
		if(statusCPU[4] ==3)
		{
        	for(j=0; j <= statusCPU[2]; ++j)
			cout<<T[j]<<" ";
		cout<<endl;
		}
		//cin>>kkk;
		}
		//*/	
	}

	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float outerTime;
        cudaEventElapsedTime(&outerTime, start, stop);
	cout<<outerTime/1000.0<<endl;



	

	

	cudaFree(Agpu);
        cudaFree(Bgpu);
        cudaFree(stateGpu);	
}





int main(int argc, char *argv[])
{
	int n = 10, m = 10, p = 7, i;

	if (argc > 1) n = atoi(argv[1]);
	if (argc > 2) m = atoi(argv[2]);
        if (argc > 3) p = atoi(argv[3]);
	
	int *A = new int[n];
	int *B = new int[m];

	for(i = 0; i < n; ++i)	
		A[i] = rand()% p;
		
	for(i = 0; i < m; ++i)	
		B[i] = rand()% p;
	/*
	cout<<endl;
	for(i = 0; i < n; ++i )
		cout<<A[i]<<" ";
	cout<<endl;
	for(i = 0; i < m; ++i )
		cout<<B[i]<<" ";
	cout<<endl;
	*/
	
	i = n-1;
	while(A[i--] == 0) --n;
	i = m-1;
	while(B[i--] == 0) --m;
	if(n>=m)
		gcdCPU(A,B, n, m, p);	
	else
		gcdCPU(B,A, m, n, p);
	
	return 0;
}
	
	

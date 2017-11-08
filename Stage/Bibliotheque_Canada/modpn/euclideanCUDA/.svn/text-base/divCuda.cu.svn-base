#include<iostream>
#include <ctime>
#include<cmath>

#define BASE_1 31


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


__global__ void nReduceM(int *A,  int *status)
{
	if(status[4] == 1)
	{
		--status[1];
		while( status[1] >= 0 && A[ status[1] ] == 0)
			--status[1];
		if(status[1] < status[2] || status[1] < 0)
			status[4] = 0;			
	}		
}


__global__ void stateCompute(int *A,  int *B, int *div, int *status)
{
	if(status[4] == 1)
	{
		status[3]= status[1] - status[2];
		status[0] =   mul_mod( inv_mod(B[ status[2]  ], status[5] ), A[ status[1] ],  status[5])  ;
		div[status[3]] = status[0];
	}
}
///*



__global__ void divGPU(int *A, int *B, int *status, int BLOCK)
{

        int k = (blockIdx.x*blockDim.x + threadIdx.x)*BLOCK;

	if(status[4] == 1 && k < status[2])
	{
		int i = k;
		k = k+ BLOCK;
		if(k> status[2]) k = status[2];
		/*
		int j=0;
		*/
		for(; i < k; ++i)
		{
			/*			
			div[j++] = i;
			div[j++] = k;
			div[j++] = A[i+ status[3]];
			div[j++] = B[i];
			*/
			A[i+ status[3]] = sub_mod(A[i+ status[3]], mul_mod( status[0], B[i],  status[5]), status[5]);		
			/*
			div[j++] = A[i+ status[3]];
			*/
		}
	}
}
//*/

__global__ void	initDiv(int itNu, int *divGpu, int BLOCK, int *status)
{
	int k = (blockIdx.x*blockDim.x + threadIdx.x)*BLOCK;
	if(status[4] == 1 && k < itNu)
	{
		int i = k;
		k = k+ BLOCK;
		if(k> itNu) k = itNu;
		for(; i < k; ++i) divGpu[i] = 0;
	}
}



/*
state[5]:=
0-> multiply with the inverse of leading and the other leading in finite field
1-> n
2-> m
3-> |n-m|
4-> status 0 means B > A
    status 1 means A >= B
5-> Prime for prime field
*/
int* divCUDA(int *A, int *B,  int *itNu, int n, int m, int P, int BLOCK, int THno)
{
	int *div;
	int i = n-1;
	while(A[i--] == 0) --n;
	i = m-1;
	while(B[i--] == 0) --m;
	if(m >= n)
	{
		div = A;
		A = B;
		B = div;
		
		i = n;
		n = m;
		m = i;
	}
	
	*itNu = n - m + 1;
	/*
	cout<<"itNu: "<<*itNu<<" n: "<<n<<" m: "<<m<<endl;
	for(i =0; i < n; ++i) cout<<A[i]<<" ";
	cout<<endl;
	for(i =0; i < m; ++i) cout<<B[i]<<" ";
	cout<<endl;
	*/	
	int *Agpu, *Bgpu, *stateGpu, *divGpu, state[6] = {-1 , n-1, m-1, -1, 1, P};
	div = new int[*itNu];
	

	cudaMalloc((void **)&Agpu, sizeof(int)*n);
        cudaMemcpy(Agpu, A, sizeof(int)*n, cudaMemcpyHostToDevice);
		
	cudaMalloc((void **)&Bgpu, sizeof(int)*m);
        cudaMemcpy(Bgpu, B, sizeof(int)*m, cudaMemcpyHostToDevice);	

	cudaMalloc((void **)&stateGpu, sizeof(int)*6);
        cudaMemcpy(stateGpu, state, sizeof(int)*6, cudaMemcpyHostToDevice);
	
	cudaMalloc((void **)&divGpu, sizeof(int)*(*itNu));
	
	
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
        cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	/*
	int Atemp[1000], divTemp[1000], status[6], kkk,j;
	*/
	dim3 threadsPerBlock(THno);
        dim3 numBlocksN = (ceil((double)(*itNu )/(double)(THno*BLOCK)));
	
	initDiv<<<numBlocksN, threadsPerBlock>>>(*itNu, divGpu, BLOCK, stateGpu);	

	numBlocksN = (ceil((double)( m )/(double)(THno*BLOCK)));	
	for(i = 0; i < *itNu; ++i)
	{
		/*
		cout<<"i: "<<i<<endl;
		*/
		
		stateCompute<<<1, 1>>>(Agpu, Bgpu, divGpu, stateGpu);
        	divGPU<<<numBlocksN, threadsPerBlock>>>(Agpu, Bgpu, stateGpu, BLOCK);		
		//cudaThreadSynchronize();


		
		/*
		cudaMemcpy(status, stateGpu, sizeof(int)*6, cudaMemcpyDeviceToHost);

		cout<<endl; 	
       		for(j=0; j < 6; ++j)
			cout<<status[j]<<" ";
		cout<<endl;	

		cudaMemcpy(Atemp, Agpu, sizeof(int)*status[1]+1, cudaMemcpyDeviceToHost);
		cout<<endl<<"A: "<<endl;
	 	
        	for(j=0; j <= status[1]; ++j)
			cout<<Atemp[j]<<" ";
		cout<<endl<<"div: "<<endl;
		cudaMemcpy(divTemp, divGpu, sizeof(int)*(*itNu), cudaMemcpyDeviceToHost);
		for(j=0; j < *itNu; ++j)
			cout<<divTemp[j]<<" ";
		cout<<endl;		
		
		cin>>kkk;
		*/
		nReduceM<<<1, 1>>>(Agpu, stateGpu);	
	}
	cudaMemcpy(div, divGpu, sizeof(int)*(*itNu), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float outerTime;
        cudaEventElapsedTime(&outerTime, start, stop);
	cout<<outerTime/1000.0<<endl;
	

	cudaFree(Agpu);
        cudaFree(Bgpu);
        cudaFree(stateGpu);
        cudaFree(divGpu);

	return div;
		
}





int main(int argc, char *argv[])
{
	int n = 10, m = 10, p = 7, i, BLOCK = 8, THno = 256;

	if (argc > 1) n = atoi(argv[1]);
	if (argc > 2) m = atoi(argv[2]);
        if (argc > 3) p = atoi(argv[3]);
	if (argc > 4) BLOCK = atoi(argv[4]);
	if (argc > 5) THno = atoi(argv[5]);
	
	int *A = new int[n];
	int *B = new int[m];
	int *div = NULL, degDiv;

	for(i = 0; i < n; ++i)	
		A[i] = rand()% p;
		
	for(i = 0; i < m; ++i)	
		B[i] = rand()% p;
	///*
	cout<<endl;
	for(i = 0; i < n; ++i )
		cout<<A[i]<<" ";
	cout<<endl;
	for(i = 0; i < m; ++i )
		cout<<B[i]<<" ";
	cout<<endl;
	//*/
	div=divCUDA(A ,B, &degDiv, n, m, p, BLOCK, THno);
	cout<<endl<<"degree: "<<degDiv<<endl<<div[0]<<endl;
	//*
	for(i = 0; i < degDiv; ++i )
		cout<<div[i]<<" ";
	cout<<endl;
	//*/		

	delete [] A;
	delete [] B;
	delete [] div;
	
	return 0;
}
	
	

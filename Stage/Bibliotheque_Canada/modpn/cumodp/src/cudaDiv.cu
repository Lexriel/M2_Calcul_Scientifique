#include<iostream>
#include "cudaDiv.h"


		
__global__ void statusUpdate(sfixn *A,  int *status)
{
	if(status[4] == -1)
	{
		if( (status[0] - s) < status[1] )
		{
			//if( (status[0] - s) >= 0 ) status[4] = status[0] - s;
			//else 
			status[4] = -3;
		}
		else
		{	
			for(status[0] -= s; status[0] >= status[1]; --status[0])
			{
				if(A[ status[0] ] != 0)	
					break;
			}
			if( status[0] <  status[1])
			{
				status[4] = -4;
			}
		}
	}
}

__global__ void zero(sfixn *A, int n)
{
	int k = (blockIdx.x*blockDim.x + threadIdx.x);
	if(k < n)
		A[k] = 0;
}

__global__ void	copyRem(sfixn *A, sfixn *R, int *status)
{
	int k = (blockIdx.x*blockDim.x + threadIdx.x);
	if(k <= status[0] )
	{
		R[k] = A[k];
	}
}


__global__ void	reduceM(sfixn *B,  int *status)
{	
	for(; status[1] >= 0; --status[1])
	{
		if(B[ status[1] ] != 0)	
			break;
	}	
	if(status[1] < 0)
	{
		status[4] = -2;
	}
		
}

__global__ void divGPU(sfixn *A, sfixn *B, int *status, sfixn *Q, sfixn *R)
{

	if(status[4] == -1)
	{
		int i, j;
		__shared__ int headA, headB;
		__shared__ int startA,  startBind;

		__shared__ sfixn inv, p;
		__shared__ sfixn sA[t*s];
		__shared__ sfixn sB[s + t*s];
		__shared__ sfixn sAcommon[s];
		__shared__ sfixn sBcommon[s];

		if(threadIdx.x == 0)
		{
					
			headA = status[0];
			headB = status[1];
			startA = headA -s - s*t*blockIdx.x;			
			startBind = headB  - s*t*blockIdx.x -1;
			p = status[2];
			inv = inv_mod(B[headB],p);		
				
		}
		__syncthreads();
	
		if(headA >= headB && headB >= 0)
		{
			j = headA - threadIdx.x;	
			if(j >= 0)
				sAcommon[threadIdx.x] = A[j];
			else
				sAcommon[threadIdx.x] = 0;
			j = headB - threadIdx.x;
			if(j >= 0)
				sBcommon[threadIdx.x] = B[j];
			else
				sBcommon[threadIdx.x] = 0;	
		
			j = startA - t*threadIdx.x; 
			for(i = 0; i < t; ++i)
			{
				if((j - i) >= 0 )
					sA[threadIdx.x*t + i] = A[j-i];
				else
					sA[threadIdx.x*t + i] = 0;			
			}		
	
			j = startBind - (t+1)*threadIdx.x;
			for(i = 0; i <= t; ++i)
			{
				if((j - i) >= 0 )
					sB[threadIdx.x*(t+1) + i] = B[j-i];
				else
					sB[threadIdx.x*(t+1) + i] = 0;				
			}
			__syncthreads();	
			int factor;		
			for(i = 0; i < s && (headA-i) >= headB; ++i)
			{			
				factor = mul_mod(sAcommon[i], inv, p);

				if(blockIdx.x == 0 && threadIdx.x == 0)
					Q[ headA - i - headB ] = factor;
				if(i <= threadIdx.x )
					sAcommon[threadIdx.x ] = sub_mod(sAcommon[threadIdx.x ], mul_mod(sBcommon[threadIdx.x - i], factor, p), p);
	
				for(j = 0; j < t; ++j)
					sA[threadIdx.x*t + j ] = sub_mod(sA[threadIdx.x*t + j ], mul_mod(sB[s - 1 - i + threadIdx.x*t + j], factor, p), p);
						
				__syncthreads();	
			}		
		
			if(i < s && blockIdx.x == 0 && threadIdx.x == 0)
			{
				status[4] = -5;
				status[0] -= s;		
				 
				for(status[3] = headB - 1; i < s &&  status[3] >= 0; ++i, --status[3])
				{
					R[status[3]] = sAcommon[i];							
				}

			}		
				
			
			

			j = startA - t*threadIdx.x; 
			for(i = 0; i < t && (j - i) >= 0; ++i)
				A[j - i ] = sA[threadIdx.x*t + i];
			


			
		}
	}
	
}


float divPrimeField(sfixn *A, sfixn *B, sfixn *R, sfixn *Q, int n, int m, sfixn p)
{
	
	sfixn *Agpu, *Bgpu, *Rgpu, *Qgpu, *stateGpu;

	cudaMalloc((void **)&Agpu,     sizeof(sfixn)*n);
	cudaMalloc((void **)&Bgpu,     sizeof(sfixn)*m);
	cudaMalloc((void **)&Qgpu,     sizeof(sfixn)*(n - m + 1));
	cudaMalloc((void **)&Rgpu,     sizeof(sfixn)*(m - 1));
	cudaMalloc((void **)&stateGpu, sizeof(sfixn)*5);

        cudaMemcpy(Agpu, A, sizeof(sfixn)*n, cudaMemcpyHostToDevice);	
        cudaMemcpy(Bgpu, B, sizeof(sfixn)*(m), cudaMemcpyHostToDevice);        

	
	//Both A and B has at least one nonzero coefficient
	sfixn state[5]= { n-1 + s, m-1, p, 0, -1};
	cudaMemcpy(stateGpu, state, sizeof(sfixn)*5, cudaMemcpyHostToDevice);

	int i;

	int numBlocksN;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
        cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	numBlocksN = (int)(ceil((double)(n - m + 1)/(double)(T)));
	zero<<<numBlocksN, T>>>(Qgpu, (n - m + 1));

	numBlocksN = (int)(ceil((double)(m-1.0)/(double)(T)));
	zero<<<numBlocksN, T>>>(Rgpu, (m - 1));

	reduceM<<<1, 1>>>(Bgpu, stateGpu);

	numBlocksN = (int)(ceil((double)( m - 1.0)/(double)(T*t)));

	for(i = n; i>= m; i -= s )
	{
		statusUpdate<<<1, 1>>>(Agpu, stateGpu);
		divGPU<<<numBlocksN, T>>>(Agpu, Bgpu, stateGpu, Qgpu, Rgpu);
	}
	numBlocksN = (int)(ceil((double)(m-1)/(double)(T)));
	copyRem<<<numBlocksN, T>>>(Agpu, Rgpu, stateGpu);


	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float outerTime;
        cudaEventElapsedTime(&outerTime, start, stop);

	

	cudaMemcpy(state, stateGpu,  sizeof(sfixn)*5, cudaMemcpyDeviceToHost);
	
		
	
	cudaMemcpy(Q, Qgpu,  sizeof(sfixn)*(n-m+1), cudaMemcpyDeviceToHost);
	
	
	cudaMemcpy(R, Rgpu,  sizeof(sfixn)*(m - 1), cudaMemcpyDeviceToHost);
	
	
	
	cudaFree(Agpu);
        cudaFree(Bgpu);
        cudaFree(stateGpu);
        cudaFree(Qgpu);
        cudaFree(Rgpu);
	return outerTime;

}






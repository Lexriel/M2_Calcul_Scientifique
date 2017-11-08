#include<iostream>
#include "../include/cudaGcd.h"

using namespace std;


__global__ void	reduceMgcd(sfixn *B,  int *status)
{
	if(status[4] == -1)
	{		
		for(; status[1] >= 0; --status[1])
		{
			if(B[ status[1] ] != 0)	
				break;
		}
	}		
}

__global__ void	reduceNgcd(sfixn *A,  int *status)
{	
	if(status[4] == -1)
	{
		for(; status[0] >= 0; --status[0])
		{
			if(A[ status[0] ] != 0)	
				break;
		}
	}			
}


__global__ void	status3(int *status)
{
	if(status[4] == -1)
	{
	     if(status[1] <  status[0]) status[3] = (status[1] + T)/(2*T) + 1;
	     else status[3] = (status[0] + T)/(2*T) + 1;
	}	
}


__global__ void	status4(int *status)
{
	if(status[4] == -1)
	{
		if(status[0] <  0 && status[1] <  0)   status[4] = -2;
		if(status[0] <  0 && status[1] >= 0 )  status[4] = -3;
		if(status[0] >= 0 && status[1] <  0 )  status[4] = -4;
	}	
}

__global__ void	status5(int *status)
{
	if(status[4] == -1)
	{
	          if(status[0] <  status[1] && status[5] == 1) status[5] = 0;
             else if(status[0] >  status[1] && status[5] == 0) status[5] = 1;	     
	}	
}



__global__ void gcdGPU(sfixn *A, sfixn *B, int *status)
{

	//
	//int r;
	//
	if(status[4] == -1 &&  blockIdx.x < status[3])
	{
		int i, j, k;
		

		__shared__ int startAcom, startBcom, endAcom, endBcom;		
		__shared__ int startAsh,  startBsh, endAsh, endBsh;
		__shared__ int headA, headB, startA, startB;
		__shared__ int selectedA, factor, p;
		__shared__ sfixn sA[3*T], sB[3*T], sAcommon[T], sBcommon[T];

		if(threadIdx.x == 0)
		{					
			headA = status[0]; headB     = status[1]; 
			p     = status[2]; selectedA = status[5];

			startA = status[0] - 2*T*blockIdx.x;
			startB = status[1] - 2*T*blockIdx.x;

			startAcom = 0; 	 startBcom = 0; 
			endAcom   = T;	 endBcom   = T;
			startAsh  = 0;   startBsh  = 0; 
			endAsh    = 3*T; endBsh    = 3*T;
		
		
			
					
		}
		__syncthreads();		
	
		i = 3*threadIdx.x;
		sAcommon[threadIdx.x] = 0;
		sBcommon[threadIdx.x] = 0;
		sA[i] = 0;    sA[i+1] = 0;  sA[i+2] = 0;
		sB[i] = 0;    sB[i+1] = 0;  sB[i+2] = 0;
		k = threadIdx.x;			
		if( (headA - k) >= 0)	sAcommon[threadIdx.x] = A[headA - threadIdx.x];		
		if( (headB - k) >= 0) sBcommon[threadIdx.x] = B[headB - threadIdx.x];		
		
		j = startA - 3*threadIdx.x;		
		for(k = 0; k < 3 && (j-k) >= 0; ++k)
			sA[i + k] =  A[j - k];	
		
		j = startB - 3*threadIdx.x;		
		for(k = 0; k < 3 && (j-k) >= 0; ++k)
			sB[i + k] = B[j - k];
		
		__syncthreads();


		/*
		if(threadIdx.x == 0 && blockIdx.x == 0)
		{
				r = 0;
				track[r++] = startAcom;
				track[r++] = endAcom;
				track[r++] = startBcom;
				track[r++] = endBcom;
				track[r++] = startAsh;
				track[r++] = endAsh;
				track[r++] = startBsh;
				track[r++] = endBsh ;
				track[r++] = startA;
				track[r++] = startB;
				track[r++] = selectedA;
				track[r++] = headA;
				track[r++] = headB;
				track[r++]=  sA[0];
				track[r++] = sA[1];
				track[r++] = sA[2];
				track[r++] = sA[3];
				track[r++] = sA[4];
				track[r++] = sA[5];
				track[r++] = sB[0];
				track[r++] = sB[1];
				track[r++] = sB[2];
				track[r++] = sB[3];
				track[r++] = sAcommon[0];
				track[r++] = sAcommon[1];
				track[r++] = sAcommon[2];
				track[r++] = sAcommon[3];
				track[r++] = sAcommon[4];
				track[r++] = sAcommon[5];
				track[r++] = sBcommon[0];
				track[r++] = sBcommon[1];
				track[r++] = sBcommon[2];
				track[r++] = sBcommon[3];
				
		}
		*/


		for(;selectedA >= 0;)
		{			
			for(;selectedA == 1;)
			{				
				if(threadIdx.x == 0)
					factor = mul_mod(sAcommon[startAcom], inv_mod(sBcommon[startBcom],p), p);
				__syncthreads();
				
				j = threadIdx.x + startBcom;
				i = threadIdx.x + startAcom;
				if(j < endBcom)
					sAcommon[i] =  sub_mod(sAcommon[i], mul_mod(sBcommon[j],factor,p), p);
				__syncthreads();

				j = (threadIdx.x*3) + startBsh;
				i = (threadIdx.x*3) + startAsh;
				for(k = 0; k < 3 && (j+k) < endBsh; ++k)
					sA[i+k] = sub_mod(sA[i+k], mul_mod(sB[j+k], factor, p), p);
				__syncthreads();
				
				if(threadIdx.x == 0)
				{
					++startAcom; --endBcom;	++startAsh; --endBsh; --startA; --headA;
					k = 0;
					if(startAcom < endAcom)
					{
						for(;sAcommon[startAcom] == 0; ++k, ++startAcom, --headA)
						{						
							if(startAcom >= endAcom || headA < 0) 
								break;						
						}
					}
					if(k > 0)
					{
						endBcom -= k;	startAsh += k; endBsh -= k; startA -= k; 
						
					}
					if( (startAcom >= endAcom)|| headA < 0 )
					{
						if(blockIdx.x == 0) status[5] = selectedA;
						 selectedA = -1;
					}
					else if(headA < headB) 
						selectedA = 0;										
				}						
				__syncthreads();				
			}							
			for(;selectedA == 0;)
			{
				if(threadIdx.x == 0)
					factor = mul_mod(sBcommon[startBcom], inv_mod(sAcommon[startAcom],p), p);
				__syncthreads();

				j = threadIdx.x + startAcom;
				i = threadIdx.x + startBcom;
				if(j < endAcom)
					sBcommon[i] = sub_mod(sBcommon[i], mul_mod(sAcommon[j],factor,p), p);
				__syncthreads();

				j = (threadIdx.x*3) + startAsh;
				i = (threadIdx.x*3) + startBsh;
				for(k = 0; k < 3 && (j+k) < endAsh; ++k)
					sB[i+k] = sub_mod(sB[i+k], mul_mod(sA[j+k], factor, p), p);
				__syncthreads();
			
				if(threadIdx.x == 0)
				{
					++startBcom; --endAcom;	++startBsh; --endAsh; --startB; --headB;
					k = 0;
					if(startBcom < endBcom)
					{
						for(;sBcommon[startBcom] == 0; ++k, ++startBcom, --headB)
						{						
							if(startBcom >= endBcom || headB < 0) 
								break;						
						}
					}
					if(k > 0)
					{
						endAcom -= k;	startBsh += k; endAsh -= k; startB -= k; 
						
					}
					if( (startBcom >= endBcom) || headB < 0 )
					{
						if(blockIdx.x == 0) status[5] = selectedA; 						 
						selectedA = -1;
					}
					else if(headB < headA) 
						selectedA = 1;
					
				}		
				__syncthreads();
			}
			
				
		}

		i = startA - threadIdx.x*2;
		if( i >= 0 )
		{
			j = startAsh + threadIdx.x*2;
			A[ i ] = sA[ j ];
			--i;
			if( i >= 0 )
				A[ i ] = sA[ j + 1 ];
			
		}
		
		i = startB - threadIdx.x*2;
		if( i >= 0 )
		{
			j = startBsh + threadIdx.x*2;
			B[ i ] = sB[ j ];
			--i;
			if( i >= 0 )
				B[ i ] = sB[ j + 1 ];				
		}
		
		if(threadIdx.x == 0 && blockIdx.x == 0)
		{
			status[0] = headA;
			status[1] = headB;						
		}

		
	}	
}


float gcdPrimeField(sfixn *A, sfixn *B, sfixn *G, int n, int m, sfixn p)
{
	
	sfixn *Agpu, *Bgpu, *stateGpu;

	cudaMalloc((void **)&Agpu,     sizeof(sfixn)*n);
	cudaMalloc((void **)&Bgpu,     sizeof(sfixn)*m);
	cudaMalloc((void **)&stateGpu, sizeof(sfixn)*6);

        cudaMemcpy(Agpu, A, sizeof(sfixn)*n, cudaMemcpyHostToDevice);	
        cudaMemcpy(Bgpu, B, sizeof(sfixn)*(m), cudaMemcpyHostToDevice);        

	
	//Both A and B has at least one nonzero coefficient
	sfixn state[6]= { n-1, m-1, p, 0, -1, 1};
	if(m > n) state[5] = 0;
	cudaMemcpy(stateGpu, state, sizeof(sfixn)*6, cudaMemcpyHostToDevice);

	int i, j;

	int numBlocksN;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
        cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	/*
	sfixn track[1000]={0};
	sfixn *trackGPU;
	cudaMalloc((void **)&trackGPU,     sizeof(sfixn)*1000);
	cudaMemcpy(trackGPU, track, sizeof(sfixn)*1000, cudaMemcpyHostToDevice);
	*/
		

	if(m <= n) 	j = m + T;
	else j = n + T;		
	numBlocksN = (int)(ceil((double)( j)/(double)(T*2)));
	
	for(i = n+m; i>= 0; i -= T )
	{
		reduceNgcd<<<1, 1>>>(Agpu, stateGpu);
		reduceMgcd<<<1, 1>>>(Bgpu, stateGpu);

		status4<<<1, 1>>>(stateGpu);		
		status3<<<1, 1>>>(stateGpu);		
		status5<<<1, 1>>>(stateGpu);
		
		gcdGPU<<<numBlocksN, T>>>(Agpu, Bgpu, stateGpu);		
	}


	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float outerTime;
        cudaEventElapsedTime(&outerTime, start, stop);
	
	cudaMemcpy(state, stateGpu,  sizeof(sfixn)*6, cudaMemcpyDeviceToHost);
	
	//
	//cudaMemcpy(track, trackGPU,  sizeof(sfixn)*1000, cudaMemcpyDeviceToHost);
	//cout<<endl;
	//for(i = 0; i < 100; ++i)
	//{
		//if(i%10 == 0) cout<<endl;
		//cout<<track[i]<<" ";
	//}
	//cout<<endl;

	//cudaMemcpy(A, Agpu,  sizeof(sfixn)*n, cudaMemcpyDeviceToHost);
	//cout<<endl;
	//for(i = 0; i < n; ++i)
		//cout<<A[i]<<" ";
	//cout<<endl;
	//cudaMemcpy(B, Bgpu,  sizeof(sfixn)*m, cudaMemcpyDeviceToHost);
	//for(i = 0; i < m; ++i)
		//cout<<B[i]<<" ";
	//cout<<endl;

	//
	
	//
	//cout<<endl<<state[0]<<" "<<state[1]<<" "<<state[2]<<" "<<state[3]<<" "<<state[4]<<" "<<state[5]<<endl;
	//

	j = -1;

	if(state[0] >= 0)
	{
		j = state[0]+1;
		cudaMemcpy(G, Agpu, sizeof(sfixn)*j, cudaMemcpyDeviceToHost);
	}
	else if(state[1] >= 0)
	{
		j = state[1] +1;
		cudaMemcpy(G, Bgpu, sizeof(sfixn)*j, cudaMemcpyDeviceToHost);
	}
	
	cudaFree(Agpu);
        cudaFree(Bgpu);
        cudaFree(stateGpu);

	return outerTime;
}



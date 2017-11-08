#include "../include/list_pointwise_mul.h"


/**
 * The list_pointwise_mul contains utilities for list_of_polynomial_multiplications
 * **/


/**
 * This kernel applies a list of pointwise multiplications
 * Once you have list_fft ed your polynomial, you could apply this guy 
 * 
 * L is the list_of polynomial
 * ln is the length of each polynomial
 * length_layer is the total length of L
 * */
__device__ __global__ void list_pointwise_mul(sfixn *L, sfixn ln, sfixn p, double pinv, sfixn length_layer)
{
	sfixn tid = threadIdx.x + blockIdx.x * blockDim.x;
	/*
	if(tid + ln < length_layer)
		L[tid] = mul_mod(L[tid], L[tid+ln], p, pinv);
	*/
	if (tid < length_layer)
	{
		//sfixn offset = tid / (ln * 2);
		sfixn iid =  tid % (ln * 2);
		if (iid < ln)
		{
			L[tid] = mul_mod(L[tid] , L[tid + ln], p , pinv); 
		}
	}
}

// Comparison for list_pointwise_mul
__host__ void CPU_list_pointwise_mul(sfixn *L, sfixn ln, sfixn num_poly, sfixn p, double pinv)
{
	for (sfixn i = 0; i < num_poly/2; i++)
	{
		sfixn offset = i * ln * 2;
		for (sfixn j = 0; j < ln; j++)
		{
			L[offset+j] = mul_mod(L[offset+j], L[offset+j+ln], p, pinv);
		}
	}
}

/**
 * To apply FFT based poly multiplication, you have to expand the polynomials first
 * This guy is to help you to expand a list of polynomials in the subproduct tree M at offset start_offset
 * from length_poly to length ln
 * and store the expanded polynomial in L
 * the bound of L is length_layer
 * */

/* Confusion of Anis on 21st February
should be:
if(rem < length_poly)
			L[tid] = M[start_offset + spoly * (length_poly+1) + rem];

*/
__device__ __global__ void list_expand_to_fft(sfixn *M, sfixn *L, sfixn length_poly, sfixn ln, sfixn start_offset, sfixn length_layer)
{
	sfixn tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < length_layer)
	{
		sfixn spoly = tid/ln;
		sfixn rem = tid%ln;
		if(rem < length_poly)
			L[tid] = M[start_offset + spoly * length_poly + rem];
		else
			L[tid] = 0;
	}
}

// Comparison for list_expand_to_fft
__host__ void CPU_list_expand_to_fft(sfixn *M, sfixn *L, sfixn length_M, sfixn length_poly, sfixn ln, sfixn start_offset)
{
	sfixn num_poly = length_M / length_poly;

	for (sfixn i = 0; i <  num_poly; i++)
	{
		sfixn offsetM = start_offset + i * length_poly;
		sfixn offsetL = i * ln;

		for (sfixn j = 0; j <  ln; j++)
		{
			if ( j >= length_poly )
				L[offsetL + j] = 0;
			else 
				L[offsetL + j] = M[offsetM + j];
		}
	}
}

/**
 * [EXPIRED] this guy is no longer used
 * After FFT based polynomial multiplication, most of the times we need to shrink
 * */
__device__ __global__ void list_truncate_for_invfft(sfixn *L, sfixn *K, sfixn ln)
{
	sfixn tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	K[tid] = L[2*(tid/ln)*ln + tid%ln];
	//__shared__ sfixn tmp;
	//sfixn bias = tid * PROCESS_PER_TRD;
	/*
	for (sfixn i = 0; i < PROCESS_PER_TRD; i++)
	{
		idx = bias + i;
		sfixn rem = idx%(ln);
		sfixn pos = idx/(ln);
		tmp = 2*pos*ln+rem;
		//K[idx] = tmp;
		//K[idx] = rem;
		K[idx] = L[tmp];
	}*/
}
// [EXPIRED]
__host__ void CPU_list_truncate_for_invfft(sfixn *L, sfixn *K, sfixn ln, sfixn num_poly)
{
	for (sfixn i = 0; i < num_poly; i++)
	{
		sfixn offsetK = i * ln;
		sfixn offsetL = i * 2 * ln;
		for (sfixn j = 0; j < ln; j++)
		{
			K[offsetK + j] = L[offsetL + j];
		}
	}
}

/**
 * shrink the polynomial on CPU 
 * */
__host__ void CPU_list_shrink_after_invfft(sfixn *L, sfixn *S, sfixn ln, sfixn length_result, sfixn num_poly)
{
	for (sfixn i = 0; i < num_poly; i++)
	{
		sfixn offsetS = length_result * i;	
		sfixn offsetL = ln * i;
		for (sfixn j = 0; j < length_result; j++)
		{
			S[offsetS + j] = L[offsetL + j];
		}
	}
}
/**
 * After FFT based polynomial, we sometimes need to shrink the result to make it sharp
 * 
 * Hence you could use this kernel to shrink L to S 
 * from length ln to length_result
 * The bound of S is length_layer
 * */
__device__ __global__ void list_shrink_after_invfft(sfixn *L, sfixn *S, sfixn ln, sfixn length_result, sfixn length_layer)
{
	sfixn tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if ( tid < length_layer)
	{
		sfixn ppos = tid / length_result;

		sfixn rem = tid % length_result;

		S[tid] = L[ppos * ln + rem];
	}
}

//////////////////////////////////////////////////////////////////////////
//BEGIN:list_shrink_after_invfft_tst
//////////////////////////////////////////////////////////////////////////

void list_shrink_after_invfft_tst(sfixn m, sfixn ln, sfixn length_result)
{
	sfixn k = m * ln;

	sfixn num_poly = m;
	
	sfixn *L = (sfixn *)malloc(num_poly * ln *sizeof(sfixn));
	sfixn *S = (sfixn *)malloc(num_poly * length_result * sizeof(sfixn));

	for (sfixn i = 0; i < k; i++)
	{
		L[i] = (i+1)%ln;
	}
	
	sfixn *S_GPU = (sfixn *)malloc(num_poly * length_result * sizeof(sfixn));

	sfixn *S_dev, *L_dev;
	
	cudaMalloc((void **)&S_dev, num_poly * length_result * sizeof(sfixn));
	cudaMalloc((void **)&L_dev, num_poly * ln * sizeof(sfixn));

	cudaMemcpy(L_dev, L, num_poly * ln * sizeof(sfixn), cudaMemcpyHostToDevice);

	list_shrink_after_invfft<<<k/N_TREAD+1, N_TREAD>>>(L_dev, S_dev, ln, length_result, m*length_result);

	cudaMemcpy(S_GPU, S_dev, num_poly * length_result * sizeof(sfixn), cudaMemcpyDeviceToHost);

	CPU_list_shrink_after_invfft(L, S, ln,length_result, num_poly);

	printf("L :=\n");
	for (sfixn i = 0; i < k; i++)
	{
		printf("%d ",L[i]);
	}
	printf(";\n");

	printf("S :=\n");
	for (sfixn i = 0; i < length_result * num_poly; i++)
	{
		printf("%d ",S[i]);
	}
	printf(";\n");


	printf("S_GPU :=\n");
	for (sfixn i = 0; i < length_result * num_poly; i++)
	{
		printf("%d ",S_GPU[i]);
	}
	printf(";\n");

	for (sfixn i = 0; i < length_result * num_poly; i++)
	{
		if( S_GPU[i] != S[i])
		{
			printf("Fail at S[%d]\n", i);
			return;
		}
	}
	
	printf("PASS\n");


	cudaFree(L_dev);
	cudaFree(S_dev);
}
//////////////////////////////////////////////////////////////////////////
//END:list_shrink_after_invfft_tst
//////////////////////////////////////////////////////////////////////////


void list_truncate_for_fft_tst()
{
	sfixn k = 1L << 10;

	sfixn *L = (sfixn *)malloc(sizeof(sfixn)*2*k);
	sfixn *S = (sfixn *)malloc(sizeof(sfixn)*k);

	sfixn *L_cpu = (sfixn *)malloc(sizeof(sfixn)*2*k);
	sfixn *S_cpu = (sfixn *)malloc(sizeof(sfixn)*k);


	for (sfixn i = 0; i < 2 * k; i++){
		L[i] = i%32;
		L_cpu[i] = L[i];
	}

	sfixn *L_dev, *S_dev;
	cudaMalloc((void **)&S_dev, k*sizeof(sfixn));
	cudaMalloc((void **)&L_dev, 2*k*sizeof(sfixn));

	cudaMemcpy(L_dev, L, 2*k*sizeof(sfixn), cudaMemcpyHostToDevice);
	
	list_truncate_for_invfft<<<k/N_TREAD,N_TREAD>>>(L_dev, S_dev,4 );

	cudaMemcpy(S, S_dev, k*sizeof(sfixn), cudaMemcpyDeviceToHost);
	
	CPU_list_truncate_for_invfft(L_cpu, S_cpu, 4, k/16); 

	for (sfixn i = 0; i < k; i++)
	{
		if (S[i] != S_cpu[i])
		{
			printf("Fail at S[%d]\n", i);
		}
	}

	printf("L := ");
	for (sfixn i = 0; i < 2*k; i++) {
		printf("%d ", L[i]);
	}
	printf(";\n");
	

	printf("S_GPU := ");
	for (sfixn i = 0; i < k; i++){
		printf("%d ", S[i]);
	}
	printf(";\n");
	printf("S_CPU := ");
	for (sfixn i = 0; i < k; i++){
		printf("%d ", S_cpu[i]);
	}
	printf(";\n");
	

	printf("PASS\n");
	cudaFree(L_dev);
	cudaFree(S_dev);
}
//////////////////////////////////////////////////////////////////////////
//BEGIN:list_pointwise_mul_tst
//////////////////////////////////////////////////////////////////////////
void list_pointwise_mul_tst(sfixn p, sfixn m, sfixn k)
{
	sfixn n = (1L << k);
	sfixn dim = n * m;
	sfixn *L = (sfixn *)malloc(sizeof(sfixn)*dim);
	sfixn *L_cpu = (sfixn *)malloc(sizeof(sfixn)*dim);

	for (sfixn i = 0; i < dim; i++) {
		L[i] = 3;
		L_cpu[i] = 3;
	}

	printf("M := ");
	for (sfixn i = 0; i < dim; i++) printf("%d ", L[i]);	
	printf(";\n");

	sfixn *L_dev;
	cudaMalloc((void **)&L_dev, dim*sizeof(sfixn));

	cudaMemcpy(L_dev, L, dim*sizeof(sfixn), cudaMemcpyHostToDevice);
	
	list_pointwise_mul<<<(dim/512)+1,512>>>(L_dev, n, p, (double)1/p, dim);

	cudaMemcpy(L, L_dev, dim*sizeof(sfixn), cudaMemcpyDeviceToHost);

	CPU_list_pointwise_mul(L_cpu, n, m , p, (double)1/p);

	printf("L := ");
	for (sfixn i = 0; i < dim; i++) printf("%d ", L[i]);
	printf(";\n");
	
	printf("L_CPU := ");
	for (sfixn i = 0; i < dim; i++) printf("%d ", L_cpu[i]);
	printf(";\n");


	for (sfixn i = 0; i < dim; i++)
	{
		if(L_cpu[i] != L[i])
		{
			printf("Fail at L[%d]\n");
			return;
		}
	}

	printf("PASS\n");

	cudaFree(L_dev);
}
//////////////////////////////////////////////////////////////////////////
//END:list_pointwise_mul_tst
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
//BEGIN:list_expand_to_fft_tst
//////////////////////////////////////////////////////////////////////////
void list_expand_to_fft_tst(sfixn m, sfixn k)
{
	sfixn length_poly = 1L << k + 1;

	sfixn n = length_poly * m;
	sfixn s = length_poly + length_poly - 1;
	sfixn e = ceiling_log2(s);
	sfixn ln = (1L << e);

	sfixn *M = (sfixn *)malloc(sizeof(sfixn)*length_poly*m);
	sfixn *L = (sfixn *)malloc(sizeof(sfixn)*ln*m);

	sfixn *M_cpu = (sfixn *)malloc(sizeof(sfixn)*length_poly*m);
	sfixn *L_cpu = (sfixn *)malloc(sizeof(sfixn)*ln*m);

	for (sfixn i = 0; i < m; i++) {
		for (sfixn j = 0; j < length_poly; j++)
		{
			M[i*length_poly + j] = j;
			M_cpu [i*length_poly + j] = j;
		}
	}

	sfixn *M_dev, *L_dev;
	cudaMalloc((void **)&M_dev, n*sizeof(sfixn));
	cudaMalloc((void **)&L_dev, ln*m*sizeof(sfixn));

	cudaMemcpy(M_dev, M, length_poly*m*sizeof(sfixn), cudaMemcpyHostToDevice);
	
	list_expand_to_fft<<<n/N_TREAD+1,N_TREAD>>>(M_dev, L_dev, length_poly, ln, 0, ln*m);

	cudaMemcpy(L, L_dev, ln*m*sizeof(sfixn), cudaMemcpyDeviceToHost);

	CPU_list_expand_to_fft(M_cpu, L_cpu, n, length_poly, ln, 0);

	printf("M := \n");
	for (sfixn i = 0; i < n; i++) {
		printf("%d ", M[i]);
	}
	printf(";\n");

	printf("C := \n");
	for (sfixn i = 0; i < n; i++){
		printf("%d ", L_cpu[i]);
	}
	printf(";\n");
	
	printf("L := \n");
	for (sfixn i = 0; i < n; i++){
		printf("%d ", L[i]);
	}
	printf(";\n");
	
	for (sfixn i = 0; i < n; i++){
		if ( L[i] != L_cpu[i])
		{
			printf("fail at L[%d]\n", i);
			return;
		}
	}
	printf("PASS\n");

	free(M);
	free(L);
	free(M_cpu);
	free(L_cpu);

	cudaFree(L_dev);
	cudaFree(M_dev);
}
///////////////////////////////////////////////////////////////////////////
//END:list_expand_to_fft_tst
///////////////////////////////////////////////////////////////////////////
/*
int main(int argc, char **argv)
{
	//list_pointwise_mul_tst();

	//list_expand_to_fft_tst();

	list_truncate_for_fft_tst();
	
	//list_shrink_after_invfft_tst();

}
*/

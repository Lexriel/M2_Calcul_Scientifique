#include "../include/list_poly_rev.h"
#include <stdio.h>
#include "../include/types.h"
#include "../include/printing.h"

/**
 * Reverse a list of polynomials in place
 *
 * F is the list of polynomials
 * length_poly: length of each polynomials
 * bound: length of F
 * */
__device__ __global__ void poly_rev(sfixn *F, sfixn length_poly, sfixn bound)
{
	sfixn tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(tid<bound)
	{
		sfixn pos = tid /length_poly;
		sfixn offset = pos * length_poly;
		sfixn rem = tid % length_poly;

		if(offset + length_poly-rem+1 < bound && rem <= (length_poly/2))
		{
			sfixn a = F[offset + rem];
			F[offset + rem] = F[offset + length_poly-rem-1];
			F[offset + length_poly-rem-1] = a;
		}
	}
}
/**
 * Reverse a list of polynomials by copy
 *
 * F is the list of polynomials
 * length_poly: length of each polynomials
 * G: the output
 * bound: length of F/G
 * */
__device__ __global__ void poly_rev_dup(sfixn *F, sfixn length_poly,sfixn *G, sfixn bound)
{
	sfixn tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(tid<bound)
	{
		sfixn pos = tid / length_poly;
		sfixn rem = tid % length_poly;
		G[tid] = F[pos * length_poly + length_poly-1-rem];
	}
}


// Test function
void poly_rev_tst(sfixn m, sfixn length)
{
	sfixn *F = (sfixn *)malloc(sizeof(sfixn)*length*m);

	for (sfixn i = 0; i < m; i++)
	{
		for (sfixn j = 0; j < length; j++)
		{
			F[i*length + j] = j+1;
		}
	}

	sfixn *G = (sfixn *)malloc(sizeof(sfixn)*length*m);
	sfixn *Gdup = (sfixn *)malloc(sizeof(sfixn)*length*m);

	sfixn *F_dev,*G1_dev,*Gd_dev;
	cudaMalloc((void **)&F_dev, sizeof(sfixn)*length*m);
	cudaMalloc((void **)&G1_dev, sizeof(sfixn)*length*m);
	cudaMalloc((void **)&Gd_dev, sizeof(sfixn)*length*m);

	cudaMemcpy(F_dev, F, sizeof(sfixn)*length*m, cudaMemcpyHostToDevice);
	cudaMemcpy(G1_dev,F_dev, sizeof(sfixn)*length*m,cudaMemcpyDeviceToDevice);

	sfixn dim = length*m;
	poly_rev<<<dim/512+1,512>>>(G1_dev,length,dim);
	poly_rev_dup<<<dim/512+1,512>>>(F_dev,length,Gd_dev,dim);

	cudaMemcpy(G,G1_dev,sizeof(sfixn)*length*m, cudaMemcpyDeviceToHost);
	cudaMemcpy(Gdup,Gd_dev,sizeof(sfixn)*length*m, cudaMemcpyDeviceToHost);

	for (sfixn i = 0; i < m; i++)
	{
		printf("F : = ");print_poly(length-1,F,'x');printf(";\n");
		printf("G1 : = ");print_poly(length-1,G,'x');printf(";\n");
		printf("Gd : = ");print_poly(length-1,Gdup,'x');printf(";\n");
	}
	cudaFree(F_dev);
	cudaFree(G1_dev);
	cudaFree(Gd_dev);
	free(F);
	free(G);
	free(Gdup);
}
/*
int main(int argc, char * argv[])
{
	poly_rev_tst(atoi(argv[1]),atoi(argv[2]));
	return 0;
}
*/


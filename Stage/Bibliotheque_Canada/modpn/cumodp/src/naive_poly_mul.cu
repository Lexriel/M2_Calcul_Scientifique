
#include "../include/naive_poly_mul.h"
/**
 * The most straight forward polynomial multiplication, works on both CPU and GPU
 */
__host__ __device__ void mul_ker(sfixn *A, sfixn *B, sfixn n, sfixn *C, sfixn p, double pinv)
{
	sfixn i;
	for(i = 0; i < (n+n-1); i++)
	{
		C[i] = 0;
		for (sfixn k = 0; k < n; k++)
			for (sfixn l = 0; l < n; l++)
				if(k+l==i){
					//C[i] += A[k]*B[l];
					sfixn mm = mul_mod(A[k],B[l],p,pinv);
					C[i] = add_mod(C[i], mm, p);
				}
	}
}
/**
 * An optimized version of polynomial multiplication
 * Segment the resulting polynomial to two parts
 * and then we do the two parts at the same time
 * NOTE: n is the size of the polynomials, not the degree!
 * the two polynomials should be in the same size for now
 *
 * TODO: tune it! idea: segment the for loop to let every
 * thread start at different places.
 */
__device__ __global__ void mul_eff_ker(sfixn *A, sfixn *B, sfixn n, sfixn *C, sfixn p, double pinv)
{
	//TODO: (later) to decide if we have to multiply the blockDim.x
	sfixn tid = threadIdx.x+blockIdx.x;
	C[tid] = 0;
	C[tid+n] = 0;
	/*Most naive version
	for (sfixn i = 0; i <= tid; i++)
	{
		//C[tid] = C[tid] + A[i]*B[tid-i];
		sfixn mm =  mul_mod(A[i], B[tid-i],p, pinv);
		C[tid] = add_mod(C[tid], mm, p);
	}
	most naive version*/
	//less naive version
	for (sfixn i = (tid/2)+1; i <= tid; i++)
	{
		sfixn mm =  mul_mod(A[i], B[tid-i],p, pinv);
		C[tid] = add_mod(C[tid], mm, p);
	}
	for (sfixn i = (tid/2); i >= 0; i--)
	{
		sfixn mm =  mul_mod(A[i], B[tid-i],p, pinv);
		C[tid] = add_mod(C[tid], mm, p);
	}
	//end of less naive version


	if(tid < n)
	{
		for(sfixn i= ((n-tid)/2)+1 ; i < (n-tid); i++)
		{	
			sfixn mm = mul_mod(A[tid+i], B[n-i], p, pinv);
			C[tid+n] = add_mod(C[tid+n], mm, p);
		}
		for(sfixn i= ((n-tid)/2) ; i >= 1; i--)
		{	
			sfixn mm = mul_mod(A[tid+i], B[n-i], p, pinv);
			C[tid+n] = add_mod(C[tid+n], mm, p);
		}
	}
}

/**
 * It's a part of the test actually.... 
 *
 */
void mul_dev(sfixn *A, sfixn *B, sfixn n, sfixn *C, sfixn p)
{
	sfixn *A_dev, *B_dev, *C_dev;

	double pinv = 1 / (double)p;

	//Allocating Memory
	cudaMalloc( (void **)&A_dev, n*sizeof(sfixn));
	cudaMalloc( (void **)&B_dev, n*sizeof(sfixn));
	cudaMalloc( (void **)&C_dev, (n+n-1)*sizeof(sfixn));

	//Copy from host to device
	cudaMemcpy( A_dev, A, n*sizeof(sfixn), cudaMemcpyHostToDevice);
	cudaMemcpy( B_dev, B, n*sizeof(sfixn), cudaMemcpyHostToDevice);

	//Only one threadblock is used
	mul_eff_ker<<<1,n>>>(A_dev, B_dev, n,C_dev,p,pinv);
	
	//Copy from device to host, we need C_dev only
	cudaMemcpy( C, C_dev, (n+n-1)*sizeof(sfixn), cudaMemcpyDeviceToHost);

	//Freeing device memory
	cudaFree(A_dev);
	cudaFree(B_dev);
	cudaFree(C_dev);

}
////////////////////////////////////////////////////////////////
//BEGIN:naive_poly_mul_tst
////////////////////////////////////////////////////////////////
void mul_host_tst(sfixn n, sfixn p)
{
	sfixn *A, *B, *C,*D;
	A = (sfixn*)malloc(n*sizeof(sfixn));
	B = (sfixn*)malloc(n*sizeof(sfixn));
	C = (sfixn*)malloc((n+n-1)*sizeof(sfixn));
	D = (sfixn*)malloc((n+n-1)*sizeof(sfixn));

	sfixn i;
	for(i=1;i<=n;i++)
	{
		A[i-1]=i;
		B[i-1]=i;
	}
	
	double pinv = 1/(double)p;
	mul_dev(A,B,n,C,p);
	mul_ker(A,B,n,D,p,pinv);

	printf("A := "); print_poly(n-1, A, 'x'); printf(":\n");
	printf("B := "); print_poly(n-1, B, 'x'); printf(":\n");
	printf("C := "); print_poly(n+n-2, C, 'x'); printf(":\n");
	printf("D := "); print_poly(n+n-2, D, 'x'); printf(":\n");

	free(A);
	free(B);
	free(C);
}
////////////////////////////////////////////////////////////////
//END:naive_poly_mul_tst
////////////////////////////////////////////////////////////////

#include "../include/subproduct_tree.h"
#include "../include/list_naive_poly_mul.h"

/**
 * The kernel to apply the naive multiplication
 * It follows the most naive multiplication.
 * It updates the subproduct tree array M
 *
 * M: the 1-D array encoding the subproduct tree
 * length_poly: length of each polynomials of the current layer
 * length_result: length of each polynomials of the next layer
 * start_offset: starting point of current layer in M
 * result_offset: starting point of next layer in M
 * p: the prime
 * */
__device__ __global__ void list_poly_mul_ker
		(sfixn *M, sfixn length_poly, sfixn length_result, sfixn start_offset, sfixn result_offset, sfixn p, double pinv)
{
	sfixn tid = threadIdx.x + blockIdx.x * blockDim.x;

	sfixn offset_A = start_offset+tid*2*length_poly;
	sfixn offset_B = start_offset+tid*2*length_poly+length_poly;


	for (sfixn i = 0; i < length_result; i++)
		M[result_offset+tid*length_result+i] = 0;


	for (sfixn k = 0; k < length_poly; k++)
	{
		for (sfixn l = 0; l < length_poly; l++)
		{
			sfixn mm = mul_mod(M[offset_A+k], M[offset_B+l], p, pinv);
			M[result_offset+tid*length_result+k+l] = add_mod(M[result_offset+tid*length_result+k+l], mm, p);

		}		
	}


	
	



	/*

	// Compute the multiplication, then write back to global memory
	for (sfixn i = 0; i < length_result; i++)
	{
		M[result_offset+tid*length_result+i] = 0;
		
		for (sfixn k = 0; k < length_poly; k++)
			for (sfixn l = 0; l < length_poly; l++)
			{
				if(k+l == i)
				{
					sfixn mm = mul_mod(M[offset_A+k], M[offset_B+l], p, pinv);
					M[result_offset+tid*length_result+i] = 
						add_mod(M[result_offset+tid*length_result+i], mm, p);
				}
			}
	}
	*/
}

/**
 * The kernel to apply less-naive multiplication
 *
 * M: the 1-D array encoding the subproduct tree
 * length_poly: length of each polynomials of the current layer
 * length_result: length of each polynomials of the next layer
 * start_offset: starting point of current layer in M
 * result_offset: starting point of next layer in M
 * p: the prime
 * USED ONLY COMPARATIVE TESTING
 * */
__device__ __global__ void list_poly_mul_ker_higher
		(sfixn *M, sfixn length_poly, sfixn length_result, sfixn start_offset, sfixn result_offset, sfixn p, double pinv)
{
	// Now bid is the position of the polynomials
	// Hence:
	// offset_A = start_offset + bid * 2 * length_poly
	// offset_B = start_offset + bid * 2 * length_poly + length_poly
	// offset_C = result_offset +  bid * length_result
	sfixn bid = blockIdx.x;
	
	sfixn offset_A = start_offset + bid * 2 * length_poly;
	sfixn offset_B = start_offset + bid * 2 * length_poly + length_poly;
	
	sfixn offset_C = result_offset + bid * length_result;

	// Now tid is the position of coefficients
	// Hence: 
	sfixn tid = threadIdx.x;
	
	M[offset_C+tid] = 0;

	for (sfixn k = 0; k < length_poly; k++)
		for (sfixn l = 0; l < length_poly; l++)
		{
			if(k+l == tid)
			{
				sfixn mm = mul_mod(M[offset_A+k], M[offset_B+l], p, pinv);
				M[offset_C + tid] = add_mod(M[offset_C + tid], mm, p);
			}
		}
}
/////////////////////////////////////////////////////////////////////////////
//BEGIN:list_naive_poly_mul_tst
/////////////////////////////////////////////////////////////////////////////
//
//This is to compare the results of the two kernel, just to compute the first layer
//in the subproduct tree
void list_poly_mul_tst(sfixn p, sfixn k)
{
	//sfixn p = 4696762049;
	sfixn n = (1L << k);
	
	sfixn sizeofM = get_layer_size(k,1) + get_layer_size(k,2);

	sfixn *M = (sfixn *)malloc(sizeofM*sizeof(sfixn));

	printf("Input := Arruay([");
	
	for (sfixn i = 1; i <= n; i++)
	{
		if( (i-1)%2 == 0)
			M[i-1] = i;
		else
		{
			M[i-1] = 1;
			if(i>1)
			{
				sfixn *F = (sfixn *)malloc(2*sizeof(sfixn));
				F[0] = M[i-2];
				F[1] = M[i-1];
				printf("("); print_poly(1, F, 'x');
				if(i<n)printf("),");
				if(i==n)printf(")])");
				free(F);
			}
		}
	}
	printf(";\n");

	sfixn *M_dev, *M_dev_low;
	cudaMalloc((void**)&M_dev, sizeofM*sizeof(sfixn));
	cudaMalloc((void**)&M_dev_low, sizeofM*sizeof(sfixn));
	cudaMemcpy(M_dev, M, n*sizeof(sfixn), cudaMemcpyHostToDevice);
	cudaMemcpy(M_dev_low, M, n*sizeof(sfixn), cudaMemcpyHostToDevice);

	sfixn num_muls =  get_layer_size(k, 2) / get_polylength_on_layer(2);

	list_poly_mul_ker_higher<<<num_muls,get_polylength_on_layer(2)>>>(M_dev, get_polylength_on_layer(1), get_polylength_on_layer(1+1), get_layer_offset(k, 1),get_layer_offset(k, 2),p, 1/(double)p);
	list_poly_mul_ker<<<num_muls,8>>>(M_dev_low, get_polylength_on_layer(1), get_polylength_on_layer(1+1), get_layer_offset(k, 1),get_layer_offset(k, 2),p, 1/(double)p);

	sfixn *M_low = (sfixn *)malloc(sizeofM*sizeof(sfixn));
	cudaMemcpy(M, M_dev, sizeofM*sizeof(sfixn), cudaMemcpyDeviceToHost);
	cudaMemcpy(M_low, M_dev_low, sizeofM*sizeof(sfixn), cudaMemcpyDeviceToHost);

	cudaFree(M_dev);
	cudaFree(M_dev_low);

	printf("L := "); print_vector(sizeofM, M_low); 
	printf("M := "); print_vector(sizeofM, M); 
	for (sfixn i = 0; i < sizeofM; i++)
	{
		if (M[i] != M_low[i])
		{
			printf("Error at %d\n", i);
			return;
		}
	}
	printf("PASS\n");

		
	printf("\n");
}
//////////////////////////////////////////////////////////////////////////////
//END:list_naive_poly_mul_tst
//////////////////////////////////////////////////////////////////////////////
/*
int main(int argc, char **argv)
{
	list_poly_mul_tst(469762049,4);
	return 0;
}
*/

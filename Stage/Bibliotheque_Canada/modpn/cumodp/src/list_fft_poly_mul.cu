#include "../include/list_fft_poly_mul.h"

/**
 * list of FFT based polynomial multiplications in subproduct tree (naive version)
 *
 * M_dev: the 1-D array encoding the subproduct tree
 * length_poly: the length of each polynomial of the current layer
 * O_dev: the 1-D array encoding the output layer (next layer) of the subproduct tree
 * length_result: the length of each polynomial of the next layer
 * num_polys: number of polynomials of current layer
 * start_offset_M: the offset of the starting point of current layer(the layer to be multiplied) in M_dev
 * p: the prime
 * pinv: the inv_mod of p
 * */
void list_fft_poly_mul_dev
	(sfixn *M_dev, sfixn length_poly, sfixn *O_dev, sfixn length_result, sfixn num_polys, sfixn start_offset_M, sfixn p, double pinv)
{
	sfixn e = ceiling_log2(length_result);
	sfixn ln = (1L << e);
	
	sfixn *L_dev;
	sfixn dim = ln * num_polys;
	sfixn num_muls = num_polys/2;

	cudaMalloc((void **)&L_dev, dim * sizeof(sfixn));
	cudaThreadSynchronize();

	// TODO: find faster algorithm to expand the result
	list_expand_to_fft<<<dim/N_TREAD+1, N_TREAD>>>(M_dev, L_dev,length_poly,ln,start_offset_M, dim);
	cudaThreadSynchronize();

	sfixn w = primitive_root(e, p);
	list_stockham_dev(L_dev, num_polys, e, w, p);
	cudaThreadSynchronize();

	list_pointwise_mul<<<dim/N_TREAD+1, N_TREAD>>>(L_dev, ln, p, (double)1/p, dim);
	cudaThreadSynchronize();

	sfixn winv, ninv;
	winv = inv_mod(w, p);
	ninv = inv_mod(ln, p);

	list_stockham_dev(L_dev, num_polys, e, winv, p);
	cudaThreadSynchronize();
	
	list_inv_mul_ker<<<dim/N_TREAD+1, N_TREAD>>>(L_dev, ninv, num_polys, p, pinv,dim);
	cudaThreadSynchronize();
	
	list_shrink_after_invfft<<<dim/N_TREAD+1, N_TREAD>>>(L_dev, O_dev, ln*2, length_result, num_muls * length_result);
	cudaThreadSynchronize();

	cudaFree(L_dev);
}

/**
 * This kernel is for the optimized version of FFT-based polynomial multiplication
 * It add up the FFT part and the non FFT part
 * And set the leading coefficient to 1
 *
 * M: the 1-D array encoding subproduct tree
 * length_poly: length of each polynomial in current layer
 * L: the FFT based part (the result of FFT)
 * ln: the length of each FFT result
 * O: the output layer (next layer)
 * length_result: the length of each polynomials in the next layer
 * num_polys: number of polynomials in current layer
 * start_offset: the starting point of the current layer in M
 * p: the prime
 * pinv: the inversion of p
 *
 * */

__device__ __global__ void list_add_and_set_1 
	(sfixn *M, sfixn length_poly, sfixn *L, sfixn ln, sfixn *O, sfixn length_result, sfixn num_polys, sfixn start_offset, sfixn p, double pinv)
{
	sfixn tid = threadIdx.x + blockDim.x * blockIdx.x;

	if (tid < length_result * num_polys)
	{
		O[tid] = 0;
		// for each O[tid] it checks the position first
		sfixn pos = tid / length_result;
		sfixn rem = tid % length_result;

		// if rem = length_result - 1 then it's one and done
		if (rem == length_result - 1)
			O[tid] = 1;
		// if rem < length_poly it keeps the value
		if (rem < length_poly)
		{
			//Do Nothing just copy
			sfixn L_idx = pos * ln;
			O[tid] = L[L_idx + rem];
		}
		// else it picks one A[rem - length_poly] and a B[rem - length_poly]
		if (rem >= length_poly - 1  && rem < length_result -1) 
		{
			sfixn L_idx = pos * ln;
			sfixn o = L[L_idx + rem];
			
			sfixn idx = start_offset + pos *  ln;
			sfixn a = M [idx + rem - length_poly];
			sfixn b = M [idx + rem];
			sfixn c = add_mod(a,o,p);
			O[tid] = add_mod(b,c,p);
		}
	}
}

/**
 * The optimized version of list of FFT-based polynomial multiplication
 * It takes shorter FFTs and plays tricks. 
 *
 * M_dev: the 1-D array of subproduct tree
 * length_poly: length of each polynomial of current layer
 * O_dev: the output layer(the next layer). This contains the output, M_dev
 *        is not modified
 * length_result: the length of each polynomial of next layer
 * num_polys: the number of polynomials of the current layer
 * start_offset_M: the starting point of current layer in M
 * p: the prime
 * pinv: the inverse of prime
 * */
void list_fft_poly_mul_eff
	(sfixn *M_dev, sfixn length_poly, sfixn *O_dev, sfixn length_result, sfixn num_polys, sfixn start_offset_M, sfixn p, double pinv)
{
	// The part needs FFT is actually the 2^h hence the expanded size should be ceiling_log2(length_result-1)
	// that's literally 2*length_poly
	sfixn e = ceiling_log2(length_result-1);
	sfixn ln = (1L << e);
	
	// Now what I need to do is to extract the parts that really needs FFT and do the fft as normal
	sfixn *L_dev;
	sfixn dim = ln * num_polys;
	//sfixn num_muls = num_polys/2;

	cudaMalloc((void **)&L_dev, dim * sizeof(sfixn));

	list_expand_to_fft<<<dim/N_TREAD+1, N_TREAD>>>(M_dev, L_dev,length_poly-1,ln,start_offset_M, dim);
	sfixn w = primitive_root(e, p);

	list_stockham_dev(L_dev, num_polys, e, w, p);

	list_pointwise_mul<<<dim/N_TREAD, N_TREAD>>>(L_dev, ln, p, (double)1/p, dim);

	sfixn winv, ninv;
	winv = inv_mod(w, p);
	ninv = inv_mod(ln, p);
	list_stockham_dev(L_dev, num_polys, e, winv, p);
	
	list_inv_mul_ker<<<dim/N_TREAD+1, N_TREAD>>>(L_dev, ninv, num_polys, p, pinv,dim);

	//list_shrink_after_invfft<<<dim/N_TREAD, N_TREAD>>>(L_dev, O_dev, ln, length_result, num_muls * length_result);
	
	// Then point-wisely add the A' and B' and then shift them
	// Here shifting actually means add up everything from the middle
	// point-wisely add A' and B' to L_dev
	// Set leading coeff to be 1 and add up everything
	list_add_and_set_1<<<dim/N_TREAD+1,N_TREAD>>>(M_dev, length_poly,L_dev, ln, O_dev, length_result, num_polys, start_offset_M, p, pinv);

	cudaFree(L_dev);
}

/**
 * [EXPIRED]
 * The most straightforward FFT-based polynomial multiplication 
 * */
void fft_poly_mul_dev
	(sfixn *M_dev, sfixn length_poly, sfixn *O_dev, sfixn length_result, sfixn num_polys, sfixn start_offset_M, sfixn p, double pinv)
{
	sfixn *A_dev, *B_dev, *C_dev;
	
	sfixn e = ceiling_log2(length_result);

	sfixn ln = (1L << e);
	sfixn memsize = (sizeof(sfixn) << e);

	cudaMalloc((void **)&A_dev, memsize);
	cudaMalloc((void **)&B_dev, memsize);
	cudaMalloc((void **)&C_dev, sizeof(sfixn)*length_result);

	for (sfixn j = 0; j < num_polys/2; j++)
	{
		cudaMemcpy(A_dev, &(M_dev[start_offset_M+j*2*length_poly]), length_poly*sizeof(sfixn), cudaMemcpyDeviceToDevice);
		cudaMemcpy(B_dev, &(M_dev[start_offset_M+j*2*length_poly+length_poly]), length_poly*sizeof(sfixn), cudaMemcpyDeviceToDevice);

		expand_to_fft_dev(ln, length_poly, A_dev);
		expand_to_fft_dev(ln, length_poly, B_dev);
		stockham_poly_mul_dev(ln, e, A_dev, B_dev, p);

		cudaMemcpy(C_dev, A_dev, length_result*sizeof(sfixn), cudaMemcpyDeviceToDevice);
		cudaMemcpy(&(O_dev[j*length_result]),C_dev, length_result*sizeof(sfixn), cudaMemcpyDeviceToDevice);
	}
	cudaFree(A_dev);
	cudaFree(B_dev);
}
/////////////////////////////////////////////////////////////////////////////////////
//BEGIN:list_fft_poly_mul_tst
/////////////////////////////////////////////////////////////////////////////////////
void list_fft_poly_mul_tst(sfixn p, sfixn m, sfixn k)
{
	sfixn length_poly = (1L << k) + 1;
	sfixn *M = (sfixn *)malloc(sizeof(sfixn) * m * length_poly); 
	sfixn length_result = length_poly * 2 -1;
	sfixn *R_list = (sfixn *)malloc(sizeof(sfixn) *(m/2) * length_result);
	sfixn *R_stable = (sfixn *)malloc(sizeof(sfixn) * (m/2) * length_result);
	
	for (sfixn i = 0; i < m; i++)
	{
		for (sfixn j = 0; j < length_poly; j++)
		{
			M[i*length_poly+j] = i%p;
			if(j == length_poly -1)
				M[i*length_poly + j] = 1; 
		} 
	}

	sfixn *R_list_dev, *R_stable_dev;
	sfixn *M_dev;

	cudaMalloc((void **)&M_dev, sizeof(sfixn) * m * length_poly);
	cudaMalloc((void **)&R_list_dev, sizeof(sfixn) * (m/2) *  length_result);
	cudaMalloc((void **)&R_stable_dev, sizeof(sfixn) * (m/2) *  length_result);

	cudaMemcpy(M_dev, M, sizeof(sfixn) *  m * length_poly, cudaMemcpyHostToDevice);

	list_fft_poly_mul_dev(M_dev, length_poly, R_list_dev, length_result, m, 0, p, (double)1/p);
	fft_poly_mul_dev(M_dev, length_poly, R_stable_dev, length_result, m, 0, p, (double)1/p);

	cudaMemcpy(R_list, R_list_dev, sizeof(sfixn) *(m/2)* length_result, cudaMemcpyDeviceToHost);
	cudaMemcpy(R_stable, R_stable_dev, sizeof(sfixn) *(m/2)* length_result, cudaMemcpyDeviceToHost);

	cudaFree(M_dev);
	cudaFree(R_list_dev);
	cudaFree(R_stable_dev);

	sfixn errs = 0;
	for (sfixn i = 0; i < ((m/2) * length_result); i++)
	{
		if (R_list[i] != R_stable[i])
		{
			//printf("Error at %d\n",i);
			errs++;
			//return;
		}
	}
	if (errs == 0)printf("PASS\n");
	else printf("Error: %d Errors among %d points\n", errs, (m/2) * length_result);
}
//////////////////////////////////////////////////////////////////////////////////////
//END:list_fft_poly_mul_tst
//////////////////////////////////////////////////////////////////////////////////////
/*
int main(int argc, char * argv[])
{
	list_fft_poly_mul_tst(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
	return 0;
}
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "../include/subproduct_tree.h"
#include "../include/list_naive_poly_mul.h"
#include "../include/list_pointwise_mul.h"
#include "../include/list_inv_stockham.h"
#include "../include/inlines.h"

#include "../include/power_inversion.h"


/**
 * The initial step for list of power series inversion
 * to compute g1
 *
 * L: the input
 * G: output g1
 * length_I: the bound of G (2*num_poly)
 * length_poly: the length of each polynomial in L
 * p: the prime
 *
 * */
__device__ __global__ void reset_ps(sfixn *L, sfixn *G, sfixn length_I, sfixn length_poly,sfixn p)
{
	sfixn tid = threadIdx.x + blockIdx.x * blockDim.x;
	sfixn sn_po = tid / 2;
	if(tid < length_I){
		//if(tid % 2 ==0) G[tid] = 1;
		//else G[tid]=0;
		G[tid] = neg_mod(L[sn_po*length_poly + tid%2],p);
		//G[tid] = 1;
		if(tid%2==0)G[tid] = add_mod(2,G[tid],p);
	}
}

/**
 * Expand the polynomial by filling 0
 * 
 * gg: the polynomials in sharp storage
 * loc: the output
 * l_g: the length of each polynomial in gg
 * l_f: the length after expanding
 * length_loc: the bound of loc
 * */
__device__ __global__ void ps_expand_to_i(sfixn *gg, sfixn *loc, sfixn l_g, sfixn l_f, sfixn length_loc)
{
	sfixn tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < length_loc)
	{
		sfixn pos = tid / l_f;
		sfixn rem = tid % l_f;
		if(rem < l_g)
			loc[tid] = gg[pos * l_g + rem];
		else
			loc[tid] = 0;
	}
}
// Test for the ps_expand_to_i 
void ps_expand_to_i_tst(sfixn i,sfixn n)
{
	sfixn l_g = (1L << (i-1));
	sfixn l_f = (1L << i);
	sfixn *A = (sfixn *)malloc(n*sizeof(sfixn)*l_g);
	sfixn *T = (sfixn *)malloc(n*sizeof(sfixn)*l_f);

	for (sfixn j = 0; j < n*l_g; j++)
	{
		A[j] = i;
	}
	printf("A := ");print_vector(n*l_g, A);
	sfixn *A_dev, *T_dev;
	cudaMalloc((void **)&A_dev, n*sizeof(sfixn)*l_g);
	cudaMalloc((void **)&T_dev, n*sizeof(sfixn)*l_f);
	cudaMemcpy(A_dev, A, n*sizeof(sfixn)*l_g, cudaMemcpyHostToDevice);

	ps_expand_to_i<<<(n*l_g)/N_TREAD+1, N_TREAD>>>(A_dev, T_dev, l_g, l_f, l_f*n);

	cudaMemcpy(T, T_dev, n*sizeof(sfixn)*l_f, cudaMemcpyDeviceToHost);
	cudaFree(A_dev);
	cudaFree(T_dev);

	printf("T := ");print_vector(n*l_f,T);
}

/**
 * Apply the pointwise operation of g = 2g - f * (g ^2) the minus part 
 *
 * L: the input f*g, then the output G
 * G: the original G
 * length: the bound of L/G
 * p: the prime
 * */
__device__ __global__ void ps_add_2g_minus(sfixn *L, sfixn *G, sfixn length, sfixn p, double pinv)
{
	sfixn tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < length)
	{
		sfixn mm = mul_mod(G[tid],2,p,pinv);
		sfixn nn = neg_mod(L[tid],p);
		L[tid] = add_mod(nn, mm, p);
	}
}
/**
 * Naive polynomial mulitplication
 *
 * A: the list of polynomials to be multiplied 
 * length_a: the length of each polynomials in A
 * B: the other list of polynomials to be multiplied
 * length_b: the length of each polynomials in B
 * C: the result
 * length_c: the length of each result
 * p: the prime
 *
 * */
__device__ __global__ void ps_list_poly_mul(sfixn *A, sfixn length_a, sfixn *B, sfixn length_b, sfixn *C, sfixn length_c, sfixn p, double pinv)
{
	sfixn bid = blockIdx.x;
	sfixn tid = threadIdx.x;

	sfixn startA = bid * length_a;
	sfixn startB = bid * length_b;
	sfixn startC = bid * length_c;

	if(tid <  length_c)
	{
		C[startC+tid]=0;
		for (sfixn i = 0; i<length_a; i++)
			for (sfixn j=0; j<length_b; j++)
				if(i+j == tid)
				{
					sfixn mm = mul_mod(A[startA+i],B[startB+j], p, pinv);
					C[startC+tid] = add_mod(C[startC+tid],mm, p);
				}
	}
}


/**
 * Pointwise square kernel
 *
 * A: the input/output
 * totalLength: the bound of A
 * p: the prime
 * */
__device__ __global__ void ps_pointwise_sq(sfixn *A, sfixn totalLength, sfixn p, double pinv)
{
	sfixn tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < totalLength)
		A[tid] = mul_mod(A[tid],A[tid],p,pinv);
}


/**
 * Pointwise multiplication
 *
 * A: the input/output
 * B: the other input
 * length: bound of A/B
 * p: the prime
 * */
__device__ __global__ void ps_pointwise_mul(sfixn *A, sfixn *B, sfixn length, sfixn p, double pinv)
{
	sfixn tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < length)
	{
		A[tid] = mul_mod(A[tid],B[tid],p,pinv);
	}
}


/**
 * Driver function of list of power series inversion
 *
 * Given a positive integers ,
 * for each input polynomial f 
 * compute g such that  f * g  = 1 mod x^l
 * We assume that all the f have the same size
 *
 * L_dev: The input list of polynomials f
 * I_dev: the resulting list of powerseries inversion (polynomials g)
 * length_L: the bound of L_dev (size of the whole array L_dev)
 * num_poly: number of polynomials in L_dev
 * l: the l in formula, the precision
 * p: the prime
 * */
void list_power_series_inversion(sfixn *L_dev, sfixn *I_dev, sfixn length_L, sfixn num_poly, sfixn l, sfixn p)
{
	sfixn length_per_poly = length_L/num_poly;
	
	sfixn r = ceiling_log2(l);
	//printf("l=%d, r=%d\n",l,r);
	sfixn ln = (1L << r); 
	double pinv = (double)1/p;
	
	sfixn dim;

	// gg_dev is the place we store our result
	// not for the final power series inversion, but also the g_{i}s during the whole process
	sfixn *gg_dev;
	cudaMalloc((void **)&gg_dev, ln*num_poly*sizeof(sfixn));

	sfixn *g_left, *g_right, *g_sq;
	for (sfixn i = 1; i <= r; i++)
	{
		sfixn l_f = (1L << i);

		if (i == 1)
		{
			// if i = 1 the first layer is just the -f mod x^2
			// compute total number of points
			dim = num_poly*2;
			reset_ps<<<dim/N_TREAD+1,N_TREAD>>>(L_dev, gg_dev, dim, length_per_poly,p);
			continue;
		}
		// we extract g_{i-1} from gg_dev first, make local copy
		sfixn l_g = (1L << (i-1));
		dim = l_f * num_poly;
		cudaMalloc((void**)&g_left, dim*sizeof(sfixn));
		cudaMalloc((void**)&g_right, dim*sizeof(sfixn));
		cudaMalloc((void**)&g_sq, dim*sizeof(sfixn));
		ps_expand_to_i<<<dim/N_TREAD+1,N_TREAD>>>(gg_dev,g_left, l_g, l_f, dim);
		ps_expand_to_i<<<dim/N_TREAD+1,N_TREAD>>>(gg_dev,g_right, l_g, l_f, dim);
		
		// then we need a truncated copy of f wrt x^{2^i}
		sfixn *ff_dev;
		cudaMalloc((void**)&ff_dev, dim*sizeof(sfixn));
		list_shrink_after_invfft<<<dim/N_TREAD+1,N_TREAD>>>(L_dev, ff_dev, length_per_poly, l_f, dim);
		/*
		sfixn *TT = (sfixn *)malloc(l_f*num_poly*sizeof(sfixn));
		cudaMemcpy(TT,g_right,l_f*num_poly*sizeof(sfixn),cudaMemcpyDeviceToHost);
		print_vector(l_f*num_poly,TT);
		*/
		// if the length of polynomial is smaller than 256 use plain/naive multiplication 
		if (i <= 8)
		{
			// We first compute g_{i-1}^2 mod x^{2^i}
			ps_list_poly_mul<<<num_poly,l_f>>>(g_left, l_f, g_right, l_f, g_sq, l_f, p, pinv);	
			// Compute g_right = f * g^2
			ps_list_poly_mul<<<num_poly,l_f>>>(g_sq, l_f, ff_dev, l_f, g_right, l_f, p, pinv);
			// compute g_left = 2g
			ps_add_2g_minus<<<dim/N_TREAD+1,N_TREAD>>>(g_right, g_left, l_f*num_poly, p, pinv);
			// copy the product to gg_dev
			cudaMemcpy(gg_dev, g_right, sizeof(sfixn)*l_f*num_poly,cudaMemcpyDeviceToDevice);
		}
		// else use FFT
		else
		{
			// first I found there is no point to expand g_right to get g_sq
			// because g_right is alread expanded
			// just do list_of_fft
			sfixn w = primitive_root(i+1,p);
			sfixn *f_ex, *gr_ex;
			dim = num_poly * l_f * 2;
			cudaMalloc((void**)&gr_ex,dim*sizeof(sfixn));
			ps_expand_to_i<<<dim/N_TREAD+1,N_TREAD>>>(gg_dev,gr_ex,l_g,l_f*2,dim);

			list_stockham_dev(gr_ex, num_poly, i+1,w,p);
			ps_pointwise_sq<<<dim/N_TREAD+1,N_TREAD>>>(gr_ex,dim,p,pinv);

			sfixn winv = inv_mod(w,p);
			sfixn ninv = inv_mod(l_f*2,p);
			/*
			list_stockham_dev(g_right, num_poly, i,winv,p);
			list_inv_mul_ker<<<dim/N_TREAD+1,N_TREAD>>>(g_right,ninv,num_poly,p,pinv,dim);
			*/
			
			// to compute f * g_right we have to expand them now
			cudaMalloc((void**)&f_ex,dim*sizeof(sfixn));
			ps_expand_to_i<<<dim/N_TREAD+1,N_TREAD>>>(ff_dev,f_ex,l_f,l_f*2,dim);
			// fft f 
			list_stockham_dev(f_ex,num_poly, i+1,w,p);
			// pointwise g 
			ps_pointwise_mul<<<dim/N_TREAD+1,N_TREAD>>>(gr_ex,f_ex, dim, p,pinv);
			// ifft g^2
			list_stockham_dev(gr_ex,num_poly,i+1,winv,p);
			list_inv_mul_ker<<<dim/N_TREAD+1,N_TREAD>>>(gr_ex,ninv,num_poly,p,pinv,dim);
			// shrink g
			dim = num_poly * l_f;
			list_shrink_after_invfft<<<dim/N_TREAD+1,N_TREAD>>>(gr_ex,g_right,l_f*2,l_f,dim);
			// basically the fft part is end
			// then same as the smaller part
			ps_add_2g_minus<<<dim/N_TREAD+1,N_TREAD>>>(g_right, g_left, l_f*num_poly, p, pinv);
			cudaMemcpy(gg_dev, g_right, sizeof(sfixn)*l_f*num_poly,cudaMemcpyDeviceToDevice);
			cudaFree(f_ex);
			cudaFree(gr_ex);
		}
		cudaFree(ff_dev);
		cudaFree(g_left);
		cudaFree(g_right);
	}
	//cudaMemcpy(I_dev, gg_dev, sizeof(sfixn)*l*num_poly, cudaMemcpyDeviceToDevice);
	dim = l * num_poly;
	list_shrink_after_invfft<<<dim/N_TREAD+1,N_TREAD>>>(gg_dev, I_dev, ln, l, dim);

	/*
	sfixn * TT;
	sfixn dd = l * num_poly;
	cudaMemcpy(TT,gg_dev,dd*sizeof(sfixn),cudaMemcpyDeviceToHost);
	print_poly(dd,TT);
	*/

	cudaFree(gg_dev);
}

////////////////////////////////////////////////////////////////////////////////////////////
//BEGIN:power_inversion_tst
////////////////////////////////////////////////////////////////////////////////////////////
void power_inversion_tst(sfixn p, sfixn k, sfixn m, sfixn l)
{
	sfixn ll = k;

	sfixn *F = (sfixn*)malloc(sizeof(sfixn)*ll*m);
	sfixn *Finv = (sfixn*)malloc(sizeof(sfixn)*l*m);

	for (sfixn i=0; i<m; i++)
	{
		for (sfixn j=0;j<ll;j++)
		{
			F[i*ll+j] = i+j+1;
		}
		F[i*ll+ll-1] = 1;
		F[i*ll] = 1;
	}

	sfixn *F_dev, *I_dev;
	cudaMalloc((void **)&F_dev, sizeof(sfixn)*ll*m);
	cudaMalloc((void **)&I_dev, sizeof(sfixn)*l*m);

	cudaMemcpy(F_dev, F, sizeof(sfixn)*ll*m,cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);

	list_power_series_inversion(F_dev, I_dev, ll*m, m, l,p);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);
	cudaEventDestroy(stop);
	
	fprintf(stdout,"TIME USED: %f\n",time);

	cudaMemcpy(Finv, I_dev, sizeof(sfixn)*l*m,cudaMemcpyDeviceToHost);

	cudaFree(F_dev);
	cudaFree(I_dev);

	sfixn *F_CPU = (sfixn *)malloc(sizeof(sfixn)*ll);
	sfixn *I_CPU = (sfixn *)malloc(sizeof(sfixn)*ll);
	for (sfixn i=0;i<m;i++)
	{
		printf("F[%d] := ",i);print_poly(ll-1, &(F[i*ll]),'x');printf(";\n");
		printf("G[%d] := ",i);print_poly(l-1, &(Finv[i*l]),'x');printf(";\n");
	}
	free(F);
	free(Finv);
}

////////////////////////////////////////////////////////////////////////////////////////////
//END:power_inversion_tst
////////////////////////////////////////////////////////////////////////////////////////////
/*
int main(int argc, char *argv[])
{
	sfixn p = atoi(argv[1]);
	sfixn k = atoi(argv[2]);
	sfixn m = atoi(argv[3]);
	sfixn l = atoi(argv[4]);
	power_inversion_tst(p,k,m,l);

}
*/

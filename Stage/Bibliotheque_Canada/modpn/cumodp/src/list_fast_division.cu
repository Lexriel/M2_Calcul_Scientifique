#include "../include/list_fast_division.h"
#include "../include/list_poly_rev.h"
/**
 * Pointwise minus of two polynimials
 * R = (R - G) mod p;
 *
 * R the former polynomial
 * G the latter
 * length (length of R/G, R and G should be in same length)
 * p the prime
 * */
__device__ __global__ void poly_minus(sfixn *R, sfixn *G, sfixn length, sfixn p)
{
	sfixn tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < length)
	{
		sfixn nn = neg_mod(G[tid],p);
		sfixn mm = R[tid];
		R[tid] = add_mod(mm,nn,p);
	}
}

/**
 * List of Fast evaluation based on power series inversion
 * About the detailed algorithm check MCA book
 *
 * A: the input polynomials (the "a" in the formular) and the output remainders R are also stored in A
 * length_poly_A: length of each polynomial in A
 * B: the input list of divisors (polynomials). They are assumed to be monic.
 * length_poly_B: length of each polynomial in B
 * num_poly: number of polynomials to be divided
 * p: the prime number
 * Uses 1D thread block, 1D grid
 * Thus 2^9 threads per block, 2^{16} blocks.
 * num_poly * length_poly_A must be at most 2^25
 * */
void list_fast_division(sfixn *A, sfixn length_poly_A, sfixn *B, sfixn length_poly_B, sfixn num_poly, sfixn p)
{
	double pinv = (double)1/p;
	// f := rev(b)
	sfixn dim = length_poly_B * num_poly;
	sfixn *F;
	cudaMalloc((void **)&F, length_poly_B * num_poly * sizeof(sfixn));
	poly_rev_dup<<<dim/N_TREAD+1, N_TREAD>>>(B, length_poly_B, F, dim);

	
	// g_inv := inverse of f mod x^{n-m+1}
	sfixn l = length_poly_A - length_poly_B + 1;
	sfixn *G_inv;
	cudaMalloc((void **)&G_inv, l * num_poly * sizeof(sfixn));
	list_power_series_inversion(F,G_inv,length_poly_B*num_poly,num_poly, l, p);
	cudaFree(F);

	// revA := rev(a)
	sfixn *revA;
	cudaMalloc((void**)&revA, length_poly_A * num_poly * sizeof(sfixn));
	dim = length_poly_A * num_poly;
	poly_rev_dup<<<dim/N_TREAD+1,N_TREAD>>>(A, length_poly_A,revA,dim);
	
	// shrink revA -> mulA
	sfixn *mulA;
	cudaMalloc((void**)&mulA, l * num_poly * sizeof(sfixn));
	dim = l * num_poly;
	list_shrink_after_invfft<<<dim/N_TREAD+1,N_TREAD>>>(revA, mulA, length_poly_A, l, dim);
	cudaFree(revA);
	
	// q_rev:= mulA * g_inv mod x^l
	sfixn *q_rev;
	cudaMalloc((void**)&q_rev, num_poly * l * sizeof(sfixn));
	if(l < (1L << 8))
	{
		ps_list_poly_mul<<<num_poly,l>>>(mulA, l, G_inv, l, q_rev, l, p, pinv);
	}
	else
	{
		sfixn *e_mulA,*e_G_inv;
		sfixn s = 2 * l -1;
		sfixn e = ceiling_log2(s);
		sfixn ln = (1L << e);
		dim = ln * num_poly;
		cudaMalloc((void**)&e_mulA,dim*sizeof(sfixn));
		cudaMalloc((void**)&e_G_inv,dim*sizeof(sfixn));
		
		ps_expand_to_i<<<dim/N_TREAD+1,N_TREAD>>>(mulA,e_mulA,l,ln,dim);
		ps_expand_to_i<<<dim/N_TREAD+1,N_TREAD>>>(G_inv,e_G_inv,l,ln,dim);

		sfixn w = primitive_root(e,p);
		
		list_stockham_dev(e_mulA,num_poly,e,w,p);
		list_stockham_dev(e_G_inv,num_poly,e,w,p);

		ps_pointwise_mul<<<dim/N_TREAD+1,N_TREAD>>>(e_mulA,e_G_inv,dim,p,pinv);

		sfixn winv = inv_mod(w,p);
		sfixn ninv = inv_mod(ln,p);
		list_stockham_dev(e_mulA,num_poly,e,winv,p);
		list_inv_mul_ker<<<dim/N_TREAD+1,N_TREAD>>>(e_mulA,ninv,num_poly,p,pinv,dim);

		dim = l * num_poly;
		list_shrink_after_invfft<<<dim/N_TREAD+1,N_TREAD>>>(e_mulA,q_rev,ln,l,dim);

		cudaFree(e_mulA);
		cudaFree(e_G_inv);
	}
	cudaFree(G_inv);	
	cudaFree(mulA);

	// q := rev(q_rev)
	sfixn *q;
	dim = l * num_poly;
	cudaMalloc((void**)&q, num_poly * l * sizeof(sfixn));
	poly_rev_dup<<<dim/N_TREAD+1,N_TREAD>>>(q_rev, l, q, dim);
	cudaFree(q_rev);
	
	// bq := B * q
	sfixn *eaq, *eab;
	cudaMalloc((void**)&eaq, num_poly * length_poly_A * sizeof(sfixn));
	cudaMalloc((void**)&eab, num_poly * length_poly_A * sizeof(sfixn));
	dim = num_poly * length_poly_A;
	ps_expand_to_i<<<dim/N_TREAD+1,N_TREAD>>>(q, eaq, l, length_poly_A, dim);
	ps_expand_to_i<<<dim/N_TREAD+1,N_TREAD>>>(B, eab, length_poly_B, length_poly_A,dim);

	sfixn *bq;
	cudaMalloc((void**)&bq, num_poly * length_poly_A * sizeof(sfixn));
	if(length_poly_A < (1L << 8))
	{
		ps_list_poly_mul<<<num_poly,length_poly_A>>>(eaq, length_poly_A, eab, length_poly_A, bq, length_poly_A, p, pinv);		
	}
	else
	{
		sfixn *e_eaq,*e_eab;
		sfixn s = 2 * length_poly_A -1;
		sfixn e = ceiling_log2(s);
		sfixn ln = (1L << e);
		dim = ln * num_poly;
		cudaMalloc((void**)&e_eaq,dim*sizeof(sfixn));
		cudaMalloc((void**)&e_eab,dim*sizeof(sfixn));
		
		ps_expand_to_i<<<dim/N_TREAD+1,N_TREAD>>>(eab,e_eab,length_poly_A,ln,dim);
		ps_expand_to_i<<<dim/N_TREAD+1,N_TREAD>>>(eaq,e_eaq,length_poly_A,ln,dim);

		sfixn w = primitive_root(e,p);
		
		list_stockham_dev(e_eab,num_poly,e,w,p);
		list_stockham_dev(e_eaq,num_poly,e,w,p);

		ps_pointwise_mul<<<dim/N_TREAD+1,N_TREAD>>>(e_eab,e_eaq,dim,p,pinv);

		sfixn winv = inv_mod(w,p);
		sfixn ninv = inv_mod(ln,p);
		list_stockham_dev(e_eab,num_poly,e,winv,p);
		list_inv_mul_ker<<<dim/N_TREAD+1,N_TREAD>>>(e_eab,ninv,num_poly,p,pinv,dim);

		dim = length_poly_A * num_poly;
		list_shrink_after_invfft<<<dim/N_TREAD+1,N_TREAD>>>(e_eab,bq,ln,length_poly_A,dim);

		cudaFree(e_eab);
		cudaFree(e_eaq);
	}

	// R := a - bq
	dim = length_poly_A * num_poly;
	poly_minus<<<dim/N_TREAD+1,N_TREAD>>>(A,bq,dim,p);
	
	cudaFree(bq);
	cudaFree(eaq);
	cudaFree(eab);
	cudaFree(q);
}
////////////////////////////////////////////////////////////////////
//BEGIN:list_fast_division_tst
////////////////////////////////////////////////////////////////////
void list_fast_division_tst(sfixn p,sfixn m, sfixn k)
{
	sfixn n = (1L << k)+1;
	sfixn nf = (1L << (k+1));
	sfixn *F = (sfixn *)malloc(sizeof(sfixn)*m*nf);
	sfixn *M = (sfixn *)malloc(sizeof(sfixn)*m*n);

	for (sfixn i = 0; i < m; i++)
	{
		for (sfixn j = 0; j < nf; j++)
		{
			F[i*nf+j] = i*nf+j+1;
		}
		for (sfixn j = 0; j < n; j++)
		{
			M[i*n+j] = 1;
		}
		M[i*n] = 1;
		M[i*n+n-1]=1;
	}

	//sfixn s = nf - n + 1; 
	sfixn *R = (sfixn *)malloc(sizeof(sfixn)*m*nf);

	sfixn *F_dev,*M_dev,*R_dev;
	cudaMalloc((void**)&F_dev, sizeof(sfixn)*m*nf);
	cudaMalloc((void**)&M_dev, sizeof(sfixn)*m*n);
	cudaMalloc((void**)&R_dev, sizeof(sfixn)*m*nf);

	cudaMemcpy(F_dev, F, sizeof(sfixn)*m*nf,cudaMemcpyHostToDevice);
	cudaMemcpy(M_dev, M, sizeof(sfixn)*m*n, cudaMemcpyHostToDevice);
	
	list_fast_division(F_dev, nf, M_dev, n, m, p);
	
	cudaMemcpy(R, F_dev, sizeof(sfixn)*m*nf,cudaMemcpyDeviceToHost);

	for (sfixn i = 0; i < m; i++)
	{
		// TODO: printout the results
		//printf("F[%d] := ",i); print_poly(nf-1,F,'x');printf(";\n");
		//printf("M[%d] := ",i); print_poly(n-1, M,'x');printf(";\n");
		printf("Rm[%d] := ",i); print_poly(nf-1, &(R[i*nf]),'x');printf(";\n"); 
		printf("r[%d] := Rem(",i);print_poly(nf-1, &(F[i*nf]), 'x'); 
		printf(",");print_poly(n-1,&(M[i*n]),'x');printf(",x) mod %d;\n",p);
	}

}

////////////////////////////////////////////////////////////////////
//END:list_fast_division_tst
////////////////////////////////////////////////////////////////////

/*
int main(int argc, char * argv[])
{
	list_fast_division_tst(atoi(argv[1]),atoi(argv[2]),atoi(argv[3]));
	return 0;
}
*/

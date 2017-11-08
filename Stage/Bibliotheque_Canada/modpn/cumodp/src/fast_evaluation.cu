#include "../include/fast_evaluation.h"


/**
 * [EXPIRED]
 * */
__device__ __global__ void double_and_copy(sfixn *A, sfixn *E, sfixn small, sfixn bound)
{
	sfixn tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < bound)
	{
		sfixn big = 2 * small;
		sfixn pos = tid / big; 
		sfixn rem = tid % big;
		if(pos * small + rem < bound)
		{
			if(rem < small)
				E[tid] = A[pos * small + rem];
			else 
				E[tid] = A[pos * small + rem - small];
		}
	}
}
// [EXPIRED]
void double_and_copy_tst(sfixn m, sfixn k)
{
	sfixn *A = (sfixn *)malloc(sizeof(sfixn)*k*m);
	sfixn *B = (sfixn *)malloc(sizeof(sfixn)*2*k*m);

	for(sfixn i=0; i<m;i++)
	{
		for(sfixn j=0; j<k;j++)
		{
			A[i*k+j] = i*k+ j+1;
		}
	}
	sfixn *A_dev,*B_dev;
	cudaMalloc((void**)&A_dev, sizeof(sfixn)*k*m);
	cudaMalloc((void**)&B_dev, sizeof(sfixn)*2*k*m);
	
	cudaMemcpy(A_dev,A,sizeof(sfixn)*k*m, cudaMemcpyHostToDevice);
	sfixn dim = 2 * k * m;
	double_and_copy<<<dim/512+1,512>>>(A_dev, B_dev, k, dim);

	cudaMemcpy(B,B_dev,sizeof(sfixn)*2*k*m,cudaMemcpyDeviceToHost);
	print_vector(dim/2, A);
	print_vector(dim, B);
	cudaFree(A_dev);
	cudaFree(B_dev);

}
/**
 * This kernel is to copy the adjacent polynomial to fill up the gap 
 * for the convenience of list_fast_division
 *
 * A is the remainder after last pass of division
 * s is the length of each remainder
 * bound is the bound of A
 * */
__device__ __global__ void trunk_copy(sfixn *A, sfixn s, sfixn bound)
{
	sfixn tid = threadIdx.x + blockIdx.x * blockDim.x;

	if(tid < bound)
	{
		sfixn rem = tid / s;
		if(rem %2 == 1)
			A[tid] = A[tid-s];
	}
}


/**
 * Driver of fast_evaluation
 * It applies fast division at levels above 8
 * then turn to plain division at lower levels
 *
 * M_dev is the 1-D array encoding the subproduct tree
 * k  is the dimension of input size of X.
 * MORE PRECISELY, k is the log of the total number
 * of coefficients at the layer of the degree 1 polynomials
 * THUS the number of points (in the sense of MCA) is 2^{k-1} 
 * F_dev is the function F
 * length_F is the length 
 * p the prime
 * */
void fast_evaluation_dev(sfixn *M_dev, sfixn k, sfixn *F_dev, sfixn length_F, sfixn p)
{
	//sfixn n = (1L << k);
	for (sfixn i = (k - 1); i >= 1; i--)
	{
		
		sfixn num_poly = get_layer_size(k,i)/get_polylength_on_layer(i);
		sfixn length_poly = get_polylength_on_layer(i);
		sfixn start_offset = get_layer_offset(k,i);
			
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		// Then apply list of division
		sfixn nf = length_F / num_poly;
		sfixn thresh = 7;
	
		if(i>(thresh))
		{
			// First extract the layer L_dev from M	
			sfixn *L_dev;
			cudaMalloc((void**)&L_dev,num_poly * length_poly * sizeof(sfixn));
			cudaMemcpy(L_dev, &(M_dev[start_offset]), num_poly * length_poly * sizeof(sfixn), cudaMemcpyDeviceToDevice);
			
			//apply fft-based fast division
			sfixn s = nf - length_poly + 1;
			
			//printf("length_F = %d, n = %d, nf = %d, length_poly = %d, num_poly = %d\n",length_F, n, nf, length_poly, num_poly);
			list_fast_division(F_dev, nf, L_dev, length_poly, num_poly, p);
		
			trunk_copy<<<length_F/N_TREAD+1,N_TREAD>>>(F_dev, s, length_F);
			/*
			sfixn * TT = (sfixn *)malloc(sizeof(sfixn)*length_F);
			cudaMemcpy(TT,F_dev,sizeof(sfixn)*length_F,cudaMemcpyDeviceToHost);
			print_vector(length_F,TT);
			*/
			/*
			sfixn *FF_dev;
			cudaMalloc((void**)&FF_dev, length_F * sizeof(sfixn));
			double_and_copy<<<n/N_TREAD+1,N_TREAD>>>(F_dev, FF_dev, s, length_F);
			cudaMemcpy(F_dev, FF_dev, length_F * sizeof(sfixn), cudaMemcpyDeviceToDevice);
			cudaFree(FF_dev);
			*/
			if (i == (thresh + 1))
			{
				sfixn *shr;
				cudaMalloc((void**)&shr,length_F*sizeof(sfixn));
				sfixn dd = length_F;
				list_shrink_after_invfft<<<dd/N_TREAD+1,N_TREAD>>>(F_dev,shr,nf,s,dd);
				sfixn *KK = (sfixn*)malloc(length_F*sizeof(sfixn));
				cudaMemcpy(KK,F_dev,length_F*sizeof(sfixn),cudaMemcpyDeviceToHost);
				printf("KK: ");print_vector(length_F,KK);
				//free(KK);
				cudaMemcpy(F_dev,shr,length_F*sizeof(sfixn),cudaMemcpyDeviceToDevice);
				cudaFree(shr);
			}
			/*
			free(TT);
			*/
			cudaFree(L_dev);
		}
		else
		{
			//apply plain division
			sfixn threadsForAdiv = (sfixn)ceil((double)length_poly/(double)W);
			sfixn divInThreadBlock = (sfixn)floor((double)T/(double)threadsForAdiv);
			sfixn blockNo = (sfixn)ceil((double)num_poly/(double)divInThreadBlock);
			
			list_divCUDA<<<blockNo,T>>>(M_dev, F_dev, start_offset, length_poly, threadsForAdiv, divInThreadBlock, num_poly, p);
		}
		
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cudaEventDestroy(stop);
		printf("LAYER %d: %f\n",i,time);
		
	}
}

/**
 * Host function to drive the fast_evaluaiton
 *
 * X the input point X
 * k the dimension of X, i.e 2^k of coeeficients, then 2^{k-1} points to be evaluated
 * F the polynomial F
 * length_F number of coefficient of F
 * p the prime
 * */

void fast_evaluation_host(sfixn *X, sfixn k, sfixn *F, sfixn length_F, sfixn p)
{
	sfixn n = (1L << k);
	sfixn *M_dev;
	
	sfixn tree_size = get_subtree_size(k);
	
	cudaMalloc((void**)&M_dev, sizeof(sfixn)*tree_size);

	sfixn *F_dev;
	cudaMalloc((void**)&F_dev, sizeof(sfixn)*n);

	cudaMemcpy(M_dev, X, n*sizeof(sfixn), cudaMemcpyHostToDevice);
	
	cudaMemcpy(F_dev, F, (length_F/2)*sizeof(sfixn), cudaMemcpyHostToDevice);
	cudaMemcpy(&(F_dev[length_F/2]), F, (length_F/2)*sizeof(sfixn), cudaMemcpyHostToDevice);

	subproduct_tree_dev(M_dev, k, p, (double)1/p);
	//////////////////////////////////////////////////////////////////////
	//Performance Measurement
	//////////////////////////////////////////////////////////////////////
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//////////////////////////////////////////////////////////////
	fast_evaluation_dev(M_dev, k, F_dev, length_F, p);
	//////////////////////////////////////////////////////////////
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(stop);
	//////////////////////////////////////////////////////////////////////

	cudaMemcpy(F, F_dev, length_F *sizeof(sfixn), cudaMemcpyDeviceToHost);

	//print subproduct tree
	sfixn *FF = (sfixn *)malloc(2*sizeof(sfixn));
	printf("Input := Array([");
	for (sfixn i = 1; i <= n; i++)
	{
		if ( (i-1) % 2 == 1)
		{
			if (i > 1)
			{
				FF[0] = X[i-2];
				FF[1] = X[i-1];
				printf("(");print_poly(1,FF,'x');
				if(i<n)printf("),");
				if(i==n)printf(")])");
			}
		}
	}
	printf(";\n");	
	free(FF);

	sfixn *M = (sfixn *)malloc(sizeof(sfixn)*tree_size);
	cudaMemcpy(M, M_dev, tree_size *sizeof(sfixn), cudaMemcpyDeviceToHost);

	sfixn maple_sn = 2;
	for (sfixn i = k-1; i > 1; i--)
	{
		sfixn num_polys = get_layer_size(k, i)/get_polylength_on_layer(i);
		sfixn offset = get_layer_offset(k,i);
		sfixn *T = (sfixn *)malloc(get_polylength_on_layer(i)*sizeof(sfixn));
		for (sfixn j = 0; j < num_polys; j++)
		{
		//	for ( sfixn ii = offset + j * get_polylength_on_layer(i); ii < offset+j * get_polylength_on_layer(i) + get_polylength_on_layer(i); ii++)
		//		T[ii-offset] = M[ii];
			sfixn polylen = get_polylength_on_layer(i);
			sfixn polyoff = offset +  j * polylen;
			for (sfixn pt = 0; pt < polylen; pt++){
				T[pt] = M[polyoff + pt];
			}
			printf("Subtree[%d] :=",maple_sn); print_poly(get_polylength_on_layer(i)-1, T, 'x'); printf(";\n");
			maple_sn++;
		}
		free(T);
	}

	//print_vector(length_F/2,F);
	for (sfixn i = 0; i < length_F/2; i++)
	{
		printf("cF[%d] := %d;\n",i+1,F[i]); 
		printf("mF[%d] := Eval(F, x=%d) mod %d;\n", i+1,neg_mod(X[i*2],p), p);
	}
	printf("\n");

	cudaFree(F_dev);
	cudaFree(M_dev);

	fprintf(stdout, "TIME USED: %f\n",time);
}

/**
 * Host function to drive the fast_evaluaiton for benchmarking
 *
 * X the input point X
 * k the dimension of X, i.e 2^k of coeeficients, then 2^{k-1} points to be evaluated
 * F the polynomial F
 * length_F number of coefficient of F
 * p the prime
 * */

void fast_evaluation_host_bchmk(sfixn *X, sfixn k, sfixn *F, sfixn length_F, sfixn p)
{
	sfixn n = (1L << k);
	sfixn *M_dev;
	
	sfixn tree_size = get_subtree_size(k);
	
	cudaMalloc((void**)&M_dev, sizeof(sfixn)*tree_size);

	sfixn *F_dev;
	cudaMalloc((void**)&F_dev, sizeof(sfixn)*n);

	cudaMemcpy(M_dev, X, n*sizeof(sfixn), cudaMemcpyHostToDevice);
	
	cudaMemcpy(F_dev, F, (length_F/2)*sizeof(sfixn), cudaMemcpyHostToDevice);
	cudaMemcpy(&(F_dev[length_F/2]), F, (length_F/2)*sizeof(sfixn), cudaMemcpyHostToDevice);

	subproduct_tree_dev(M_dev, k, p, (double)1/p);
	//////////////////////////////////////////////////////////////////////
	//Performance Measurement
	//////////////////////////////////////////////////////////////////////
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//////////////////////////////////////////////////////////////
	fast_evaluation_dev(M_dev, k, F_dev, length_F, p);
	//////////////////////////////////////////////////////////////
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(stop);
	//////////////////////////////////////////////////////////////////////
	cudaFree(F_dev);
	cudaFree(M_dev);

	fprintf(stdout, "TIME USED: %f\n",time);
}

///////////////////////////////////////////////////////////////////
//BEGIN:fast_evaluation_tst
///////////////////////////////////////////////////////////////////
void fast_evaluation_tst(sfixn k)
{	
	sfixn p = 469762049;
	sfixn n = (1L << k);
	sfixn *X = (sfixn *)malloc(sizeof(sfixn)*n);
	
	sfixn *F = (sfixn *)malloc(sizeof(sfixn)*n);
	for (sfixn i = 0; i < n/2; i++)
		F[i] = (n/2-i)%p;

	F[n/2-1]=1;
	F[0]=1;
	printf("F:= ");print_poly(n/2-1,F,'x');printf(";\n");

	//printf("At:\n");
	for (sfixn i = 0; i < n; i++)
	{
		if(i%2 == 1)
			X[i] = 1;
		else
		{
			X[i] = neg_mod((i/2+1),p);
			//printf("F(%d),",neg_mod(X[i],p)); 
		}
	}
	//printf(";\n");
	fast_evaluation_host(X, k, F, n, p);

}
void fast_evaluation_bench(sfixn k)
{	
	sfixn p = 469762049;
	sfixn n = (1L << k);
	sfixn *X = (sfixn *)malloc(sizeof(sfixn)*n);
	
	sfixn *F = (sfixn *)malloc(sizeof(sfixn)*n);
	for (sfixn i = 0; i < n/2; i++)
		F[i] = (n/2-i)%p;

	F[n/2-1]=1;
	F[0]=1;

	for (sfixn i = 0; i < n; i++)
	{
		if(i%2 == 1)
			X[i] = 1;
		else
		{
			X[i] = neg_mod((i/2+1),p);
		}
	}
	fast_evaluation_host_bchmk(X, k, F, n, p);

}
void fast_evaluation_tst()
{	
	// Read file
	sfixn p;
	sfixn k;

	FILE * inf;
	FILE * inx;
	FILE * outf;

	inf = fopen ("inputf.txt", "r");
	inx = fopen ("inputx.txt", "r");
	outf = fopen ("output.txt","w");

	fscanf(inf, "%d", &p);
	if (!is_prime(p))
	{
		fprintf(stderr,"P is not a prime\n");
		return;
	}
	fscanf(inx, "%d", &k);
	if (k >= 24 || k < 2)
	{
		fprintf(stderr,"Invalid K\n");
		return;
	}

	sfixn n = (1L << k);

	// Here is the problem: how to know which is for F which is X(the points)
	sfixn *X, *F;
	X = (sfixn *)malloc(sizeof(sfixn)*n);
	sfixn buf;
	for(sfixn i = 0; i < n/2; i++)
	{
		fscanf(inx,"%d", &buf);
		X[i*2] = neg_mod(buf,p);
		X[i*2+1] = 1;
	}

	F = (sfixn *)malloc(sizeof(sfixn)*n);
	for (sfixn i = 0; i < n; i++)
	{
		F[i] = 0;
		fscanf(inf,"%d", &F[i]);
	}

	printf("F:= ");print_poly(n/2-1,F,'x');printf(";\n");
	// compute
	fast_evaluation_host(X, k, F, n, p);

	fclose(inx);
	fclose(inf);
	// Write file
	
	for (sfixn i = 0; i < n/2; i++)
	{
		fprintf(outf, "%d ", F[i]);
	}
	
	fclose(outf);
	
}
///////////////////////////////////////////////////////////////////
//END:fast_evaluation_tst
///////////////////////////////////////////////////////////////////
/*
int main(int argc, char * argv[])
{
	//double_and_copy_tst(atoi(argv[1]),atoi(argv[2]));
	fast_evaluation_tst(atoi(argv[1]));
	//fast_evaluation_tst();
	return 0;
}
*/
	

#include "../include/subproduct_tree.h"

/**
 * The main driver function to construct subproduct tree with efficient FFT-based multiplication
 * This is much faster than the previous one
 *
 * M_dev: the array encoding the subproduct tree, the first 2^k coefficients
 * is the copy of an input array X (which was read from CPU)
 * k: the dimension of input X (power of 2). MORE PRECISELY
 * the total number of points (in the MCA
 * p: the prime
 * pinv: the inversion of p
 * */
void subproduct_tree_dev_eff(sfixn *M_dev, sfixn k, sfixn p, double pinv)
{
	sfixn treesize = get_subtree_size(k);
	for (sfixn i = 1; i < (k-1); i++)
	{	
		/*Measure Step*/
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0);
		/*Measure Step*/

		// TODO: find a way to apply effective multiplication
		// How many polynomial multiplications should take place in this layer
		sfixn num_muls = (get_layer_size(k, i) / get_polylength_on_layer(i))/2;
		// length of each polynomials
		sfixn length_poly = get_polylength_on_layer(i);
		// resulting polynomial length
		sfixn length_result = get_polylength_on_layer(i+1);
		// resulting starting offset for the next layer
		sfixn result_offset = get_layer_offset(k, i+1);
		// starting point of the polynomials
		sfixn start_offset = get_layer_offset(k, i);

		// for FFT
		sfixn num_poly = num_muls * 2;
	
		// if the length_poly is bigger than 512, use FFT instead
		if (length_poly <= (1L << 8))
		{
			if( i <= 2 )
			{
				if (num_muls > N_TREAD)
					list_poly_mul_ker<<<num_muls/N_TREAD,N_TREAD>>>(M_dev, length_poly, length_result, start_offset, result_offset, p, pinv);
				else if (num_muls > 8)
					list_poly_mul_ker<<<num_muls/8,8>>>(M_dev, length_poly, length_result, start_offset, result_offset, p, pinv);
				else 
					list_poly_mul_ker<<<num_muls,1>>>(M_dev, length_poly, length_result, start_offset, result_offset, p, pinv);
			}
			else
			{
				//list_poly_mul_ker_higher<<<num_muls,length_result>>>(M_dev, length_poly, length_result, start_offset, result_offset, p, pinv);
				/********************************************************
				* Number of threads responsible for one multiplication  *
				*********************************************************/	
				int threadsForAmul = 2*length_poly;

				/************************************************
				* Number of multiplications in one thread block *
				*************************************************/
				int mulInThreadBlock = (int)floor((double)Tmul/(double)threadsForAmul);

				/****************************
				* Number of blocks required *
				****************************/
				int blockNo = (int)ceil( ((double)num_poly/(double) mulInThreadBlock)*0.5  );	

				printf("st_off = %d, length_poly = %d, num_poly = %d, tFA = %d, mITB = %d\n",start_offset, length_poly, num_poly, threadsForAmul, mulInThreadBlock);
				
				cudaThreadSynchronize();
				
				listPlainMulGpu<<<blockNo, Tmul>>>(M_dev, start_offset, length_poly, num_poly, threadsForAmul, mulInThreadBlock, p);
	
				cudaThreadSynchronize();
			}
		}
		else // use the FFT
		{
				sfixn *O_dev;
				cudaMalloc((void **)&O_dev, length_result*num_muls*sizeof(sfixn));
				//list_fft_poly_mul_dev(M_dev, length_poly, O_dev, length_result, num_poly, start_offset, p, pinv);
				list_fft_poly_mul_eff(M_dev, length_poly, O_dev, length_result, num_poly, start_offset, p, pinv);
				cudaMemcpy(&(M_dev[result_offset]), O_dev, length_result*num_muls*sizeof(sfixn), cudaMemcpyDeviceToDevice);
				cudaFree(O_dev);
		}
		/*Measure Step*/
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cudaEventDestroy(stop);
		cudaEventDestroy(start);
		printf("LAYER %d : %f\n",i,time);
		/*Measure Step*/
	}
}
// [EXPIRED] The one using whole FFT
void subproduct_tree_dev(sfixn *M_dev, sfixn k, sfixn p, double pinv)
{
	sfixn treesize = get_subtree_size(k);
	for (sfixn i = 1; i < (k-1); i++)
	{	
		/*Measure Step*/
		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, 0);
		/*Measure Step*/

		// TODO: find a way to apply effective multiplication
		// How many polynomial multiplications should take place in this layer
		sfixn num_muls = (get_layer_size(k, i) / get_polylength_on_layer(i))/2;
		// length of each polynomials
		sfixn length_poly = get_polylength_on_layer(i);
		// resulting polynomial length
		sfixn length_result = get_polylength_on_layer(i+1);
		// resulting starting offset for the next layer
		sfixn result_offset = get_layer_offset(k, i+1);
		// starting point of the polynomials
		sfixn start_offset = get_layer_offset(k, i);

		// for FFT
		sfixn num_poly = num_muls * 2;
	
		// if the length_poly is bigger than 512, use FFT instead
		if (length_poly <= (1L << 7))
		{
			if( i <= -1 )
			{
				if (num_muls > N_TREAD)
					list_poly_mul_ker<<<num_muls/N_TREAD,N_TREAD>>>(M_dev, length_poly, length_result, start_offset, result_offset, p, pinv);
				else if (num_muls > 8)
					list_poly_mul_ker<<<num_muls/8,8>>>(M_dev, length_poly, length_result, start_offset, result_offset, p, pinv);
				else 
					list_poly_mul_ker<<<num_muls,1>>>(M_dev, length_poly, length_result, start_offset, result_offset, p, pinv);
			}
			else
			{
				//list_poly_mul_ker_higher<<<num_muls,length_result>>>(M_dev, length_poly, length_result, start_offset, result_offset, p, pinv);
				/********************************************************
				* Number of threads responsible for one multiplication  *
				*********************************************************/	
				int threadsForAmul = 2*length_poly;

				/************************************************
				* Number of multiplications in one thread block *
				*************************************************/
				int mulInThreadBlock = (int)floor((double)Tmul/(double)threadsForAmul);

				/****************************
				* Number of blocks required *
				****************************/
				int blockNo = (int)ceil( ((double)num_poly/(double) mulInThreadBlock)*0.5  );	

				printf("st_off = %d, length_poly = %d, num_poly = %d, tFA = %d, mITB = %d\n",start_offset, length_poly, num_poly, threadsForAmul, mulInThreadBlock);
				
				cudaThreadSynchronize();
				
				listPlainMulGpu<<<blockNo, Tmul>>>(M_dev, start_offset, length_poly, num_poly, threadsForAmul, mulInThreadBlock, p);
	
				cudaThreadSynchronize();
			}
		}
		else // use the FFT
		{
			if( i <= 26 )
			{
				sfixn *O_dev;
				cudaMalloc((void **)&O_dev, length_result*num_muls*sizeof(sfixn));
				cudaThreadSynchronize();
				list_fft_poly_mul_dev(M_dev, length_poly, O_dev, length_result, num_poly, start_offset, p, pinv);
				cudaThreadSynchronize();
				cudaMemcpy(&(M_dev[result_offset]), O_dev, length_result*num_muls*sizeof(sfixn), cudaMemcpyDeviceToDevice);
				cudaThreadSynchronize();
				cudaFree(O_dev);
				cudaThreadSynchronize();
			}
			else{
			
				// the polynomials for computation
				sfixn *A_dev, *B_dev, *C_dev;
	
				sfixn s = length_poly+length_poly-1;
				sfixn e = ceiling_log2(s);
				sfixn ln = (1L << e);
				sfixn mem_size = (sizeof(sfixn) << e );
			
				cudaMalloc((void **)&A_dev, mem_size);
				cudaMalloc((void **)&B_dev, mem_size);
				cudaMalloc((void **)&C_dev, length_result*sizeof(sfixn));
			
				for (sfixn j = 0; j < num_muls; j++)
				{
					cudaMemcpy(A_dev, &(M_dev[start_offset+j*2*length_poly]), length_poly*sizeof(sfixn), cudaMemcpyDeviceToDevice);
					cudaMemcpy(B_dev, &(M_dev[start_offset+j*2*length_poly+length_poly]), length_poly*sizeof(sfixn), cudaMemcpyDeviceToDevice);
				
					expand_to_fft_dev(ln, length_poly, A_dev);
					expand_to_fft_dev(ln, length_poly, B_dev);
					stockham_poly_mul_dev(ln, e, A_dev, B_dev, p);

					cudaMemcpy(C_dev, A_dev, length_result*sizeof(sfixn), cudaMemcpyDeviceToDevice);

					cudaMemcpy(&(M_dev[result_offset+j*length_result]), C_dev, length_result*sizeof(sfixn), cudaMemcpyDeviceToDevice);
				}
				cudaFree(A_dev);
				cudaFree(B_dev);
				cudaFree(C_dev);
			}	
		}
		/*Measure Step*/
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cudaEventDestroy(stop);
		printf("LAYER %d : %f\n",i,time);
		/*Measure Step*/
	}
}


// This is to test the subproduct tree construction
void subproduct_tree_host(sfixn *X, sfixn *M, sfixn k, sfixn p)
{
	sfixn n = (1L << k);
	/*
	for (sfixn i=0; i<n; i++)
		M[i] = X[i];*/
	double pinv = 1 / (double)p;

	sfixn treesize = get_subtree_size(k);
	sfixn *M_dev;
	cudaMalloc((void **)&M_dev, treesize*sizeof(sfixn));

	////////////////////////////////////////////////////////////////////////
	// Building the sub product tree
	// The first layer, we directly copy the array
	cudaMemcpy(M_dev, X, n*sizeof(sfixn), cudaMemcpyHostToDevice);

	////////////////////////////////////////////////////////////////////////
	// This part is for performance analysis
	////////////////////////////////////////////////////////////////////////
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	subproduct_tree_dev(M_dev,k,p,pinv);

	////////////////////////////////////////////////////////////////////////
	// Performance Analysis
	//
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(stop);

	////////////////////////////////////////////////////////////////////////
	cudaMemcpy(M, M_dev, treesize*sizeof(sfixn), cudaMemcpyDeviceToHost);

	cudaFree(M_dev);
}

// This is to benchmark the subproduct tree construction
void subproduct_tree_host_bchmk(sfixn *X, sfixn *M, sfixn k, sfixn p)
{
	sfixn n = (1L << k);
	/*
	for (sfixn i=0; i<n; i++)
		M[i] = X[i];*/
	double pinv = 1 / (double)p;

	sfixn treesize = get_subtree_size(k);
	sfixn *M_dev;
	cudaMalloc((void **)&M_dev, treesize*sizeof(sfixn));

	////////////////////////////////////////////////////////////////////////
	// Building the sub product tree
	// The first layer, we directly copy the array
	cudaMemcpy(M_dev, X, n*sizeof(sfixn), cudaMemcpyHostToDevice);

	////////////////////////////////////////////////////////////////////////
	// This part is for performance analysis
	////////////////////////////////////////////////////////////////////////
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	subproduct_tree_dev_eff(M_dev,k,p,pinv);

	////////////////////////////////////////////////////////////////////////
	// Performance Analysis
	//
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(stop);

	printf("TIME USED: %f\n", time);
	printf("Memory: %d\n", get_subtree_size(k)*sizeof(sfixn));
	////////////////////////////////////////////////////////////////////////

	cudaFree(M_dev);
}

////////////////////////////////////////////////////////////////////////////
//BEGIN:subproduct_tree_tst
////////////////////////////////////////////////////////////////////////////
void subproduct_tree_tst(sfixn p, sfixn k)
{
	//sfixn k = 3;
	//sfixn p = 257;
	sfixn n = (1L << k);

	sfixn *X = (sfixn *)malloc(n*sizeof(sfixn));

	sfixn sizeofM = get_subtree_size(k);
	sfixn *M = (sfixn *)malloc(sizeofM*sizeof(sfixn));

	sfixn *FF = (sfixn *)malloc(2*sizeof(sfixn));
	printf("Input := Array([");
	for (sfixn i=1; i<=n; i++){
		if( (i-1) % 2 == 0){
			X[i-1] = neg_mod(i,p);
		}
		else{
			X[i-1] = 1;
			if(i>1){
				FF[0] = X[i-2];
				FF[1] = X[i-1];
				printf("("); print_poly(1, FF, 'x'); 
				if(i<n)printf("),");
				if(i==n)printf(")])");
			}
		}
	}
	printf(";\n");

	free(FF);

	subproduct_tree_host(X, M, k, p);

	printf("Output :\n");
	sfixn offset_top = get_layer_offset(k, k-1);
	sfixn size_top = get_layer_size(k, k-1);
	sfixn size_top_poly = size_top/2;
	
	sfixn *F = (sfixn *)malloc(size_top_poly*sizeof(sfixn));
	sfixn *G = (sfixn *)malloc(size_top_poly*sizeof(sfixn));
	for(sfixn i = offset_top; i <  (offset_top + (size_top_poly)); i++)
	{
		F[i-offset_top] = M[i];
		G[i-offset_top] = M[i+size_top_poly];
	}
	printf("F := "); print_poly(size_top_poly-1, F, 'x'); printf(";\n");
	printf("G := "); print_poly(size_top_poly-1, G, 'x'); printf(";\n");
	free(F);
	free(G);

	/* This part is for maple to test the subproduct tree with input size
	 * k no more than 15
	 */
	
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
			printf("Test[%d] :=",maple_sn); print_poly(get_polylength_on_layer(i)-1, T, 'x'); printf(";\n");
			maple_sn++;
		}
		free(T);
	}
	
	/*
	for(sfixn i = 0; i < n; i++) printf("%d ",X[i]); printf("\n");
	for(sfixn i = 0; i < get_subtree_size(k); i++) printf("%d ",M[i]); printf("\n");
	*/

}

void subproduct_tree_tst()
{	
	// Read file
	sfixn p = 469762049;
	sfixn k;

	FILE * inx;
	inx = fopen ("inputx.txt", "r");
	
	fscanf(inx, "%d", &k);

	sfixn n = (1L << k);

	// Here is the problem: how to know which is for F which is X(the points)
	sfixn *X; 
	X = (sfixn *)malloc(sizeof(sfixn)*n);
	sfixn buf;
	for(sfixn i = 0; i < n/2; i++)
	{
		fscanf(inx,"%d", &buf);
		X[i*2] = neg_mod(buf,p);
		X[i*2+1] = 1;
	}

	printf("Input := Array([");
	for (sfixn i=1; i<=n; i++){
		if( (i-1) % 2 == 0){
		}
		else{
			if(i>1){
				sfixn *F = (sfixn *)malloc(2*sizeof(sfixn));
				F[0] = X[i-2];
				F[1] = X[i-1];
				printf("("); print_poly(1, F, 'x'); 
				if(i<n)printf("),");
				if(i==n)printf(")])");
				free(F);
			}
		}
	}
	printf(";\n");

	sfixn sizeofM = get_subtree_size(k);
	sfixn *M = (sfixn *)malloc(sizeofM*sizeof(sfixn));
	// compute
	subproduct_tree_host(X, M, k, p);

	fclose(inx);
	
	sfixn offset_top = get_layer_offset(k, k-1);
	sfixn size_top = get_layer_size(k, k-1);
	

	/* This part is for maple to test the subproduct tree with input size
	 * k no more than 15
	 */
	
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
			printf("Test[%d] :=",maple_sn); print_poly(get_polylength_on_layer(i)-1, T, 'x'); printf(";\n");
			maple_sn++;
		}
		free(T);
	}
}

void subproduct_tree_bench(sfixn p, sfixn k)
{
	//sfixn k = 3;
	//sfixn p = 257;
	sfixn n = (1L << k);

	sfixn *X = (sfixn *)malloc(n*sizeof(sfixn));

	sfixn sizeofM = get_subtree_size(k);
	sfixn *M = (sfixn *)malloc(sizeofM*sizeof(sfixn));

	for (sfixn i=1; i<=n; i++){
		if( (i-1) % 2 == 0){
			X[i-1] = neg_mod(i,p);
		}
		else{
			X[i-1] = 1;		
		}
	}

	subproduct_tree_host_bchmk(X, M, k, p);

}
////////////////////////////////////////////////////////////////////////////
//END:subproduct_tree_tst
////////////////////////////////////////////////////////////////////////////
//TODO: main func to be removed
/*
int main(int argc, char *argv[])
{
	sfixn k;
	if(argc < 2) k = 16;
	else k = atoi(argv[1]);
	subproduct_tree_tst(469762049,k);
	return 0;
}
*/



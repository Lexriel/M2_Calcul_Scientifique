#ifndef _SUBPRODUCT_TREE_H
#define _SUBPRODUCT_TREE_H
/**
 * this part is for building the subproduct tree
 *
 * @author: Jiajian Yang
 * */
#include "../include/defines.h"
#include "../include/rdr_poly.h"
#include "../include/inlines.h"
#include "../include/printing.h"
#include "../include/cudautils.h"
#include "../include/fft_aux.h"
#include "../include/naive_poly_mul.h"
#include "../include/stockham.h"
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#include "../include/list_naive_poly_mul.h"
#include "../include/list_pointwise_mul.h"
#include "../include/list_inv_stockham.h"
#include "../include/list_plain_mul.h"

#include "../include/list_fft_poly_mul.h"

void subproduct_tree_host(sfixn *X, sfixn *M, sfixn k, sfixn p);

__host__ __device__ __inline__ sfixn get_subtree_size(sfixn k)
{
	sfixn sum = 0;
	for(sfixn i = 1; i <= k-1; ++i)
		sum += ( (1L <<(i-1)) +1 ) * ( 1L << (k-i)) ;
	return sum;
}


__host__ __device__ __inline__ sfixn get_layer_size(sfixn k, sfixn i)
{
	sfixn deg = (1L << (i-1));
	return (deg + 1)*(1L<<(k-i));
}

__host__ __device__ __inline__ sfixn get_polylength_on_layer(sfixn i)
{
	return ((1L << (i-1)) +1);
}

__host__ __device__ __inline__ sfixn get_subtree_depth(sfixn k)
{
	return k-1;
}

__host__ __device__ __inline__ sfixn get_layer_offset(sfixn k, sfixn i)
{
	sfixn sum = 0;
	for (sfixn l = 1; l < i; l++)
		sum += get_layer_size(k,l);
	return sum;
}

void subproduct_tree_host_bchmk(sfixn *X, sfixn *M, sfixn k, sfixn p);
void subproduct_tree_bench(sfixn p, sfixn k);

void subproduct_tree_dev(sfixn *M_dev, sfixn k, sfixn p, double pinv);


#endif


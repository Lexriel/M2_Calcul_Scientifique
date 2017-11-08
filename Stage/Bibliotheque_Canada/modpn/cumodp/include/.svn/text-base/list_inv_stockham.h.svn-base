#ifndef _LIST_INV_STOCKHAM_H_
#define _LIST_INV_STOCKHAM_H_

#include "../include/defines.h"
#include "../include/rdr_poly.h"
#include "../include/inlines.h"
#include "../include/list_stockham.h"
#include "../include/printing.h"

void list_inv_stockham_host(sfixn *X, sfixn m, sfixn k, sfixn w, sfixn p);

__device__ __global__ void list_inv_mul_ker(sfixn *X, sfixn invn, sfixn m, sfixn p, double pinv, sfixn length_layer);

__device__ __global__ void inv_mul_ker(sfixn *X, sfixn invn, sfixn m, sfixn, sfixn p, double pinv);

__device__ __host__ __inline__ sfixn get_num_of_blocks_for_inv(sfixn m, sfixn k)
{
	sfixn n = (1L << k);
	if (m < 512)
	{
		if (n > m)
			return (n/m + 1);
		else
			return n;
	}
	else
	{
		return ((n * m)/512);
	}
}
__device__ __host__ __inline__ sfixn get_num_of_threads_for_inv(sfixn m, sfixn k)
{
	if (m < 512)
	{
			return m;
	}
	else return 512;
}

#endif

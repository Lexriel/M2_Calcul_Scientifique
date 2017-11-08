#ifndef _LIST_FFT_POLY_MUL_H
#define _LIST_FFT_POLY_MUL_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "../include/list_naive_poly_mul.h"
#include "../include/list_pointwise_mul.h"
#include "../include/list_inv_stockham.h"

#include "../include/subproduct_tree.h"

void list_fft_poly_mul_dev
	(sfixn *M_dev, sfixn length_poly, sfixn *O_dev, sfixn length_result, sfixn num_polys, sfixn start_offset_M, sfixn p, double pinv);


void fft_poly_mul_dev
	(sfixn *M_dev, sfixn length_poly, sfixn *O_dev, sfixn length_result, sfixn num_polys, sfixn start_offset_M, sfixn p, double pinv);

__device__ __global__ void list_add_and_set_1 
	(sfixn *M, sfixn length_poly, sfixn *L, sfixn ln, sfixn *O, sfixn length_result, sfixn num_polys, sfixn start_offset, sfixn p, double pinv);

void list_fft_poly_mul_eff
	(sfixn *M_dev, sfixn length_poly, sfixn *O_dev, sfixn length_result, sfixn num_polys, sfixn start_offset_M, sfixn p, double pinv);


#endif



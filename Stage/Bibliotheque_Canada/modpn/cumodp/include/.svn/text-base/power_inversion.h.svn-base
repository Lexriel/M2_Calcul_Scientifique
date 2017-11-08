#ifndef _LIST_POWER_SERIES_INVERSION_H
#define _LIST_POWER_SERIES_INVERSION_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "../include/subproduct_tree.h"
#include "../include/list_naive_poly_mul.h"
#include "../include/list_pointwise_mul.h"
#include "../include/list_inv_stockham.h"
#include "../include/inlines.h"

__device__ __global__ void reset_ps(sfixn *L, sfixn *G, sfixn length_I, sfixn length_poly,sfixn p);

__device__ __global__ void ps_expand_to_i(sfixn *gg, sfixn *loc, sfixn l_g, sfixn l_f, sfixn length_loc);

__device__ __global__ void ps_add_2g_minus(sfixn *L, sfixn *G, sfixn length, sfixn p, double pinv);

__device__ __global__ void ps_list_poly_mul(sfixn *A, sfixn length_a, sfixn *B, sfixn length_b, sfixn *C, sfixn length_c, sfixn p, double pinv);

__device__ __global__ void ps_pointwise_sq(sfixn *A, sfixn totalLength, sfixn p, double pinv);

__device__ __global__ void ps_pointwise_mul(sfixn *A, sfixn *B, sfixn length, sfixn p, double pinv);

void list_power_series_inversion(sfixn *L_dev, sfixn *I_dev, sfixn length_L, sfixn num_poly,sfixn l, sfixn p);



#endif

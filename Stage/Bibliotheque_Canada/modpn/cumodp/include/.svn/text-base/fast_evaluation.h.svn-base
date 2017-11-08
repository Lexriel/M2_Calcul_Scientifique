#ifndef _FAST_EVALUATION_H_
#define _FAST_EVALUATION_H_

#include "../include/list_fast_division.h"
#include "../include/list_fft_poly_mul.h"
#include "../include/list_naive_poly_mul.h"
#include "../include/list_plain_division.h"
#include "../include/list_poly_rev.h"
#include "../include/power_inversion.h"
#include "../include/subproduct_tree.h"

__device__ __global__ void double_and_copy(sfixn *A, sfixn *E, sfixn small, sfixn bound);
void double_and_copy_tst(sfixn m, sfixn k);
void fast_evaluation_dev(sfixn *M_dev, sfixn k, sfixn *F_dev, sfixn length_F, sfixn p);
void fast_evaluation_host(sfixn *X, sfixn k, sfixn *F, sfixn length_F, sfixn p);
void fast_evaluation_host_bchmk(sfixn *X, sfixn k, sfixn *F, sfixn length_F, sfixn p);
void fast_evaluation_bench(sfixn k);
#endif



#ifndef _NAIVE_POLY_MUL_H_
#define _NAIVE_POLY_MUL_H_

#include <stdio.h>
#include <stdlib.h>

#include "../include/inlines.h"
#include "../include/defines.h"
#include "../include/types.h"
#include "../include/printing.h"
#include "../include/cudautils.h"
#include "../include/rdr_poly.h"

__device__ __global__ void mul_eff_ker(sfixn *A, sfixn *B, sfixn n, sfixn *C, sfixn p, double pinv);

void mul_dev(sfixn *A, sfixn *B, sfixn n, sfixn *C, sfixn p);

__host__ __device__ void mul_ker(sfixn *A, sfixn *B, sfixn n, sfixn *C, sfixn p, double pinv);

#endif

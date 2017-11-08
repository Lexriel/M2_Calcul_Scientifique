#ifndef _LIST_POLY_REV_H_
#define _LIST_POLY_REV_H_
#include <stdio.h>
#include "../include/types.h"
#include "../include/printing.h"
__device__ __global__ void poly_rev(sfixn *F, sfixn length_poly, sfixn bound);

__device__ __global__ void poly_rev_dup(sfixn *F, sfixn length_poly,sfixn *G, sfixn bound);
#endif

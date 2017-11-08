#ifndef _OPT_POLY_DIV_H_
#define _OPT_POLY_DIV_H_

#include <iostream>
#include <ctime>
#include <cmath>

#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/cumodp_simple.h"
#include "../include/defines.h"
#include "../include/rdr_poly.h"
#include "../include/inlines.h"
#include "../include/printing.h"
#include "../include/cudautils.h"
#include "../include/types.h"
#include "../include/list_naive_poly_mul.h"
#include "../include/list_pointwise_mul.h"
#include "../include/list_inv_stockham.h"
#include "../include/subproduct_tree.h"
#include "../include/list_stockham.h"


const int T = 512;
const int s = T;
const int t = 2;

/*
Interesting Problem:
Let d = degree difference
s >= 0
t >= 0
3s + 2t <= 2^12
s <= t
minimize d/s
maximize t
*/


		
__global__ void statusUpdate(sfixn *A,  int *status);
__global__ void zero(sfixn *A, int n);
__global__ void	copyRem(sfixn *A, sfixn *R, int *status);
__global__ void	reduceM(sfixn *B,  int *status);
__global__ void divGPU(sfixn *A, sfixn *B, int *status, sfixn *Q, sfixn *R);
float divPrimeField(sfixn *A, sfixn *B, sfixn *R, sfixn *Q, int n, int m, sfixn p);

#endif

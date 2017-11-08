#ifndef _OPT_POLY_GCD_H_
#define _OPT_POLY_GCD_H_

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


const int T = 480;

__global__ void	reduceMgcd(sfixn *B,  int *status);
__global__ void	reduceNgcd(sfixn *A,  int *status);

__global__ void	status3(int *status);
__global__ void	status4(int *status);
__global__ void	status5(int *status);

__global__ void gcdGPU(sfixn *A, sfixn *B, int *status);


float gcdPrimeField(sfixn *A, sfixn *B, sfixn *G, int n, int m, sfixn p);

#endif

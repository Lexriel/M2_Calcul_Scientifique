#ifndef _LIST_OPT_POLY_MUL_H_
#define _LIST_OPT_POLY_MUL_H_

#include <iostream>
#include <ctime>
#include <cmath>

#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/naive_poly_mul.h"

/*********************************
*    Tx*BLOCK <= 2^11             *
*    (n*m)/(Tx*W^2) <= 2^16       *
*********************************/
//const int BASE_1 = 31;

const int BLOCK = 8;
const int Tx = 128; 
const int T = Tx*BLOCK+BLOCK-1;

sfixn add_modCPU(sfixn a, sfixn b, sfixn P);

sfixn mul_modCPU(sfixn a, sfixn b, int P);

__global__ void zeroC(sfixn *C, int n);

__global__ void copyA(sfixn *A, sfixn *C, int n);
			
__global__ void addFinal(sfixn *Cgpu, sfixn *CgpuFinal, int c, int n, int w, int P);
			
__global__ void add(sfixn *Cgpu, sfixn *CgpuFinal, int c, int n, int w, int i, int j, int k, int P);

__global__ void mul(sfixn *A, sfixn *B, sfixn *C, int n, int m, int c, int P, int unitBlocks);

#endif


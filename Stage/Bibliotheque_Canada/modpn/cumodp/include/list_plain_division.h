#ifndef _LIST_PLAIN_DIVISION_H_
#define _LIST_PLAIN_DIVISION_H_
/*******************************************************************
* This code is intended for for computing remainder i.e f rem m    *
* The degree of m is T*W-1 that is 511 and the degree of f is 1020
********************************************************************/

#include<iostream>
#include <ctime>
#include<cmath>

#include "../include/inlines.h"
#include "../include/printing.h"
#include "../include/types.h"

/*************************************************************
*                 T*W <= 2^9                                 *
* For bigger polynomial, make T = 32 then W will be W = 16   *
*************************************************************/
const int T = 128;
/***********************************************************
* one thread is responsible for computing "W" coefficients *
***********************************************************/
const int W = 4; 


#define BASE_1 31
__global__ void list_divCUDA(sfixn *M, sfixn *F, int start, int length, int threadsPerDiv, int DivPerBlock, int polyNum, int P);

void list_div(sfixn *M, sfixn *F, int n, int m, int  start_offset, int length_poly, int poly_on_layer, int p );

#endif

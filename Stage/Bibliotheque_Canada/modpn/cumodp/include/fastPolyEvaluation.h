#ifndef _FASTPOLYEVALUATION_H
#define _FASTPOLYEVALUATION_H


/*
Author: Sardar Haque Email: haque.sardar@gmail.com

*/


#include <stdio.h>
#include <stdlib.h>

#include <iostream>


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








#define MAX_LEVEL 25

//const int Tmul = 512;
const int Tmax = 512;
const int Tinv = 16;

const int plainMulLimit = 8;

struct PolyEvalSteps
{
	sfixn *Ml[MAX_LEVEL], *Mr[MAX_LEVEL];
	sfixn *InvMl[MAX_LEVEL], *InvMr[MAX_LEVEL];
	sfixn *fL[MAX_LEVEL], *fR[MAX_LEVEL];
	float timeSubTree, timeSinvTree, timeEva;
};

void subProductTree(int k, sfixn p);
void subInvTree(int k, sfixn p);

struct PolyEvalSteps fastEvaluation(sfixn *F, sfixn *points, int k, int p, int flag);	

__global__ void copyMgpu(sfixn *dest, sfixn *source, int length_poly);
__global__ void allZero( sfixn *X, int n);
__global__ void pointAdd2(sfixn *dest, sfixn *source, int l,  int n, int p);
__global__ void zeroInbetween(sfixn *X, sfixn *Y, int n, int l);
__global__ void pointMul(sfixn *dest, sfixn *source, int n, int p);
__global__ void scalarMul(sfixn *A, sfixn ninv, int L, int p);
__global__ void pointAdd(sfixn *dest, sfixn *source, int n, int p);
__global__ void listPlainMulGpu( sfixn *Mgpu1, sfixn *Mgpu2 , int length_poly, int poly_on_layer, int threadsForAmul, int mulInThreadBlock, int p);
__global__ void listPolyinv( sfixn *Mgpu, sfixn *invMgpu,  int poly_on_layer, int prime);
__global__ void listReversePoly(sfixn *revMgpu, sfixn *Mgpu, int length_poly, int poly_on_layer);
__global__ void listCpLdZeroPoly(sfixn *B, sfixn *A, int length_poly, int poly_on_layer);
__global__ void allNeg( sfixn *X, int n, sfixn p);
__global__ void listPolyDegInc( sfixn *Mgpu, sfixn *extDegMgpu, int length_poly, int poly_on_layer, int newLength);
__global__ void listCpUpperCuda(sfixn *dest, sfixn *source, int n, int l);
__global__ void listCpLowerCuda(sfixn *dest, sfixn *source, int n, int l);
__global__ void list2wayCp(sfixn *dest, sfixn *source, int l, int totalLength, sfixn p);
__global__ void	leavesSubproductTree(sfixn *M1gpu, sfixn *Mgpu, int numPoints, int rightSubtree);


#endif


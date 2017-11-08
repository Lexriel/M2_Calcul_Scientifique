/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#include "MultiDFFT.h"

///////////////////////////////////////////////////////////////////////////////
// Log:
//
// WP, 2009/09/03/
//
// (0) Align all the functions.
//
// (1) Functions TFTMultiD and TFTInvMultiD added:
//
//     TFTMultiD is used to evaluate a multivariate polynomial represented by
//     an array of coefficients in a recursive manner.
//
//     TFTInvMultiD is the inverse of TFTMultiD.
//
// (2) Some comments added.
/////////////////////////////////////////////////////////////////////////////////

extern int Interrupted;

//================================================================
//
// All inner functions for Plain or Fast multivariate polynomial 
//  multiplication. Not supposed to be used by the user.
//
// The Exported versions are in "MPMMTS.c"
//
//
//================================================================
 
void decomposePoly2(sfixn N, sfixn *ccum, sfixn *res, sfixn num, sfixn N2, sfixn *dgs2,
                    sfixn *coeffs2, sfixn *ccum2, MONTP_OPT2_AS_GENE *pPtr)
{
    int i, d;
    if (N2==0) {
        res[0]=AddMod(res[0],  MontMulMod_OPT2_AS_GENE(coeffs2[0], num, pPtr), pPtr->P);
        return;
    }

    if (N2==1) { 
        d = shrinkDeg(dgs2[1], coeffs2, ccum2[1]);
        for (i=0; i<=d; i++)
            decomposePoly2(N, ccum, res+ccum[1]*i, num, 0, dgs2, coeffs2+ccum2[1]*i, ccum2, pPtr);
        return;
    }

    d = shrinkDeg(dgs2[N2], coeffs2, ccum2[N2]);
    for (i=0; i<=d; i++){
        decomposePoly2(N, ccum, res+ccum[N2]*i, num, N2-1, dgs2, coeffs2+ccum2[N2]*i, ccum2, pPtr);
    }
}
 
void decomposePoly(sfixn N, sfixn *ccum, sfixn *res, sfixn N1, sfixn *dgs1, sfixn *coeffs1, sfixn *ccum1,
                   sfixn N2, sfixn *dgs2, sfixn *coeffs2, sfixn *ccum2, MONTP_OPT2_AS_GENE *pPtr, sfixn R, sfixn SFT)
{
    int i, d;
    if (N1==0) {
        if (!coeffs1[0]) return;
        R =(MulMod(coeffs1[0], R, pPtr->P))<<SFT;    
        decomposePoly2(N, ccum, res, R, N2, dgs2, coeffs2, ccum2, pPtr); 
        return;
    }

    d = shrinkDeg(dgs1[N1], coeffs1, ccum1[N1]);
    for(i=0; i<=d; i++){
        decomposePoly(N, ccum, res+ccum[N1]*i, N1-1, dgs1, coeffs1+ccum1[N1]*i, ccum1, N2, dgs2, coeffs2, ccum2, pPtr, R, SFT);
    }
}

//================================================================
// Classical Multiplication.
// dgs1[0], dgs2[0] are useless.
// ccumx[i] keeps the base before ith-dimension.
//================================================================
void plainMultiDMul(sfixn N, sfixn *ccum, sfixn *res, sfixn *ccum1, sfixn *dgs1, sfixn *ccum2, sfixn *dgs2,
                    sfixn *coeffs1, sfixn *coeffs2, MONTP_OPT2_AS_GENE *pPtr)
{
    int i, d;
    sfixn p = pPtr->P, R = (1L<<pPtr->Rpow)%p, SFT = pPtr->Base_Rpow;
    d = shrinkDeg(dgs1[N], coeffs1, ccum1[N]);
    for (i=0; i<=d; i++){
        decomposePoly(N, ccum, res+ccum[N]*i,  N-1, dgs1, coeffs1+ccum1[N]*i, ccum1, N, dgs2, coeffs2, ccum2, pPtr, R, SFT);
    }
}

/* Example
 *
 * Let f = a+2*b+3*b*a+4*b^2+5*b^2*a+6*c+7*c*a+8*c*b+9*c*b*a+10*c*b^2+11*c*b^2*a+
 *         12*c^2+13*c^2*a+14*c^2*b+15*c^2*b*a+16*c^2*b^2+17*c^2*b^2*a
 *
 * (0) N = 3
 *
 *     The number of variables
 *
 * (1) Represent f as a 1D array
 *
 *     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16, 17]
 *
 *     This vector is given by coeffs.
 *
 * (2) ccum = [0, 1, 2, 6]
 *
 *     f can be written recursively as
 *
 *     f = ( (0+1*a) + (2+3*a)*b + (4+5*a)*b^2 )
 *         +
 *         ( (6+7*a) + (8+9*a)*b + (10+11*a)*b^2 )*c
 *         +
 *         ( (12+13*a) + (14+15*a)*b + (16+17*a)*b^2 )*c^2
 *
 *     The size of coefficients of the variable a is 1; represented by ccum[1] = 1;
 *     The size of coefficients of the variable b is 2; represented by ccum[2] = 2;
 *     The size of coefficients of the variable c is 6; represented by ccum[3] = 6;
 *
 * (3) deg = [0, 1, 2, 2]
 *
 *     The partial degree of f in a is 1; represented by deg[1] = 1;
 *     The partial degree of f in b is 2; represented by deg[2] = 2;
 *     The partial degree of f in c is 2; represented by deg[3] = 2;
*
 * (4) dims = [0, 2, 4, 4]
 *
 *     dims[1] = 2^deg[1] = 2;
 *     dims[2] = 2^deg[2] = 4;
 *     dims[3] = 2^deg[3] = 4;
 *
 * (5) rccum = [0, 1, 2, 8]
 *
 *     rccum[1] = 1;
 *     rccum[2] = rccum[1] * dims[1] = 2;
 *     rccum[3] = rccum[2] * dims[2] = 8;
 *
 *
 * (6) The following will rerange coeffs as follows
 *
 *
 *       0  1  2  3  4  5  0  0 ,
 *
 *       6  7  8  9 10 11  0  0 ,
 *
 *      12 13 14 15 16 17  0  0
 *
 *     The first row corresponds to the coefficient  (0+1*a) + (2+3*a)*b + (4+5*a)*b^2 + (0+0*a)*b^3
 *     The second row corresponds to the coefficient (6+7*a) + (8+9*a)*b + (10+11*a)*b^2 + (0+0*a)*b^3
 *     The third row corresponds to the coefficient  (12+13*a) + (14+15*a)*b + (16+17*a)*b^2 + (0+0*a)*b^3
 *
 *     Under the above arrangement, each coefficient is ready for FFT.
 *
 *     Note that the first row can be viewed as
 *
 *     0 1  ==> 0+1*a
 *     2 3  ==> 2+3*a
 *     4 5  ==> 4+5*a
 *     0 0  ==> 0+0*a
 *
 *  In summary, the procedure fromtofftRepMultiD copies the input multivariate coefficients
 *  into an array which is ready from in-place FFT or Inverse FFT. More precisely, along each
 *  dimension, the size is expanded to a power of 2.
 *
 */

// suppose y>x
// m rows, n columns
// dgs is always the smaller one.

void fromtofftRepMultiD(sfixn N, sfixn *rccum, sfixn *res, sfixn *ccum,  sfixn *dgs, sfixn *coeffs){
    int i, d, tmpRes = 0, tmpCoeffs = 0;
    if (N==0){
        res[0] = coeffs[0];
        return;
    }
    if (N==1) {
        d = shrinkDeg(dgs[1], coeffs, 1);
        for(i=0; i<=d; i++){ res[i] = coeffs[i]; }
        return;
    }
    d = shrinkDeg(dgs[N], coeffs, ccum[N]);
    for (i=0; i<=d; i++){
        tmpCoeffs = i*ccum[N];
        tmpRes = i*rccum[N];
        fromtofftRepMultiD(N-1, rccum, res+tmpRes, ccum, dgs, coeffs+tmpCoeffs);
    }
}

//================================================================
// Multi-dimensional FFT.
//
// es[0] and dims[0] are useless;
// es[i] is the exponent of dims[i];
// dims[i] is the # of elements on dims-i
//================================================================
/**
 * fftMultiD_test:
 * @coeffs1: coefficient vector for 'f1'.
 * @coeffs2: coefficient vector for 'f2'.
 * @N: number of variables.
 * @es: 2^es[i] = dims [i] for i = 1..n.
 * @dims: the FFT sizes on each dimension, dims[i] is the FFT on dimensional i, i=1..N.
 * @pPtr: the information of the prime number.
 * 
 * Return value: 
 **/
void fftMultiD_test(sfixn *coeffs1, sfixn *coeffs2, sfixn N, sfixn *es, sfixn *dims, MONTP_OPT2_AS_GENE *pPtr)
{ 
    sfixn i, j, m = 0, n = 1, tmp;
    sfixn *rootsPtr, *tmprootsPtr;

    signal(SIGINT, catch_intr);

    //printf("\ndims in fftMultiD_test");
    //printVec(N,dims);
    //printf("\n");
    for (i=1; i<=N; i++) { n *= dims[i]; m += dims[i]; } 
    rootsPtr = (sfixn *) my_calloc(m, sizeof(sfixn));
    tmprootsPtr = rootsPtr;

    // Multi-DFT
    if (es[1]) {
        EX_Mont_GetNthRoots_OPT2_AS_GENE(es[1], dims[1], tmprootsPtr, pPtr);
        for(j=0; j<n; j+=dims[1]) {
            EX_Mont_DFT_OPT2_AS_GENE_1 (dims[1], es[1], tmprootsPtr, coeffs1+j, pPtr);
            EX_Mont_DFT_OPT2_AS_GENE_1 (dims[1], es[1], tmprootsPtr, coeffs2+j, pPtr);
        }  
    }

    for (i=2; i<=N; i++) {
        // last dim -> [1]
        if(Interrupted==1) { my_free(rootsPtr); return; }

        tmprootsPtr += dims[1];
        multi_mat_transpose(N, n, i, dims, coeffs1);
        multi_mat_transpose(N, n, i, dims, coeffs2);

        if (es[i]) {
            EX_Mont_GetNthRoots_OPT2_AS_GENE(es[i], dims[i], tmprootsPtr, pPtr);
            for (j=0; j<n; j+=dims[i]) {
                EX_Mont_DFT_OPT2_AS_GENE_1(dims[i], es[i], tmprootsPtr, coeffs1+j, pPtr);
                EX_Mont_DFT_OPT2_AS_GENE_1(dims[i], es[i], tmprootsPtr, coeffs2+j, pPtr);
            }
        }
        // multi_mat_transpose didn't change dims accordings, so we need to do this.
        tmp = dims[1];
        dims[1] = dims[i];
        dims[i] = tmp;
        tmp = es[1];
        es[1] = es[i];
        es[i] = tmp;
    }

    //Pairwise-Mul
    EX_Mont_PairwiseMul_OPT2_AS(n, coeffs1, coeffs2, pPtr->P);

    //Multi-invdft
    if (es[1]) {
        for (j=0; j<n; j+=dims[1]) {
            EX_Mont_INVDFT_OPT2_AS_GENE_1(dims[1], es[1], tmprootsPtr, coeffs1+j, pPtr);
        }
    }
    for (i=N; i>=2; i--) {
        tmprootsPtr -= dims[i];
        multi_mat_transpose(N, n, i, dims, coeffs1);
        if (es[i]) {
            for (j=0; j<n; j+=dims[i]) {
	            EX_Mont_INVDFT_OPT2_AS_GENE_1(dims[i], es[i], tmprootsPtr, coeffs1+j, pPtr);
            }
        }
        //multi_mat_transpose didn't change dims accordings, so we need to do this.
        tmp = dims[1];
        dims[1] = dims[i];
        dims[i] = tmp;
        tmp = es[1];
        es[1] = es[i];
        es[i] = tmp;
    }
    my_free(rootsPtr);
}

/**
 * TFTInvMultiD:
 * @coeffs: coefficient vector for 'f'.
 * @N: number of variables.
 * @es: 2^es[i] = dims[i] for i = 1..N.
 * @dims: the FFT sizes on each dimension, dims[i] is the FFT in dimensional i, i=1..N.
 * @ls:   the TFT size in each dimension, ls[i] is the TFT size in dimension i, i=1..N
 * @rootsPtr : an array of primitive roots of unity
 *             1, w1, w1^2, ..., w1^{dims[1]-1},
 *             1, w2, w2^2, ..., w2^{dims[2]-1}},
 *             ...
 *             1, wN, wN^2, ..., wN^{dims[N]-1}
 *
 *             In total, the size of rootsPtr is dims[1] + ... + dims[N]
 * @pPtr: the information of the prime number.
 *
 * Return value:
 *
 * (1) coeffs will be updated by its inverse TFT.
 *
 * (2) es, dims, ls, rootsPtr remains unchanged overall.
 *
 **/
void TFTInvMultiD (sfixn *coeffs, sfixn N, sfixn *es, sfixn *dims, sfixn *ls, sfixn *rootsPtr, MONTP_OPT2_AS_GENE *pPtr)
{
    sfixn i, j, k, m=0, n=1, tmp, maxdim=0;
    sfixn *tmprootsPtr, *tmpVec;

    signal(SIGINT, catch_intr);

    // n is the size of coeffs.
    // maxdim is the maximal value in all direction
    for (i=1; i<=N; i++) {
        n *= ls[i];
        m += dims[i];
        if (dims[i]>maxdim) maxdim = dims[i];
    }

    // Current work space.
    tmpVec = (sfixn *)my_calloc(maxdim, sizeof(sfixn));
    // Pointer to the current primitive roots of unity.
    tmprootsPtr = rootsPtr;
    if (es[1]) {
        for (j=0; j<n; j+=ls[1]) {
            copyVec_0_to_d(ls[1]-1, tmpVec, coeffs+j);
            for (k=ls[1]; k<dims[1]; k++) tmpVec[k]=0;
            EX_Mont_INVTDFT_OPT2_AS_GENE_1(ls[1], tmprootsPtr, tmpVec, pPtr);
            copyVec_0_to_d(ls[1]-1, coeffs+j, tmpVec);
        }
    }

    for (i=2; i<=N; i++) {
        // dims[1] is ALWAYS the number of unities used in the last step.
        // dims[1] will be updated constantly.
        tmprootsPtr += dims[1];
        multi_mat_transpose(N, n, i, ls, coeffs);
        if (es[i]) {
            cleanVec(maxdim-1, tmpVec);
            for (j=0; j<n; j+=ls[i]) {
                copyVec_0_to_d(ls[i]-1, tmpVec, coeffs+j);
                EX_Mont_INVTDFT_OPT2_AS_GENE_1(ls[i], tmprootsPtr, tmpVec, pPtr);
                copyVec_0_to_d(ls[i]-1, coeffs+j, tmpVec);
                for (k=ls[i]; k<dims[i]; k++) tmpVec[k]=0;
            }
        }
        // multi_mat_transpose didn't change dims accordings,
        // so we need to do this.
        tmp = dims[1];
        dims[1] = dims[i];
        dims[i] = tmp;
        tmp = es[1];
        es[1] = es[i];
        es[i] = tmp;
        tmp = ls[1];
        ls[1] = ls[i];
        ls[i] = tmp;
    }
    // Transpose evaluation martrix, dims, es and ls back to the orginal order.
    for (i=N; i>=2; i--) {
        multi_mat_transpose(N, n, i, ls, coeffs);
        tmp = dims[1];
        dims[1] = dims[i];
        dims[i] = tmp;

        tmp = es[1];
        es[1] = es[i];
        es[i] = tmp;

        tmp = ls[1];
        ls[1] = ls[i];
        ls[i] = tmp;
    }
    my_free(tmpVec);
}

/**
 * TFTMultiD:
 * @coeffs: coefficient vector for 'f'.
 * @N: number of variables.
 * @es: 2^es[i] = dims[i] for i = 1..N.
 * @dims: the FFT sizes on each dimension, dims[i] is the FFT in dimensional i, i=1..N.
 * @ls:   the TFT size in each dimension, ls[i] is the TFT size in dimension i, i=1..N
 * @rootsPtr : an array of primitive roots of unity
 *             1, w1, w1^2, ..., w1^{dims[1]-1},
 *             1, w2, w2^2, ..., w2^{dims[2]-1}},
 *             ...
 *             1, wN, wN^2, ..., wN^{dims[N]-1}
 *
 *             In total, the size of rootsPtr is dims[1] + ... + dims[N]
 * @pPtr: the information of the prime number.
 *
 * Return value:
 *
 * (1) coeffs will be filled by it TFT evaluations.
 *
 * (2) dims, es and ls will be transposed back in the end and remains unchanged.
 *     coeffs will be transposed back also.
 *
 * (3) rootsPtr remains unchanged.
 **/
void TFTMultiD(sfixn *coeffs, sfixn N, sfixn *es, sfixn *dims, sfixn *ls, sfixn *rootsPtr, MONTP_OPT2_AS_GENE *pPtr)
{
    sfixn i, j, k,  m=0, n=1, tmp, maxdim=0;
    sfixn *tmprootsPtr, *tmpVec;
    sfixn R=(1L<<pPtr->Rpow)%pPtr->P;
    sfixn BRsft=pPtr->Base_Rpow;

    signal(SIGINT, catch_intr);

    for (i=1; i<=N; i++) {
        n *= ls[i];
        m += dims[i];
        if (dims[i]>maxdim) maxdim = dims[i];
    }
    
    //printf("Primitive roots used:\n");
    //printf("m=%d\n", m);
    //for (i=0; i<m; i++) printf("%d  ", QuoMod(((usfixn)rootsPtr[i])>>BRsft, R, pPtr->P));
    //printf("\n");
    
    tmprootsPtr = rootsPtr;
    tmpVec = (sfixn *)my_calloc(maxdim, sizeof(sfixn));
    if (es[1]) {
        for (j=0; j<n; j+=ls[1]) {
            copyVec_0_to_d(ls[1]-1, tmpVec, coeffs+j);
            EX_Mont_TDFT_OPT2_AS_GENE_1 (ls[1], tmprootsPtr, tmpVec, pPtr);
            copyVec_0_to_d(ls[1]-1, coeffs+j, tmpVec);
            for (k=ls[1]; k<dims[1]; k++) tmpVec[k]=0;
        }
    }

    for (i=2; i<=N; i++) {
        if(Interrupted==1) { my_free(tmpVec); return; }
        // dims[1] is ALWAYS the number of unities used in the last step.
        // dims[1] will be updated constantly.
        tmprootsPtr += dims[1];
        multi_mat_transpose(N, n, i, ls, coeffs);

        if (es[i]) {
            cleanVec(maxdim-1, tmpVec);
            for (j=0; j<n; j+=ls[i]) {
                copyVec_0_to_d(ls[i]-1, tmpVec, coeffs+j);
                EX_Mont_TDFT_OPT2_AS_GENE_1 (ls[i], tmprootsPtr, tmpVec, pPtr);
                copyVec_0_to_d(ls[i]-1, coeffs+j, tmpVec);
                for (k=ls[i]; k<dims[i]; k++) tmpVec[k]=0;
            }
        }
        // multi_mat_transpose didn't change dims accordings, so we need to do this.
        tmp = dims[1];
        dims[1] = dims[i];
        dims[i] = tmp;
        tmp = es[1];
        es[1] = es[i];
        es[i] = tmp;
        tmp = ls[1];
        ls[1] = ls[i];
        ls[i] = tmp;
    }
    // Transpose evaluation martrix, dims, es and ls back to the orginal order.
    for (i=N; i>=2; i--) {
        multi_mat_transpose (N, n, i, ls, coeffs);

        tmp = dims[1];
        dims[1] = dims[i];
        dims[i] = tmp;

        tmp = es[1];
        es[1] = es[i];
        es[i] = tmp;

        tmp = ls[1];
        ls[1] = ls[i];
        ls[i] = tmp;
    }
    my_free(tmpVec);
}

//================================================================
// Squaring by Multi-dimensional FFT.
//
// es[0] and dims[0] are useless;
// es[i] is the exponent of dims[i];
// dims[i] is the # on dims-i
//================================================================

void fftMultiD_square_test(sfixn *coeffs1, sfixn N, sfixn *es, sfixn *dims, MONTP_OPT2_AS_GENE *pPtr)
{ 
    sfixn i, j, m=0, n=1, tmp;
    sfixn *rootsPtr, *tmprootsPtr;

    signal(SIGINT, catch_intr);
    // printf("\ndims in fftMultiD_test");
    // printVec(N,dims);
    // printf("\n");
    for (i=1; i<=N; i++) { n *= dims[i]; m += dims[i]; } 
    rootsPtr =(sfixn *) my_calloc(m, sizeof(sfixn));
    tmprootsPtr = rootsPtr;

    // Multi-DFT
    if (es[1]) {
        EX_Mont_GetNthRoots_OPT2_AS_GENE(es[1], dims[1], tmprootsPtr, pPtr);
        for (j=0; j<n; j+=dims[1]) {
            EX_Mont_DFT_OPT2_AS_GENE_1 (dims[1], es[1], tmprootsPtr, coeffs1+j, pPtr);
        }  
    }

    for (i=2; i<=N; i++) {
        // last dim -> [1]
        tmprootsPtr += dims[1];
        multi_mat_transpose(N, n, i, dims, coeffs1);

        if (es[i]) {
            EX_Mont_GetNthRoots_OPT2_AS_GENE(es[i], dims[i], tmprootsPtr, pPtr);
            for (j=0; j<n; j+=dims[i]) {
                EX_Mont_DFT_OPT2_AS_GENE_1(dims[i], es[i], tmprootsPtr, coeffs1+j, pPtr);
            }
        }
        // multi_mat_transpose didn't change dims accordings, so we need to do this.
        tmp = dims[1];
        dims[1] = dims[i];
        dims[i] = tmp;
        tmp = es[1];
        es[1] = es[i];
        es[i] = tmp;
    }

    // Pairwise-Mul
    EX_Mont_PairwiseMul_OPT2_AS(n, coeffs1, coeffs1, pPtr->P);

    // Multi-invdft
    if (es[1]) {
        for (j=0; j<n; j+=dims[1]) {
            EX_Mont_INVDFT_OPT2_AS_GENE_1(dims[1], es[1], tmprootsPtr, coeffs1+j, pPtr);
        }
    }

    for (i=N; i>=2; i--) {
        if (Interrupted==1) { my_free(rootsPtr); return; }
        tmprootsPtr -= dims[i];
        multi_mat_transpose(N, n, i, dims, coeffs1);

        if (es[i]) {
            for (j=0; j<n; j+=dims[i]) {
	            EX_Mont_INVDFT_OPT2_AS_GENE_1(dims[i], es[i], tmprootsPtr, coeffs1+j, pPtr);
            }
        }
        // multi_mat_transpose didn't change dims accordings, so we need to do this.
        tmp = dims[1];
        dims[1] = dims[i];
        dims[i] = tmp;
        tmp = es[1];
        es[1] = es[i];
        es[i] = tmp;
    }

    my_free(rootsPtr);
}


//================================================================
// Multi-dimensional TFT
//================================================================
/**
 * tftMultiD_test:
 * @coeffs1: coefficient vector for 'f1'.
 * @coeffs2: coefficient vector for 'f2'.
 * @N: number of variables.
 * @es: 2^es[i] = dims [i] for i = 1..n.
 * @dims: the FFT sizes on each dimension, dims[i] is the FFT on dimensional i, i=1..N.
 * @pPtr: the information of the prime number.
 * 
 * Return value: 
 **/
void tftMultiD_test(sfixn *coeffs1, sfixn *coeffs2, sfixn N, sfixn *es, sfixn *dims, sfixn *ls,  MONTP_OPT2_AS_GENE *pPtr)
{ 
    sfixn i, j, k,  m=0, n=1, tmp, maxdim=0;
    sfixn *rootsPtr, *tmprootsPtr, *tmpVec;

    signal(SIGINT, catch_intr);

    //printf("\ndims in fftMultiD_test");
    //printVec(N,dims);
    //printf("\n");
    for (i=1; i<=N; i++) {
        n *= ls[i]; 
        m += dims[i]; 
        if (dims[i]>maxdim) maxdim = dims[i];
    } 
    rootsPtr = (sfixn *)my_calloc(m, sizeof(sfixn));
    tmprootsPtr = rootsPtr;
    tmpVec = (sfixn *)my_calloc(maxdim, sizeof(sfixn));
  
    // Multi-DFT
    if (es[1]) {
        EX_Mont_GetNthRoots_OPT2_AS_GENE(es[1], dims[1], tmprootsPtr,  pPtr);
        for (j=0; j<n; j+=ls[1]) {
            copyVec_0_to_d(ls[1]-1, tmpVec, coeffs1+j);
            EX_Mont_TDFT_OPT2_AS_GENE_1(ls[1], tmprootsPtr, tmpVec, pPtr);
            copyVec_0_to_d(ls[1]-1, coeffs1+j, tmpVec);
            copyVec_0_to_d(ls[1]-1, tmpVec, coeffs2+j);
            for (k=ls[1]; k<dims[1]; k++) tmpVec[k]=0;       
            EX_Mont_TDFT_OPT2_AS_GENE_1(ls[1], tmprootsPtr, tmpVec, pPtr);
            copyVec_0_to_d(ls[1]-1, coeffs2+j, tmpVec);
            for (k=ls[1]; k<dims[1]; k++) tmpVec[k]=0;  
        }
    }

    for (i=2;i<=N;i++) {
        // last dim -> [1]
        if (Interrupted==1) { my_free(rootsPtr); my_free(tmpVec); return; }
        tmprootsPtr += dims[1];
        multi_mat_transpose(N, n, i, ls, coeffs1);
        multi_mat_transpose(N, n, i, ls, coeffs2);

        if (es[i]) {
            EX_Mont_GetNthRoots_OPT2_AS_GENE(es[i], dims[i], tmprootsPtr, pPtr);
            cleanVec(maxdim-1, tmpVec);
            for (j=0; j<n; j+=ls[i]) {
                copyVec_0_to_d(ls[i]-1, tmpVec, coeffs1+j); 
                EX_Mont_TDFT_OPT2_AS_GENE_1(ls[i], tmprootsPtr, tmpVec, pPtr);
                copyVec_0_to_d(ls[i]-1, coeffs1+j, tmpVec);
                copyVec_0_to_d(ls[i]-1, tmpVec, coeffs2+j);
                for (k=ls[i]; k<dims[i]; k++) tmpVec[k]=0;  
                EX_Mont_TDFT_OPT2_AS_GENE_1(ls[i], tmprootsPtr, tmpVec, pPtr);
                copyVec_0_to_d(ls[i]-1, coeffs2+j, tmpVec); 
                for (k=ls[i]; k<dims[i]; k++) tmpVec[k]=0;
            }
        }
        // multi_mat_transpose didn't change dims accordings, so we need to do this.
        tmp = dims[1];
        dims[1] = dims[i];
        dims[i] = tmp;
        tmp = es[1];
        es[1] = es[i];
        es[i] = tmp;
        tmp = ls[1];
        ls[1] = ls[i];
        ls[i] = tmp;
    }

    // Pairwise-Mul
    EX_Mont_PairwiseMul_OPT2_AS(n, coeffs1, coeffs2, pPtr->P);

    // Multi-invdft
    if (es[1]) {
        for (j=0; j<n; j+=ls[1]) {
            copyVec_0_to_d(ls[1]-1, tmpVec, coeffs1+j);   
            for (k=ls[1]; k<dims[1]; k++) tmpVec[k]=0;  
            EX_Mont_INVTDFT_OPT2_AS_GENE_1(ls[1], tmprootsPtr, tmpVec, pPtr);
            copyVec_0_to_d(ls[1]-1, coeffs1+j, tmpVec);
            for (k=ls[1]; k<dims[1]; k++) tmpVec[k]=0;  
        }
    }

    for (i=N; i>=2; i--) {
        tmprootsPtr -= dims[i];
        multi_mat_transpose(N, n, i, ls, coeffs1);

        if (es[i]) {
            cleanVec(maxdim-1, tmpVec);
            for (j=0; j<n; j+=ls[i]) {
                copyVec_0_to_d(ls[i]-1, tmpVec, coeffs1+j); 
                EX_Mont_INVTDFT_OPT2_AS_GENE_1(ls[i], tmprootsPtr, tmpVec, pPtr);
                copyVec_0_to_d(ls[i]-1, coeffs1+j, tmpVec);  
                for (k=ls[i]; k<dims[i]; k++) tmpVec[k]=0;
            }
        }
 
        // multi_mat_transpose didn't change dims accordings, so we need to do this.
        tmp = dims[1];
        dims[1] = dims[i];
        dims[i] = tmp;
        tmp = es[1];
        es[1] = es[i];
        es[i] = tmp;
        tmp = ls[1];
        ls[1] = ls[i];
        ls[i] = tmp;
    }
  
    my_free(rootsPtr);
    my_free(tmpVec);
}

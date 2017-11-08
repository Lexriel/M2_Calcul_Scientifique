/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#include "IteratedResultant.h"

extern int Interrupted;
extern int PrimeError;

extern int CUDA_TAG;
////////////////////////////////////////////////////////////////////////////////
//   Initialization of SCUBE data structure
////////////////////////////////////////////////////////////////////////////////

/**
 *  Example :
 *
 *  P = (1 + 2 x) + (3 + 4 x) y + (5 + 6 x) y^2 + (7 + 8 x) y^3
 *  Q = (6 + 5 x) + (4 + 3 x) y + (2 + 1 x) y^2 
 * 
 *  Thus the degree of the resultant of P and Q in x is
 *
 *  1 * 3 + 1 * 2 + 1 = 6
 *
 *  Then we have 
 *
 *       M = 1                the number of variable to be evaluated;
 *       w = 2                the smaller degree of P and Q.
 *  bounds = [0, 6]           the degree bound in x is 6
 *     dim = 3                the array length
 *   Sdims = [0, 6, 2, 2]     the last data should be useless!
 *   Ssize = 6 * 2 * 2        the size of Sdata
 * doneBfr = 0
 */
SPTreeChain_t* InitSPTreeChain(sfixn M, sfixn *bounds, sfixn w) {
    int i;
    SPTreeChain_t *scube;
    scube = (SPTreeChain_t *)my_malloc(sizeof(SPTreeChain_t));
    scube->w = w;
    scube->dim = M + 2;
    scube->Sdims = (sfixn *) my_calloc((scube->dim) + 1, sizeof(sfixn));
    scube->points_trees = NULL;
    scube->Ssize = 1;
    for (i = 1; i <= M; i++) {
        scube->Sdims[i] = bounds[i];
        scube->Ssize *= scube->Sdims[i];   
    }
    scube->Sdims[M+1] = scube->Sdims[M+2] = w;
    scube->Ssize *= w * w;
    scube->Sdata = (sfixn *)my_calloc(scube->Ssize, sizeof(sfixn));
    scube->SLcs = (preFFTRep **) my_calloc(w, sizeof(preFFTRep *));
    scube->SPolys = (preFFTRep **)my_calloc(w, sizeof(preFFTRep *));
    scube->doneBfr = 0;
    return scube;
}

void FreeSPTreeChain(SPTreeChain_t *scube) {
    int i;
    if (scube != NULL) {
        if(scube->Sdims != NULL) my_free(scube->Sdims);
        if(scube->Sdata != NULL) my_free(scube->Sdata);
        if(scube->points_trees != NULL){ freePTS_TREE(scube->points_trees); }
        if(scube->SLcs !=NULL) {
            for(i = 1; i < scube->w; i++){ EX_FreeOnePoly((scube->SLcs)[i]); }
            my_free(scube->SLcs);
        }
        if(scube->SPolys != NULL) {
            for(i = 1; i < scube->w; i++){ EX_FreeOnePoly((scube->SPolys)[i]); }
            my_free(scube->SPolys);
        }
        my_free(scube);
    }
}

void PrintSPTreeChain(SPTreeChain_t *scube){
    sfixn i, N, w; 
    printf("...................................\n");
    printf("Subproduct tree subresultant chain\n");
    if (scube == NULL) { printf("NULL scube!\n"); return; }
    N = scube->dim;
    w = scube->w;
    printf("scube->w = %d\n", scube->w); 
    printf("scube->dim = %d\n", scube->dim);  
    printVec(N, scube->Sdims);
    printf("scube->Ssize = %d\n", scube->Ssize); 
    printf("scube->Sdata = ...\n"); 
    printVec((scube->Ssize)-1, scube->Sdata);
    printf("scube->points_trees = ...\n"); 
    printf("scube->SLcs = \n");
    for(i = 0; i < w; i++){
        printf("SLcs[%d] = ", i);
        printPoly(scube->SLcs[i]);
    }
    for(i = 0; i < w; i++){
        printf("SPolys[%d] = ", i);
        printPoly(scube->SPolys[i]);
    }
    printf("scube->doneBfr = %d\n", scube->doneBfr);
    printf("...................................\n");
}

/**
 *  Example :
 *
 *  P = (1 + 2 x) + (3 + 4 x) y + (5 + 6 x) y^2 + (7 + 8 x) y^3
 *  Q = (6 + 5 x) + (4 + 3 x) y + (2 + 1 x) y^2 
 * 
 *  Thus the degree of the resultant of P and Q in x is
 *
 *  1 * 3 + 1 * 2 + 1 = 6 < 2^3 = 8
 *
 *  Then we have 
 *
 *       M = 1                the number of variable to be evaluated;
 *       w = 2                the smaller degree of P and Q.
 *   Sdims = [0, 8]           the FFT sizes in each variable for i = 1..M
 *     Ses = [0, 3]           the exponents of FFT sizes
 *   Ssize = 8 * 2 * 2        the size of Sdata
 */
DftChain_t* InitDftChain(sfixn M, sfixn *es, sfixn w) {
    int i;
    DftChain_t *scube;
    scube = (DftChain_t *)my_malloc(sizeof(DftChain_t));
    scube->w = w;
    scube->dim = M + 2;
    scube->Sdims = (sfixn *) my_calloc(M + 3, sizeof(sfixn));
    scube->Sdims[0] = 0;
    scube->Sdims[M + 1] = w;
    scube->Sdims[M + 2] = w;
    scube->Ses = (sfixn *) my_calloc(M + 1, sizeof(sfixn));
    scube->Ses[0] = 0;
    scube->Ssize = 1;
    for (i = 1; i <= M; i++) {
        scube->Ses[i] = es[i];
        scube->Sdims[i] = ((sfixn)1 << es[i]);
        scube->Ssize *= scube->Sdims[i];   
    }
    scube->roots = NULL;
    scube->Ssize *= w * w;
    scube->Sdata = (sfixn *)my_calloc(scube->Ssize, sizeof(sfixn));
    scube->SLcs = (preFFTRep **) my_calloc(w, sizeof(preFFTRep *));
    scube->SPolys = (preFFTRep **)my_calloc(w, sizeof(preFFTRep *));
    return scube;
}

/**
 *  Conversion from ListTriangle to ListRectangle (N = 2)
 *
 *  @B, the number of images
 *  @w, the size of the first subresultant
 *  @dout, the output of size B * w * w
 *  @din, the input of size B * w * (w + 1) / 2 
 *
 *  For each image, the transformation is 
 *
 *  0  1  2        5  0  0
 *  3  4     ===>  3  4  0
 *  5              0  1  2
 */
void ListTri2Rect(sfixn b, sfixn w, sfixn *dout, const sfixn *din)
{
    sfixn i; // image index, from 0 to B - 1
    sfixn j; // resultant index, from w - 1 to 0
    sfixn k; // coefficient index of each subresultant 
    sfixn sz = (w * w + w) / 2; // triangular chain size
    sfixn ww = w * w;           // rectangle chain size
    sfixn offset, ri, B = ((sfixn)1 << b);

    for (i = 0; i < B; ++i) {
        offset = 0;
        ri = partialBitRev(i, b); // Take the bit-reversal ordering
                                  // to consider the bivariate case   
        for (j = w - 1; j >= 0; --j) {
            for (k = 0; k <= j; ++k) {
                dout[i * ww + j * w + k] = din[ri * sz + offset + k];
            }
            offset += (j + 1);
        }
    }
}

/**
 * N = 3.
 *
 * Example: Normal order is
 *
 * (0, 0)  (1, 0)  (2, 0)  (3, 0)
 * (0, 1)  (1, 1)  (2, 1)  (3, 1)
 * (0, 2)  (1, 2)  (2, 2)  (3, 2)
 * (0, 3)  (1, 3)  (2, 3)  (3, 3)
 *
 * Taking the bit-reversal permutation in both directions gives
 *
 * (0, 0)  (2, 0)  (1, 0)  (3, 0)
 * (0, 2)  (2, 2)  (1, 2)  (3, 2)
 * (0, 1)  (2, 1)  (1, 1)  (3, 1)
 * (0, 3)  (2, 3)  (1, 3)  (3, 3)
 *
 * Taking the transposition gives
 *
 * (0, 0)  (0, 2)  (0, 1)  (0, 3)
 * (2, 0)  (2, 2)  (2, 1)  (2, 3) 
 * (1, 0)  (1, 2)  (1, 1)  (1, 3) 
 * (3, 0)  (3, 2)  (3, 1)  (3, 3)
 *
 */
void ListTri3Rect(sfixn bx, sfixn by, sfixn w, sfixn *dout, const sfixn *din) {
    sfixn rx, ix; // image index, from 0 to Bx - 1
    sfixn ry, iy; // image index, from 0 to Bx - 1
    sfixn j; // resultant index, from w - 1 to 0
    sfixn k; // coefficient index of each subresultant 
    sfixn sz = (w * w + w) / 2; // triangular chain size
    sfixn ww = w * w;           // rectangle chain size
    sfixn Bx = ((sfixn)1 << bx), By = ((sfixn)1 << by);
    sfixn offset, ii, io;

    for (iy = 0; iy < By; ++iy) {
        ry = partialBitRev(iy, by); 
        for (ix = 0; ix < Bx; ++ix) {
            rx = partialBitRev(ix, bx);
            ii = ((iy << bx) + ix) * sz;  // offset for element (ix, iy) input
            io = ((rx << by) + ry) * ww;  // offset for element (ry, rx) output
            offset = 0;
            for (j = w - 1; j >= 0; --j) {
                for (k = 0; k <= j; ++k) {
                    dout[io + j * w + k] = din[ii + offset + k];
                }
                offset += (j + 1);
            }
        }
    }
}

/**
 * Conversion from StrideTriangle to ListRectangle
 *
 * @B, the number of images
 * @w, the size of the first subresultant
 * @dout, the output of size B * w * w
 * @din, the input of size B * w * (w + 1) / 2
 *
 * Example, the transformation is from
 *
 * 0 1 2 -> 6  7  8  -> 12 13 14 -> 18 19 20   
 * 3 4   -> 9 10     -> 15 16    -> 21 22 
 * 5     -> 11       -> 17       -> 23
 * 
 * 5  0  0     11  0   0       17   0   0     23  0   0
 * 3  4  0     9   10  0       15  16   0     21  22  0
 * 0  1  2     6   7   8       12  13  14     18  19  0
 * 
 */
void StrideTri2ListRect(sfixn b, sfixn w, sfixn *dout, const sfixn *din) {
    sfixn i; // image index, from 0 to B - 1
    sfixn j; // the subresultant index from w - 1 to 0
    sfixn k; // coefficient index of each subresultant 
    sfixn ww = w * w;
    sfixn offset = 0;
    sfixn j1, B = ((sfixn)1 << b), ii, io;

    for (j = w - 1; j >= 0; --j) {
        j1 = j + 1;
        for (i = 0; i < B; ++i) {
            ii = partialBitRev(i, b) * j1;
            io = i * ww;
            for (k = 0; k <= j; ++k) {
                dout[io + j * w + k] = din[ii + offset + k];
            }
        }
        offset += (B * j1);
    }
}

void StrideTri3ListRect(sfixn bx, sfixn by, sfixn w, sfixn *dout, 
    const sfixn *din) 
{
    sfixn rx, ix; // image index, from 0 to Bx - 1
    sfixn ry, iy; // image index, from 0 to By - 1
    sfixn j;  // the subresultant index from w - 1 to 0
    sfixn k;  // coefficient index of each subresultant 
    sfixn ww = w * w;
    sfixn Bx = ((sfixn)1 << bx), By = ((sfixn)1 << by);
    sfixn j1, ii, io;
    sfixn offset = 0;

    for (j = w - 1; j >= 0; --j) {
        j1 = j + 1;
        for (iy = 0; iy < By; ++iy) {
            ry = partialBitRev(iy, by); 
            for (ix = 0; ix < Bx; ++ix) {
                rx = partialBitRev(ix, bx);
                ii = ((iy << bx) + ix) * j1;
                io = ((rx << by) + ry) * ww + j * w;  // transposition
                for (k = 0; k <= j; ++k) {
                    dout[io + k] = din[ii + offset + k];
                }   
            }
        }
        offset += (j1 << (bx + by));
    }
}

void FreeDftChain(DftChain_t *scube) {
    int i;
    if (scube != NULL) {
        if (scube->Ses != NULL) my_free(scube->Ses);
        if (scube->Sdims != NULL) my_free(scube->Sdims);
        if (scube->roots != NULL) my_free(scube->roots);
        if (scube->Sdata != NULL) my_free(scube->Sdata);
        if (scube->SLcs != NULL) {
            for(i = 1; i < scube->w; i++){ EX_FreeOnePoly((scube->SLcs)[i]); }
            my_free(scube->SLcs);
        }
        if (scube->SPolys != NULL) {
            for(i = 1; i < scube->w; i++){ EX_FreeOnePoly((scube->SPolys)[i]); }
            my_free(scube->SPolys);
        }
        my_free(scube);
    }
}

void PrintDftChain(DftChain_t *scube) {
    int i, N;
    // int j, k;
    printf("...................................\n");
    printf("DFT subresultant chain\n");
    if (scube == NULL) { printf("NULL Dft subresultant chain!\n"); return; }
    N = scube->dim - 1;
    printf("scube->w = %d\n", scube->w); 
    printf("scube->dim = %d\n", scube->dim); 
    printf("scube->Ses = ");
    printVec(N - 1, scube->Ses);
    printf("scube->Sdims =  ");
    printVec(N + 1 , scube->Sdims);
    printf("scube->Ssize = %d\n", scube->Ssize); 
    printf("scube->Sdata = ...\n");
    //for (i = 0; i < scube->Ssize/scube->w/scube->w; ++i) {
    //    printf("subres[%d] : \n", i);
    //    for (j = 0; j < scube->w; ++j) {
    //        for (k = 0; k < scube->w; ++k) printf("%10d ", 
    //            (scube->Sdata + i * scube->w * scube->w + j * scube->w)[k]);
    //        printf("\n");
    //    }
    //}
    printf("scube->SLcs = \n");
    for (i = 0; i < scube->w; i++){
        printf("SLcs[%d] = ", i);
        printPoly(scube->SLcs[i]);
    }
    for (i = 0; i < scube->w; i++) {
        printf("SPolys[%d] = ", i);
        printPoly(scube->SPolys[i]);
    }
    printf("...................................\n");
}

//////////////////
//  CUDA SCUBE 
//////////////////

/**
 * @N, the number of variables in the input polynomials
 * @sz_p, the partial size vector of first polynomial, length N
 * @sz_q, the partial size vector of second polynomial, length N
 * @fp, the fourier prime
 */
CuDftChain_t *InitCuDftChain(sfixn N, const sfixn *sz_p, 
    const sfixn *sz_q, sfixn fp) 
{
#ifdef _cuda_modp_ 
    CuDftChain_t *scube;
    void *cudft = NULL;
    int i;
    assert(sz_p[N-1] >= sz_q[N-1]);
    scube = (CuDftChain_t *)my_malloc(sizeof(CuDftChain_t));
    scube->nvars = N;
    scube->partial_sz_p = (sfixn *)my_malloc(sizeof(sfixn) * N);
    scube->partial_sz_q = (sfixn *)my_malloc(sizeof(sfixn) * N);
    for (i = 0; i < N; ++i) {
        scube->partial_sz_p[i] = sz_p[i];
        scube->partial_sz_q[i] = sz_q[i];
    }
    cudft = init_cuda_scube(N, sz_p, sz_q, fp);
    scube->cudft_scube = cudft;

    if (cudft == NULL) {
        FreeCuDftChain(scube);
        return NULL;
    } 
    return scube;
#else 
    return NULL;
#endif
}

void FreeCuDftChain(CuDftChain_t *scube) {
    if (scube != NULL) {
        my_free(scube->partial_sz_p);
        my_free(scube->partial_sz_q);
#ifdef _cuda_modp_
        free_cuda_scube(scube->cudft_scube);
#endif
        my_free(scube);
    }
}

void PrintCuDftChain(const CuDftChain_t *scube) {
    if (scube == NULL) {
        printf("## empty cuda scube\n");
    } else {
#ifdef _cuda_modp_
        print_cuda_scube(scube->cudft_scube);
#endif
    }
}

/**
 * Initialize an SCUBE data structure from an internal chain
 *
 * @ct, chain type 
 * @chain, the internal chain
 */
SCUBE* EX_SCUBE_Init(ChainType_t ct, void *chain)
{
    if (ct == TreeChain) {
        SCUBE *scube = (SCUBE *)my_malloc(sizeof(SCUBE));
        scube->cType = ct;
        (scube->cPtr).sptPtr = (SPTreeChain_t *)chain;
        return scube;
    } else if (ct == DftChain) {
        SCUBE *scube = (SCUBE *)my_malloc(sizeof(SCUBE));
        scube->cType = ct;
        (scube->cPtr).dftPtr = (DftChain_t *)chain;
        return scube;
    } else if (ct == CuDftChain) {
        SCUBE *scube = (SCUBE *)my_malloc(sizeof(SCUBE));
        scube->cType = ct;
        (scube->cPtr).cudftPtr = (CuDftChain_t *)chain;
        return scube;
    } else {
        return NULL;
    }
}

/**
 * Free an SCUBE data structure
 */
void EX_SCUBE_Free(SCUBE *scube) {
    if (scube == NULL) return;

    if (scube->cType == TreeChain) {
        FreeSPTreeChain((scube->cPtr).sptPtr);
    } else if (scube->cType == DftChain) {
        FreeDftChain((scube->cPtr).dftPtr);
    } else if (scube->cType == CuDftChain) {
        FreeCuDftChain((scube->cPtr).cudftPtr);
    }

    my_free(scube);
}

/**
 * Print an SCUBE data structure
 */
void EX_SCUBE_Print(SCUBE *scube) {
    if (scube == NULL) printf("NULL SCUBE\n");
    switch (scube->cType) {
        case TreeChain:
            PrintSPTreeChain((scube->cPtr).sptPtr);
            break;
        case DftChain:
            PrintDftChain((scube->cPtr).dftPtr);
            break;
        case CuDftChain:
            PrintCuDftChain((scube->cPtr).cudftPtr);
            break;
        default:
            printf("Invalid chain type\n");
    }
}

////////////////////////////////////////////////////////////////////////////////
//   Construction of SCUBE
////////////////////////////////////////////////////////////////////////////////
SPTreeChain_t *SubResultantChain_SPTree(preFFTRep *f1, preFFTRep *f2, sfixn N,
    MONTP_OPT2_AS_GENE *pPtr) 
{
    int i;
    sfixn M, w, d1, d2;
    PTS_TREE* pts_tree;
    sfixn *bounds, *dims1, *dims2, Esz1, *E1, Esz2, *E2;
    SPTreeChain_t *scube;
    
    d1 = shrinkDeg(BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    d2 = shrinkDeg(BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
    assert(d1 > 0 && d2 > 0);
    if (d1 < d2) { return SubResultantChain_SPTree(f2, f1, N, pPtr); }
    
    M = N - 1;
    w = d2;
    bounds = (sfixn *)my_calloc(M+1, sizeof(sfixn));
    dims1  = (sfixn *)my_calloc(N+1, sizeof(sfixn));
    dims2  = (sfixn *)my_calloc(N+1, sizeof(sfixn));
    dims1[N] = (BUSZSI(f1, N)) + 1;
    dims2[N] = (BUSZSI(f2, N)) + 1;
    
    for(i = 1; i<= M; i++){
        // original version
        // bounds[i] = (BUSZSI(f1,i))*(BUSZSI(f2,N)) 
        //           + (BUSZSI(f2,i))*(BUSZSI(f1,N)) + 1;
        // should be tighten as
        bounds[i] = (BUSZSI(f1, i)) * d2 + (BUSZSI(f2, i)) * d1 + 1;
        dims1[i] = bounds[i];
        dims2[i] = bounds[i];
    }

    pts_tree = createGoodPtsForf1f2(N, d1, f1, d2, f2, M, bounds, 
        dims1, dims2, pPtr);

    if (DEBUG) assert(pts_tree != NULL);

    for(i=1, Esz1=1; i<=N; i++) { Esz1 *= dims1[i]; }
    for(i=1, Esz2=1; i<=N; i++) { Esz2 *= dims2[i]; }
    E1 = (sfixn *)my_calloc(Esz1, sizeof(sfixn));
    E2 = (sfixn *)my_calloc(Esz2, sizeof(sfixn));

    fastEvalMulti_test(M, dims1, E1, f1, pts_tree->trees, pPtr);
    fastEvalMulti_test(M, dims2, E2, f2, pts_tree->trees, pPtr);

    scube = InitSPTreeChain(M, bounds, w);
    scube->points_trees = pts_tree;
    my_free(bounds);

    getSubResultantChains(N, w, scube->Ssize, scube->Sdata, 
        dims1, E1, dims2, E2, pPtr);

    my_free(dims1); my_free(E1); 
    my_free(dims2); my_free(E2); 
    return scube;
}

sfixn max2(sfixn a, sfixn b) { return (a > b) ? a : b; }

/**
 * FFT based method to construct subresultant chain data structure. 
 * This method may fail due to the lack of proper primitive root of unity.
 * More precisely, one cannot find a good primtive root of unity such that
 * both initial of f1 and f2 do not vanish at its powers. 
 *
 * N should be at least 2. If failed the it returns NULL pointer.
 */
DftChain_t *SubResultantChain_FFT(preFFTRep *f1, preFFTRep *f2, sfixn N, 
    MONTP_OPT2_AS_GENE *pPtr) 
{
    sfixn i, w, d1, d2, b, goodRoots, M;
    sfixn *es, *dims1, *dims2, Esz1, *E1, Esz2, *E2, *rootsPtr, rsz = 0;
    DftChain_t *scube; 

    if (N < 2) return NULL;

    d1 = shrinkDeg(BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    d2 = shrinkDeg(BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
 
    assert(d1 > 0 && d2 > 0);
    if (d1 < d2) { return SubResultantChain_FFT(f2, f1, N, pPtr); }
    
    //////////////////////////////////////////////////////////////
    // Collect data for evaluating f1 and f2 at N - 1 variables //
    //////////////////////////////////////////////////////////////
    M = N - 1;
    w = d2;
    es = (sfixn *)my_calloc(M + 1, sizeof(sfixn));
    dims1  = (sfixn *)my_calloc(N + 1, sizeof(sfixn));
    dims2  = (sfixn *)my_calloc(N + 1, sizeof(sfixn));
    dims1[N] = (BUSZSI(f1, N)) + 1;
    dims2[N] = (BUSZSI(f2, N)) + 1;
    for (i = 1; i <= M; i++) {
        // degree bound for variable xi
        b = (BUSZSI(f1, i)) * d2 + (BUSZSI(f2, i)) * d1 + 1;
        es[i] = logceiling(b);
        dims2[i] = dims1[i] = ((sfixn)1 << es[i]);
        rsz += dims1[i];
    }
    //////////////////////////////////////////////////////////////////////////
    // Find good roots of unity, without vanishing the initials of f1 and f2.
    //
    // If we correctly build the scube then rootsPtr should NOT be freed 
    // in this function, and it will be freed by the scube struct. 
    // Otherwise, if we fail to find good roots, then it will be freed here.
    //////////////////////////////////////////////////////////////////////////
    rootsPtr = (sfixn *)my_calloc(rsz, sizeof(sfixn));
    goodRoots = findGoodRoots(N, f1, f2, es, dims1, rootsPtr, pPtr);
    if (goodRoots == -1) {
        my_free(es);
        my_free(dims1);
        my_free(dims2);
        my_free(rootsPtr);
        return NULL;
    }

    scube = InitDftChain(M, es, w);
    scube->roots = rootsPtr;
//#if defined(_cuda_modp_)
//    err = CUMODP_UNKNOWN_ERROR;
//    if (CUDA_TAG && (N == 2) && (es[1] >= 10) && 
//        can_call_subres2(max2(BUSZSI(f1, 1), BUSZSI(f2, 1)), d1, d2)) 
//    {
//        // allocate space for the result from GPU
//        B = dims1[1];
//        T = (sfixn *)my_calloc(B * w * (w + 1) / 2, sizeof(sfixn));
//        wx = MontMulMod_OPT2_AS_GENE(rootsPtr[1], 1, pPtr);
//    
//        err = cumodp_subres_chain2_fine(T, dims1[1], wx, 
//            BUSZSI(f1, 1) + 1, d1 + 1, DAT(f1), 
//            BUSZSI(f2, 1) + 1, d2 + 1, DAT(f2), pPtr->P);
//
//        if (err == CUMODP_SUCCESS) {
//            if (DEBUG) printf("fine Scube2 constructed.\n");
//            StrideTri2ListRect(es[1], w, scube->Sdata, T);
//            my_free(T);  
//            my_free(es);
//            my_free(dims1); 
//            my_free(dims2);
//            return scube;
//        } else if (err == CUMODP_SUBRES_NON_REGULAR) {
//            err = cumodp_subres_chain2_coarse(T, dims1[1], wx, 
//                BUSZSI(f1, 1) + 1, d1 + 1, DAT(f1), 
//                BUSZSI(f2, 1) + 1, d2 + 1, DAT(f2), pPtr->P);
//
//            if (err == CUMODP_SUCCESS) {
//                if (DEBUG) printf("coarse Scube2 constructed.\n");
//                ListTri2Rect(es[1], w, scube->Sdata, T);
//
//                my_free(T);  
//                my_free(es);
//                my_free(dims1); 
//                my_free(dims2);
//                return scube;
//            }
//        } else {
//            if (DEBUG) printf("cumodp scube2 failed\n");
//            my_free(T);   
//        }
//    } else if (CUDA_TAG && (N == 3) && (es[1] >= 8) && (es[2] >= 8) && 
//        can_call_subres3(max2(BUSZSI(f1, 1), BUSZSI(f2, 1)), 
//                         max2(BUSZSI(f1, 2), BUSZSI(f2, 2)), d1, d2)) 
//    {
//        B = ((sfixn)1 << (es[1] + es[2]));
//        T = (sfixn *)my_calloc(B * w * (w + 1) / 2, sizeof(sfixn));
//        wx = MontMulMod_OPT2_AS_GENE(rootsPtr[1], 1, pPtr);
//        wy = MontMulMod_OPT2_AS_GENE(rootsPtr[dims1[1] + 1], 1, pPtr);
//
//        err = cumodp_subres_chain3_fine(T, dims1[1], dims1[2], wx, wy,  
//            BUSZSI(f1, 1) + 1, BUSZSI(f1, 2) + 1, d1 + 1, DAT(f1), 
//            BUSZSI(f2, 1) + 1, BUSZSI(f2, 2) + 1, d2 + 1, DAT(f2), pPtr->P);
//
//        if (err == CUMODP_SUCCESS) {
//            if (DEBUG) printf("fine Scube3 constructed.\n");
//            StrideTri3ListRect(es[1], es[2], w, scube->Sdata, T);
//            my_free(T);  
//            my_free(es);
//            my_free(dims1); 
//            my_free(dims2);
//            return scube;
//        } else if (err == CUMODP_SUBRES_NON_REGULAR) {
//            err = cumodp_subres_chain3_coarse(T, dims1[1], dims1[2], 
//                wx, wy, BUSZSI(f1, 1) + 1, BUSZSI(f1, 2) + 1, d1 + 1, DAT(f1), 
//                BUSZSI(f2, 1) + 1, BUSZSI(f2, 2) + 1, d2 + 1, DAT(f2), pPtr->P);
//
//            if (err == CUMODP_SUCCESS) {
//                if (DEBUG) printf("coarse Scube3 constructed.\n");
//                ListTri3Rect(es[1], es[2], w, scube->Sdata, T);
//                my_free(T);  
//                my_free(es);
//                my_free(dims1); 
//                my_free(dims2);
//                return scube;
//            }
//        } else {
//            if (DEBUG) printf("cumodp scube3 failed\n");
//            my_free(T);   
//        }
//    }
//#endif
    /////////////////////////
    // Evaluating f1 and f2
    /////////////////////////
    for(i = 1, Esz1 = 1; i <= N; i++) { Esz1 *= dims1[i]; }
    for(i = 1, Esz2 = 1; i <= N; i++) { Esz2 *= dims2[i]; }
    E1 = (sfixn *) my_calloc(Esz1, sizeof(sfixn));
    E2 = (sfixn *) my_calloc(Esz2, sizeof(sfixn));
    fastDftMulti_test(M, es, dims1, E1, f1, rootsPtr, pPtr);
    fastDftMulti_test(M, es, dims2, E2, f2, rootsPtr, pPtr);

    //////////////////////////
    // Constructing the SCUBE
    //////////////////////////
    getSubResultantChains(N, w, scube->Ssize, scube->Sdata, 
        dims1, E1, dims2, E2, pPtr);
   
    my_free(es);
    my_free(dims1); 
    my_free(dims2);
    my_free(E1); 
    my_free(E2); 
    return scube;
}


/**
 * Construct a cuda scube from cumodp library.
 */
CuDftChain_t *SubResultantChain_CUFFT(preFFTRep *f1, preFFTRep *f2, 
    sfixn N, sfixn fp) 
{
#ifdef _cuda_modp_
    CuDftChain_t *scube;
    sfixn *sz_p, *sz_q, d1, d2, i;
    cumodp_err err;

    // only handle bivariate or trivariate case, to relax
    if (N != 2 && N != 3) return NULL;

    d1 = shrinkDeg(BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    d2 = shrinkDeg(BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
    assert(d1 > 0 && d2 > 0);
    if (d1 < d2) { return SubResultantChain_CUFFT(f2, f1, N, fp); }
    
    sz_p = (sfixn *)my_malloc(sizeof(sfixn)*N);
    sz_q = (sfixn *)my_malloc(sizeof(sfixn)*N);
    for (i = 0; i < N; ++i) {
        // the 0-th entry is not used in f1 or f2
        sz_p[i] = BUSZSI(f1, i + 1) + 1;
        sz_q[i] = BUSZSI(f2, i + 1) + 1;
    }

    // iniatialize the scube
    scube = InitCuDftChain(N, sz_p, sz_q, fp);
    if (scube == NULL) {
        if (DEBUG) printf("fail to build cuda scube\n");
        my_free(sz_p);
        my_free(sz_q);
        return NULL;
    }

    // build the scube
    err = build_cuda_scube(scube->cudft_scube, sz_p, DAT(f1), sz_q, DAT(f2));
    
    if (err != CUMODP_SUCCESS) {
        if (DEBUG) printf("CUDA FFT based method failed\n");
        my_free(sz_p);
        my_free(sz_q);
        FreeCuDftChain(scube);
        return NULL;
    }

    return scube;
#else
    return NULL;
#endif
}

/**
 * Build preFFTRep polynomials from a vector of coefficients.
 * for univariate or bivariate input
 * 
 * @coeff, coefficient vector
 * @nx, the partial size in x of the coefficient vector
 * @ny, the partial size in y of the coefficient vector
 *
 * 
 * For example, 
 *
 * [0, 1, 2, 3, 4, 5, 0, 0, 0] ==> a degree 5 polynomial
 *
 * [0, 1, 2, 0,
 *  3, 4, 5, 0,  
 *  6, 7, 0, 0,  
 *  0, 0, 0, 0]
 *
 * ==> a bivariate polynomial with partial degree 2 in x,
 *     partial degree 2 in y.
 **/
preFFTRep* coeff_vec_to_unipoly(sfixn nx, const sfixn *coeff) 
{
    sfixn dx = nx - 1;
    while (dx >= 0 && coeff[dx] == 0) { --dx; }
    if (dx < 0) return CreateZeroPoly();
    return CreateUniPoly(dx, coeff);
}

static int is_zero_vector(sfixn n, const sfixn *coeff) {
    sfixn i;
    for (i = 0; i < n; ++i) if (coeff[i]) return 0;   
    return 1;
}

/**
 * check if the k-th column of a matrix containing zeros only
 * the number of rows is ny.
 */
static int is_zero_column(sfixn k, sfixn nx, sfixn ny, const sfixn *coeff) 
{
    sfixn j;
    for (j = 0; j < ny; ++j) {
        if (coeff[j * nx + k]) return 0;
    }
    return 1;
}

preFFTRep* coeff_vec_to_bipoly(sfixn nx, sfixn ny, const sfixn *coeff) 
{
    preFFTRep *poly;
    sfixn dx = nx - 1;
    sfixn dy = ny - 1;
    sfixn i, j;
    sfixn dgs[3] = {0, 0, 0};

    for (; dy >= 0; --dy) { 
        if (!is_zero_vector(nx, coeff + dy * nx)) break; 
    }

    if (dy < 0) return CreateZeroPoly();

    for (; dx >= 0; --dx) {
        if (!is_zero_column(dx, nx, ny, coeff)) break;   
    }

    // dx >= 0 since coeff contains nozero elements.
    // if both dx and by are zero, then it is a non-zero constant.
    if (dx == 0 && dy == 0) return CreateConsPoly(coeff[0]);

    // general case
    dgs[1] = dx, dgs[2] = dy;
    poly = EX_InitOnePoly(2, dgs);
    
    // fill the coefficient
    for (j = 0; j <= dy; ++j) {
        for (i = 0; i <= dx; ++i) {
            poly->data[j * (dx + 1) + i] = coeff[j * nx + i];
        }
    }
    return poly;
}

preFFTRep* ResultantFromChain_CUFFT(CuDftChain_t *scube) {
#ifdef _cuda_modp_
    const sfixn *coeff; // coefficient vector
    sfixn nx, ny;       // partial size in each variable
    if (scube->nvars == 2) {
        coeff = interp_subres_coeff2(&nx, scube->cudft_scube, 0, 0);
        return coeff_vec_to_unipoly(nx, coeff);
    } else if (scube->nvars == 3) {
        coeff = interp_subres_coeff3(&nx, &ny, scube->cudft_scube, 0, 0);
        return coeff_vec_to_bipoly(nx, ny, coeff);
    } 
#endif
    return NULL;
}

/**
 * Wrapper function for computer subresultant chain. 
 *
 * Assuming that the main degree of f1 and f2 are positive!!
 */
SCUBE* EX_SubResultantChain(preFFTRep *f1, preFFTRep *f2, sfixn N, 
    MONTP_OPT2_AS_GENE *pPtr)
{
    SPTreeChain_t *sc = SubResultantChain_SPTree(f1, f2, N, pPtr);
    return EX_SCUBE_Init(TreeChain, sc);
}

SCUBE* EX_SubResultantChainOpt(preFFTRep *f1, preFFTRep *f2, sfixn N, 
    MONTP_OPT2_AS_GENE *pPtr)
{
    SPTreeChain_t *sc = NULL;
    DftChain_t    *dc = NULL;
    CuDftChain_t  *cdc = NULL;

    // cuda fft based
    if (CUDA_TAG) {
        cdc = SubResultantChain_CUFFT(f1, f2, N, pPtr->P);
        if (cdc != NULL) {
            if (DEBUG) PrintCuDftChain(cdc);
            return EX_SCUBE_Init(CuDftChain, cdc);
        }
    }  

    // modpn fft based
    dc = SubResultantChain_FFT(f1, f2, N, pPtr);
    if (dc != NULL) {
        return EX_SCUBE_Init(DftChain, dc);
    } else {
        // reset PrimeError since FFT based method failed
        PrimeError = 0;
        sc = SubResultantChain_SPTree(f1, f2, N, pPtr);
        return EX_SCUBE_Init(TreeChain, sc);
    }
}

/**
 * Compute the resultant from the subresultant chain, subproduct tree.
 */
preFFTRep* ResultantFromChain_SPTree(SPTreeChain_t *scube, 
    MONTP_OPT2_AS_GENE *pPtr) 
{
    sfixn tmp;
    preFFTRep *res, *finalRes;
    
    tmp = scube->Sdims[(scube->dim)-1];
    scube->Sdims[(scube->dim)-1] = 1;

    res = interpIthDthSlice(0, 0, (scube->dim)-1, (scube->dim)-2, scube->w, 
        ((scube->Ssize)/(scube->w))/(scube->w), scube->Sdims, scube->Ssize, 
        scube->Sdata, scube->points_trees, pPtr); 

    scube->Sdims[(scube->dim)-1] = tmp;
    finalRes = EX_NormalizePoly(res);
    EX_FreeOnePoly(res);
    return finalRes;
}

/**
 * Compute the resultant from the subresultant chain, FFT.
 */
preFFTRep* ResultantFromChain_FFT(DftChain_t *scube, MONTP_OPT2_AS_GENE *pPtr) 
{
    preFFTRep *res, *finalRes;
    sfixn N = scube->dim - 1, M = N - 1, w = scube->w;
    sfixn i, subslicesz = 1, tmp;

    for (i = 1; i <= M; ++i) subslicesz <<= scube->Ses[i];
    // For some technique reason, reset Sdims[N] to 1
    tmp = scube->Sdims[(scube->dim)-1];
    scube->Sdims[(scube->dim)-1] = 1;
    res = interpIthDthSliceDFT(0, 0, N, M, w, subslicesz, scube->Ses, 
        scube->Sdims, scube->Ssize, scube->Sdata, scube->roots, pPtr);
    // restore the value of Sdims[N]
    scube->Sdims[(scube->dim)-1] = tmp;
    
    finalRes = EX_NormalizePoly(res);
    EX_FreeOnePoly(res);
    return finalRes;
}
   
/**
 * Wapper function for computing resultant from the subresultant chain.
 */
preFFTRep* EX_ResultantFromChain(SCUBE *scube, MONTP_OPT2_AS_GENE *pPtr) {
    switch (scube->cType) {
    case TreeChain:
        if (DEBUG) printf("##using tree\n");
        return ResultantFromChain_SPTree((scube->cPtr).sptPtr, pPtr);
    case DftChain:
        if (DEBUG) printf("##using dft\n");
        return ResultantFromChain_FFT((scube->cPtr).dftPtr, pPtr);    
    case CuDftChain:
        if (DEBUG) printf("##using cudft\n");
        return ResultantFromChain_CUFFT((scube->cPtr).cudftPtr);
    default:
        fprintf(stderr, "unexpected subresultant chain type!\n");
        exit(1);
    }
}

preFFTRep *EX_Resultant_Multi(preFFTRep *f1, preFFTRep *f2, sfixn N,
    MONTP_OPT2_AS_GENE *pPtr)
{ 
    SCUBE *scube;
    preFFTRep *res;
    scube = EX_SubResultantChainOpt(f1, f2, N, pPtr);
    res = EX_ResultantFromChain(scube, pPtr);
    EX_SCUBE_Free(scube);
    return res;
}

///////////////////////////////////////////////////////////////////////////////
//  Iterated resultant computation
///////////////////////////////////////////////////////////////////////////////

sfixn iteratedResultant_zerodim(preFFTRep *poly, TriSet *ts, 
    MONTP_OPT2_AS_GENE *pPtr)
{ 
    sfixn  N, M;
    preFFTRep * newpoly, *tmpnewpoly, *tmpres, *tspoly, *res;
    int i, invertibility;
    TriRevInvSet * tris;
    sfixn *dgs, finalRes, dgP, dgQ;

    N = N(ts);
    M = N(poly);
    assert(M<=N);

	signal(SIGINT,catch_intr);
    invertibility=MonicizeTriSet_1(N, ts, pPtr);
        
    if(invertibility==-1){
        printf("Error: input triangular set can not be normalized!\n");
        fflush(stdout);
        Interrupted=1;
        return -1;
	}

    dgs=(sfixn *)my_calloc(N+1, sizeof(sfixn));
    for(i=1; i<=N; i++){
        dgs[i]=BDSI(ts,i)+1;
        if ((i<=M) && (dgs[i]<BUSZSI(poly, i))) dgs[i]=BUSZSI(poly, i);
        dgs[i]<<=1;
    }

    tris = EX_initTriRevInvSet(dgs, N, ts);
    getRevInvTiSet(dgs, N, tris, ts, pPtr);
    my_free(dgs);
	tmpnewpoly = EX_NormalForm(M, poly, ts, tris, pPtr);
    newpoly=EX_NormalizePoly(tmpnewpoly);

    if ((Interrupted==1)||(PrimeError==1)) {
        EX_FreeOnePoly(tmpnewpoly);
        EX_freeTriRevInvSet(tris);
        return -1; 
    }

    EX_FreeOnePoly(tmpnewpoly);
	EX_freeTriRevInvSet(tris);
    M = N(newpoly);

    while (constantPolyp(newpoly)== 0) {
        tspoly = ELEMI(ts, M);
        if (M==1){
            dgP=shrinkDegUni(BUSZSI(newpoly, 1), DAT(newpoly));
            dgQ=shrinkDegUni(BUSZSI(tspoly, 1), DAT(tspoly));
		    finalRes = EX_Resultant_Uni(dgP, DAT(newpoly), dgQ, DAT(tspoly), pPtr);
            EX_FreeOnePoly(newpoly);
            return finalRes;
        } else {
            if(Interrupted==1) { EX_FreeOnePoly(newpoly); return 0; }
            tmpres = EX_Resultant_Multi(newpoly, tspoly, M, pPtr);
            EX_FreeOnePoly(newpoly);
            res=EX_NormalizePoly(tmpres);
            EX_FreeOnePoly(tmpres);
            M = N(res);
            if(M==1) {
                tspoly = ELEMI(ts, M);
                dgP=shrinkDegUni(BUSZSI(res, 1), DAT(res));
                dgQ=shrinkDegUni(BUSZSI(tspoly, 1), DAT(tspoly));
		        finalRes = EX_Resultant_Uni(dgP, DAT(res), dgQ, DAT(tspoly), pPtr);
                EX_FreeOnePoly(res);
                return finalRes;
	        }

	        dgs=(sfixn *)my_calloc(M+1, sizeof(sfixn));
            for(i=1; i<=M; i++){
                dgs[i]=BDSI(ts,i)+1;
                if (dgs[i]<BUSZSI(res, i)) dgs[i]=BUSZSI(res, i);
                dgs[i]<<=1;
            }

            tris = EX_initTriRevInvSet(dgs, M, ts);
            getRevInvTiSet(dgs, M, tris, ts, pPtr);
            my_free(dgs);
            newpoly = EX_NormalForm(M, res, ts, tris, pPtr);

            if ((Interrupted==1)||(PrimeError==1)) {
  	            EX_freeTriRevInvSet(tris);
                EX_FreeOnePoly(newpoly);
                EX_FreeOnePoly(res);
                return -1; 
            }
  	        EX_freeTriRevInvSet(tris);
            EX_FreeOnePoly(res);
        }
    }

    // return a constant resultant.
    finalRes = DAT(newpoly)[0];
    EX_FreeOnePoly(newpoly);
    if ((Interrupted==1)||(PrimeError==1)) { return -1; }
    return finalRes;
}

InterpRFRPreST *EX_InterpRFRPreST_Init(sfixn no) {  
    InterpRFRPreST *ST;
    ST = (InterpRFRPreST *)my_malloc(sizeof(InterpRFRPreST));
    ST->no = no;
    ST->tree = 0;
    ST->points = (sfixn *) my_calloc(ST->no, sizeof(sfixn));
    ST->values = (sfixn *)my_calloc(ST->no, sizeof(sfixn));
    ST->chains = (TriSet **)my_calloc(ST->no, sizeof(TriSet *));
    ST->polys = (preFFTRep **)my_calloc(ST->no, sizeof(preFFTRep *));  
    return ST;
}

void EX_InterpRFRPreST_Free(InterpRFRPreST *ST){  
    int i;
    if (ST) {
        if (ST->tree)  subProdTreeFree(ST->tree);
        if (ST->points) my_free(ST->points);
        if (ST->values) my_free(ST->values);
        if (ST->chains) {
            for(i=0; i<ST->no; i++){ EX_freeTriSet((ST->chains)[i]);}
            my_free(ST->chains);
        }
        if (ST->polys){
            for(i=0; i<ST->no; i++){ EX_FreeOnePoly((ST->polys)[i]);}
            my_free(ST->polys);
        }
        my_free(ST);
    }
}

// Suppose N(newpoly) = 0 is the constant case!
void evalPolyAtPt(preFFTRep *newpoly, preFFTRep *poly, sfixn freeVarNo, 
    sfixn pt, MONTP_OPT2_AS_GENE *pPtr)
{
    sfixn chunk, offset1, offset2, offset3;
    sfixn i, j, co;
    assert(N(newpoly) == N(poly) -1);

    if(N(poly)==freeVarNo){
        chunk = SIZ(poly);
    } else {
        chunk = CUMI(poly, freeVarNo+1);
    }
    offset1 = 0; offset2 = 0; offset3 = 0; 
  
    for (offset1=0, i=0; offset1<SIZ(poly); offset1+=chunk, i++)
    {
        for (offset2=0, j=0; offset2<chunk; offset2+=CUMI(poly, freeVarNo), j++)
        {
            co = PowerMod(pt, j, pPtr->P);
            coMulAddVec(co, CUMI(poly, freeVarNo)-1, DAT(newpoly)+offset3, 
                DAT(poly)+offset1+offset2, pPtr);
        }
        offset3+=CUMI(poly, freeVarNo);
    }
}

// the input poly will drop 1 dimension.
preFFTRep * EX_evalPolyAtPt(preFFTRep *poly , sfixn freeVarNo, sfixn pt, 
    MONTP_OPT2_AS_GENE *pPtr)
{
    preFFTRep *newpoly;
    sfixn *dgs;
    int i;
    if (freeVarNo > N(poly)){ return  EX_CopyOnePoly(poly); }
    // to handle univariate case!!! Later.
    dgs=(sfixn *)my_calloc(N(poly), sizeof(sfixn));
    for(i=1; i<freeVarNo; i++){ dgs[i] = BUSZSI(poly, i); }
    for(i=freeVarNo+1; i<=N(poly); i++){ dgs[i-1] = BUSZSI(poly, i); }
    newpoly = EX_InitOnePoly(N(poly)-1, dgs);
    evalPolyAtPt(newpoly, poly, freeVarNo, pt, pPtr);
    my_free(dgs);
    return newpoly;
}


// evaluate a one-dim chain to a zero-dim chain.
// dests = eval (srcts, vars[freeVarNo] = pt).
TriSet *EX_evalChain(TriSet *srcts, sfixn freeVarNo, sfixn pt, 
    MONTP_OPT2_AS_GENE *pPtr)
{
    int i;
    TriSet *dests=(TriSet *)my_malloc(sizeof(TriSet));
    N(dests)=N(srcts) - 1;
    ELEM(dests)=(preFFTRep **)my_calloc((N(dests)+1),sizeof(preFFTRep *) );
    for(i=1; i<=N(dests); i++){
        if(i<freeVarNo){ ELEMI(dests,i)= EX_CopyOnePoly( ELEMI(srcts, i)); }
        if (i>=freeVarNo) {
            ELEMI(dests, i) = EX_evalPolyAtPt(ELEMI(srcts, i+1), freeVarNo, pt, pPtr);
        }  
    } 

    BDS(dests)=(sfixn *) my_calloc((N(dests)+1),sizeof(sfixn));
    for(i=1; i<=N(dests); i++){ BDSI(dests, i)=BUSZSI(ELEMI(dests, i),i) -1;}

    // to computer the new BDS
    return dests;
}


// poly is a multivariate polynomial.
// ts is a triangular set.
// bound is number of points for evaluation.
// freeVarNo is the variable no if free variable of ts.
InterpRFRPreST *InterpRFRUniPre (preFFTRep *poly, TriSet *ts, sfixn bound, 
    sfixn freeVarNo, MONTP_OPT2_AS_GENE *pPtr)
{
    InterpRFRPreST *ST;
    TriSet *rc;
    preFFTRep *initial, *h;
    sfixn *items;
    sfixn pt=0;
    int i, j;

    ST = EX_InterpRFRPreST_Init(bound);
    items = (sfixn *)my_calloc(2*(ST->no), sizeof(sfixn));
    j=0;
    initial = EX_getInitial(poly);
    for(i=0; i<2*(ST->no); i+=2){
        pt = pt + 1;
        rc = EX_evalChain(ts, freeVarNo, pt, pPtr);
        h = EX_evalPolyAtPt(initial, freeVarNo, pt, pPtr);
        while ((rc == NULL) || (zeroPolyp(h))){
	        EX_FreeOnePoly(h);
            pt = pt + 1;
            rc = EX_evalChain(ts, freeVarNo, pt, pPtr);
            h = EX_evalPolyAtPt(initial, freeVarNo, pt, pPtr);
        }
        EX_FreeOnePoly(h);
        items[i] = (pPtr->P) - pt;
        items[i+1]=1;
        ST->points[j] = items[i];
        ST->chains[j] = rc;
        ST->polys[j] = EX_evalPolyAtPt(poly, freeVarNo, pt, pPtr);
        j++;
        if(pt >= pPtr->P){
	        PrimeError=1;
            EX_FreeOnePoly(initial);
            ST->tree=NULL;
            return ST;
        }
    }
  
    EX_FreeOnePoly(initial);
    ST->tree = subProdTreeCre(ST->no, 2, items, pPtr->P);
    my_free(items);
    return ST;
}

InterpRFRST *EX_InterpRFRST_Init(sfixn bound) 
{
    InterpRFRST *RFRST;
    RFRST = (InterpRFRST *) my_calloc(1,sizeof(InterpRFRST));
    RFRST->bound = bound;
    RFRST->Num = (sfixn *) my_calloc(RFRST->bound,sizeof(sfixn));
    RFRST->Den = (sfixn *) my_calloc(RFRST->bound,sizeof(sfixn));   
    return RFRST;
}

InterpRFRST *InterpRFRUni(InterpRFRPreST *PreST, MONTP_OPT2_AS_GENE *pPtr) 
{ 
    InterpRFRST *fraction;
    sfixn *F, FDg, MDg, *FDgAddr=&FDg, res;
    fraction = EX_InterpRFRST_Init(PreST->no);
    F=fastInterp(FDgAddr, PreST->no, PreST->points, PreST->tree, PreST->values, pPtr->P);
    MDg=(PreST->tree->W)[(PreST->tree->h)+1];
    MDg=shrinkDegUni(MDg-1, (PreST->tree->data));
    (fraction->degDen)=(PreST->no)-1;
    res = ExGcd_Uni_RFR(PreST->no, fraction->Den, &(fraction->degDen), 
        fraction->Num, &(fraction->degNum), PreST->tree->data, MDg, F, FDg, pPtr);

    my_free(F);
    if(res==-1) { printf("ExGcd_Uni_RFR failed!\n"); fflush(stdout); Interrupted=1; }
    return fraction;
}

// Input: 'poly' is a multivariate polynomial.
//        'ts' is a 1 dimensional regular chain. the freeVarNo-th polyomials is
//         missing from the chain.
//        'bound' is the product of input polynomials' total degrees.
//        'freeVarNo' is the index of free variable wrt. 'ts'.
// Output: a univeristate polynoila -- (*resDgAddr, num).

sfixn *iteratedResultant_onedim(sfixn *resDgAddr, preFFTRep *poly, TriSet *ts, 
    sfixn bound, sfixn freeVarNo, MONTP_OPT2_AS_GENE * pPtr)
{
    InterpRFRPreST *PreST;
    InterpRFRST *fraction;
    int i;
    sfixn *res;

    signal(SIGINT,catch_intr);
    bound =  2 * (2 * bound + 1);
    PreST = InterpRFRUniPre(poly, ts, bound, freeVarNo, pPtr);
    for (i=0; i < bound; i++){

    if ((Interrupted==1)||(PrimeError==1)) { 
        EX_InterpRFRPreST_Free(PreST); 
        return NULL; 
    }

    (PreST->values)[i] = iteratedResultant_zerodim((PreST->polys)[i], 
        (PreST->chains)[i], pPtr); }
    
    fraction = InterpRFRUni(PreST, pPtr);
    *resDgAddr = fraction->degNum;
    res = fraction->Num;

    if(fraction){
        if(fraction->Den) my_free(fraction->Den);
        my_free(fraction);
    }

    EX_InterpRFRPreST_Free(PreST);
    return res;
}
///////////////////////// END OF FILE /////////////////////////////////////////

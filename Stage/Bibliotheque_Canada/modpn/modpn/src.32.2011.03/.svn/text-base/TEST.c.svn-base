/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */


#include "Types.h"
#include "generalFuncs.h"
#include "MultiDFFT.h"
#include "MPMMTS.h"
#include "FINTERP.h"
#include "Factorization.h"
#include "MapleCConverter.h"
#include "solve2.h"
#include <math.h>
#include "solve2.h"
#include <time.h>
#include "HGCD.h"

extern sfixn FP[];
extern sfixn CUDA_TAG;

void resultant3_tst(int argc, char **argv) {
    sfixn p = 919601153;
    //sfixn p = 257;
    sfixn N = 3, d1 = 6, d2 = 6, d3 = 6;
    sfixn dgs1[] = {0, 6, 6, 6};
    sfixn dgs2[] = {0, 6, 6, 6};
    preFFTRep *F1;
    preFFTRep *F2;
    preFFTRep *result = NULL;
    MONTP_OPT2_AS_GENE prime;
    MONTP_OPT2_AS_GENE *pPtr = &prime;
    double time0, time1;
    clock_t t0, t1;

    EX_MontP_Init_OPT2_AS_GENE(pPtr, p);
    if (argc == 2) {
        d3 = d2 = d1 = atoi(argv[1]);
        dgs1[1] = dgs2[1] = d1;
        dgs1[2] = dgs2[2] = d2;
        dgs1[3] = dgs2[3] = d3;
    } 

    F1 = EX_randomPoly(N, dgs1, p);
    F2 = EX_randomPoly(N, dgs2, p);
    
    CUDA_TAG = 1;
    t0 = clock();
    result = EX_Resultant_Multi(F1, F2, N, pPtr);
    t0 = clock() - t0;
    //EX_Poly_Print(F1);
    //EX_Poly_Print(F2);
    //EX_Poly_Print(result);
    EX_FreeOnePoly(result);
    time0 = (double)t0 / (double)CLOCKS_PER_SEC;

    CUDA_TAG = 0;
    t1 = clock();
    result = EX_Resultant_Multi(F1, F2, N, pPtr);
    t1 = clock() - t1;
    EX_FreeOnePoly(result);
    time1 = (double)t1 / (double)CLOCKS_PER_SEC;
    
    printf("d1 = %d, d2 = %d, d3 = %d, GPU = %.3f, CPU = %.3f\n", d1, d2, d3, time0, time1);
}

void resultant2_tst(int argc, char **argv) {
    sfixn p = 919601153;
    //sfixn p = 257;
    sfixn N = 2, d1 = 10, d2 = 10;
    sfixn dgs1[] = {0, 10, 10};
    sfixn dgs2[] = {0, 10, 10};
    preFFTRep *F1;
    preFFTRep *F2;
    preFFTRep *result = NULL;
    MONTP_OPT2_AS_GENE prime;
    MONTP_OPT2_AS_GENE *pPtr = &prime;
    double time0, time1;
    clock_t t0, t1;

    EX_MontP_Init_OPT2_AS_GENE(pPtr, p);
    if (argc == 2) {
        d1 = atoi(argv[1]);
        d2 = d1; 
        dgs1[1] = dgs1[2] = d1;
        dgs2[1] = dgs2[2] = d2;
    }

    F1 = EX_randomPoly(N, dgs1, p);
    F2 = EX_randomPoly(N, dgs2, p);
    
    CUDA_TAG = 1;
    t0 = clock();
    result = EX_Resultant_Multi(F1, F2, N, pPtr);
    t0 = clock() - t0;
    //EX_Poly_Print(F1);
    //EX_Poly_Print(F2);
    //EX_Poly_Print(result);
    EX_FreeOnePoly(result);
    time0 = (double)t0 / (double)CLOCKS_PER_SEC;

    CUDA_TAG = 0;
    t1 = clock();
    result = EX_Resultant_Multi(F1, F2, N, pPtr);
    t1 = clock() - t1;
    EX_FreeOnePoly(result);
    time1 = (double)t1 / (double)CLOCKS_PER_SEC;
    
    printf("d1 = %d, d2 = %d, GPU = %.3f, CPU = %.3f\n", d1, d2, time0, time1);

}

void uniquo_tst(int argc, char ** argv){
    sfixn *A, dA = 9, *B, dB = 9, *Q, dQ;
    sfixn i;
    sfixn p = 919601153;
    //sfixn p = 257;
    MONTP_OPT2_AS_GENE prime;
    MONTP_OPT2_AS_GENE *pPtr = &prime;
    clock_t t0, t1;
    double time0, time1;

    EX_MontP_Init_OPT2_AS_GENE(pPtr, p);
    if (argc == 2) {
        dB = atoi(argv[1]);
        dA = dB; 
    } else if (argc == 3) {
        dA = atoi(argv[1]);
        dB = atoi(argv[2]);
    }

    A = (sfixn *)my_malloc(sizeof(sfixn)*(dA + 1));
    B = (sfixn *)my_malloc(sizeof(sfixn)*(dB + 1));
    for (i = 0; i <= dA; ++i) A[i] = rand() % p;
    for (i = 0; i <= dB; ++i) B[i] = rand() % p;
    B[dB] = 1;
    
    dQ = dA - dB;
    Q = (sfixn *)my_malloc(sizeof(sfixn)*(dQ + 1));
    
    t0 = clock();
    fastQuo(dQ, Q, dA, A, dB, B, pPtr);
    t0 = clock() - t0;
    dQ = shrinkDegUni(dQ, Q);
    time0 = (double)t0 / (double)CLOCKS_PER_SEC;

    dQ = dA - dB;
    t1 = clock(); 
    plainQuo(dQ, Q, dA, A, dB, B, pPtr);
    t1 = clock() - t1;
    time1 = (double)t1 / (double)CLOCKS_PER_SEC;
    printf("dA = %d, dB = %d, fastQuo = %.3f, plainQuo = %.3f\n", dA, dB, time0, time1);
    //EX_printPolyUni(dA, A, 'x');
    //EX_printPolyUni(dB, B, 'x');
    //EX_printPolyUni(dQ, Q, 'x');

    my_free(A);
    my_free(B);
    my_free(Q);
}

void HGCD_tst(int argc, char** argv) 
{
    //sfixn AA[] = { 22, 199, 234, 97, 205, 19, 200, 144, 13, 177};
    //sfixn BB[] = {112, 68, 245, 142, 35, 130, 124, 240, 174, 26};
    sfixn *A, dA = 9, *B, dB = 9, *G, dG;
    sfixn i;
    sfixn p = 919601153;
    //sfixn p = 257;
    MONTP_OPT2_AS_GENE prime;
    MONTP_OPT2_AS_GENE *pPtr = &prime;
    clock_t t0, t1;
    double time0, time1;

    EX_MontP_Init_OPT2_AS_GENE(pPtr, p);
    if (argc == 2) {
        dA = dB = atoi(argv[1]);
    } else if (argc == 3) {
        dA = atoi(argv[1]);
        dB = atoi(argv[2]);
    }

    A = (sfixn *)my_malloc(sizeof(sfixn)*(dA + 1));
    B = (sfixn *)my_malloc(sizeof(sfixn)*(dB + 1));
    for (i = 0; i <= dA; ++i) A[i] = rand() % p;
    for (i = 0; i <= dB; ++i) B[i] = rand() % p;
    //for (i = 0; i <= dA; ++i) A[i] = AA[i];
    //for (i = 0; i <= dB; ++i) B[i] = BB[i];
    //EX_printPolyUni(dA, A, 'x');
    //EX_printPolyUni(dB, B, 'x');

    t0 = clock();
    G = HalfGCD(&dG, A, dA, B, dB, pPtr);
    t0 = clock() - t0;
    //EX_printPolyUni(dG, G, 'x');
    my_free(G);
    time0 = (double)t0 / (double)CLOCKS_PER_SEC;
      
    t1 = clock();
    G = EX_GCD_UNI(&dG, A, dA, B, dB, pPtr);
    t1 = clock() - t1;
    //EX_printPolyUni(dG, G, 'x');
    my_free(G);
    time1 = (double)t1 / (double)CLOCKS_PER_SEC;
    
    printf("dA = %d, dB = %d, HGCD = %.3f, PlainGCD = %.3f\n", dA, dB, time0, time1);

    my_free(A);
    my_free(B);
}


// n := 5;
// unknowns := [x, y];
// sys := [ x^(2*n) + a*y^n - y, y^(2*n) +b * x^n - x ];
void hass_solve2(int argc, char** argv) 
{
    sfixn i, j, a = 1, b = 1, n = 60, p = 919601153;
    MONTP_OPT2_AS_GENE prime;
    MONTP_OPT2_AS_GENE *pPtr = &prime;
    sfixn dgs1[3] = {0, 120, 60};
    sfixn dgs2[3] = {0, 60, 120};
    LinkedQueue *resQueue = NULL;
    clock_t t0, t1;
    double time0, time1;

    EX_MontP_Init_OPT2_AS_GENE(pPtr, p);
    if (argc == 2) {
        n = atoi(argv[1]);
        dgs1[1] = dgs2[2] = 2 * n;
        dgs1[2] = dgs2[1] = n;
    } else if (argc == 4) {
        n = atoi(argv[1]);
        dgs1[1] = dgs2[2] = 2 * n;
        dgs1[2] = dgs2[1] = n;
        a = atoi(argv[2]);
        b = atoi(argv[3]);
    }

    preFFTRep *F1 = EX_InitOnePoly(2, dgs1);
    preFFTRep *F2 = EX_InitOnePoly(2, dgs2);

    (F1->data)[2*n] = 1;
    (F1->data)[2*n+1] = p - 1;
    (F1->data)[(2*n+1) * n] = a;

    (F2->data)[1] = p - 1;
    (F2->data)[n] = b;
    (F2->data)[(n + 1) * 2 * n] = 1;

    preFFTRep *zerop = CreateZeroPoly();

    t0 = clock();
    resQueue = modular_solve2_select(0, F2, F1, zerop, pPtr);
    t0 = clock() - t0;
    time0 = (double)t0 / (double)CLOCKS_PER_SEC;
    //EX_LinkedQueue_Print(resQueue, EX_RegularChain2_Print);
    EX_LinkedQueue_Free(resQueue, EX_RegularChain2_Free);
    printf(" total = %.3f  ||  ", time0);

    //t1 = clock();
    //resQueue = modular_solve2(1, F2, F1, zerop, pPtr);
    //t1 = clock() - t1;
    //time1 = (double)t1 / (double)CLOCKS_PER_SEC;
    ////EX_LinkedQueue_Print(resQueue, EX_RegularChain2_Print);
    //EX_LinkedQueue_Free(resQueue, EX_RegularChain2_Free);
    //printf(" total = %.3f\n", time1);

    EX_FreeOnePoly(F1);
    EX_FreeOnePoly(F2);
    EX_FreeOnePoly(zerop);
}

void random_dense_solve2(int argc, char** argv) 
{
    sfixn p = 469762049;
    MONTP_OPT2_AS_GENE prime;
    MONTP_OPT2_AS_GENE *pPtr = &prime;
    sfixn dgs1[3] = {0, 60, 60};
    sfixn dgs2[3] = {0, 60, 60};
    LinkedQueue *resQueue = NULL;
    clock_t t0, t1;
    double time0, time1;

    EX_MontP_Init_OPT2_AS_GENE(pPtr, p);
    if (argc == 2) {
        dgs1[1] = dgs1[2] = dgs2[1] = dgs2[2] = atoi(argv[1]);
    } else if (argc == 5) {
        dgs1[1] = atoi(argv[1]);
        dgs1[2] = atoi(argv[2]);
        dgs2[1] = atoi(argv[3]);
        dgs2[2] = atoi(argv[4]);
    }
    srand(time(NULL));
    preFFTRep *F1 = EX_randomPoly(2, dgs1, p);
    preFFTRep *F2 = EX_randomPoly(2, dgs2, p);
    preFFTRep *zerop = CreateZeroPoly();

    t0 = clock();
    resQueue = modular_solve2_select(0, F1, F2, zerop, pPtr);
    t0 = clock() - t0;
    time0 = (double)t0 / (double)CLOCKS_PER_SEC;
    //EX_LinkedQueue_Print(resQueue, EX_RegularChain2_Print);
    EX_LinkedQueue_Free(resQueue, EX_RegularChain2_Free);
    printf(" total = %.3f  ||  ", time0);

    t1 = clock();
    resQueue = modular_solve2_select(1, F1, F2, zerop, pPtr);
    t1 = clock() - t1;
    time1 = (double)t1 / (double)CLOCKS_PER_SEC;
    //EX_LinkedQueue_Print(resQueue, EX_RegularChain2_Print);
    EX_LinkedQueue_Free(resQueue, EX_RegularChain2_Free);
    printf(" total = %.3f\n", time1);

    EX_FreeOnePoly(F1);
    EX_FreeOnePoly(F2);
    EX_FreeOnePoly(zerop);
}

void hgcd_bug() {
    sfixn p = 469762049;
    sfixn i;
    MONTP_OPT2_AS_GENE prime;
    MONTP_OPT2_AS_GENE *pPtr = &prime;
    sfixn dgs1[3] = {0, 49, 1};
    sfixn dgs2[3] = {0, 1, 49};
    LinkedQueue *resQueue = NULL;
    EX_MontP_Init_OPT2_AS_GENE(pPtr, p);

    preFFTRep *F1 = EX_InitOnePoly(2, dgs1);
    preFFTRep *F2 = EX_InitOnePoly(2, dgs2);
    
    // F1 = (1 + x^49) + (1) * y
    F1->data[0]  = 1;
    F1->data[49] = 1;
    F1->data[50] = 1;
    // F2 = (1 + x) + (1) * y^49
    F2->data[0] = 1;
    F2->data[1] = 1;
    F2->data[2*49] = 1;

    printf("Input poly 1 : ");
    EX_Poly_Print(F1);
    printf("\nInput poly 2 : ");
    EX_Poly_Print(F2);
    printf("\n");

    resQueue = EX_ModularSolve2(F1, F2, p);
    EX_LinkedQueue_Print(resQueue, EX_RegularChain2_Print);
    EX_LinkedQueue_Free(resQueue, EX_RegularChain2_Free);
    EX_FreeOnePoly(F1);
    EX_FreeOnePoly(F2);
}

void solve2_tst() {
    sfixn p = 257;
    MONTP_OPT2_AS_GENE prime;
    MONTP_OPT2_AS_GENE *pPtr = &prime;
    sfixn dgs1[3] = {0, 3, 4};
    sfixn dgs2[3] = {0, 2, 3};
    LinkedQueue *resQueue = NULL;
    EX_MontP_Init_OPT2_AS_GENE(pPtr, p);

    preFFTRep *F1 = EX_randomPoly(2, dgs1, p);
    preFFTRep *F2 = EX_randomPoly(2, dgs2, p);
    preFFTRep *zerop = CreateZeroPoly();

    printf("Input poly 1 : ");
    EX_Poly_Print(F1);
    printf("\nInput poly 2 : ");
    EX_Poly_Print(F2);
    printf("\n");

    resQueue = modular_solve2_select(1, F1, F2, zerop, pPtr);
    EX_LinkedQueue_Print(resQueue, EX_RegularChain2_Print);

    EX_FreeOnePoly(F1);
    EX_FreeOnePoly(F2);
    EX_FreeOnePoly(zerop);
    EX_LinkedQueue_Free(resQueue, EX_RegularChain2_Free);
}

void poly_tst() {
    sfixn p = 257;
    MONTP_OPT2_AS_GENE prime;
    MONTP_OPT2_AS_GENE *pPtr = &prime;
    sfixn dgs[] = {0, 3, 3};
    preFFTRep *zerop, *init, *tail, *poly1;
    sfixn i, j;

    EX_MontP_Init_OPT2_AS_GENE(pPtr, p);

    poly1 = EX_InitOnePoly(2, dgs);
    for (j = 0; j <= dgs[2]; ++j) {
        for (i = 0; i <= dgs[1]; ++i) {
            poly1->data[j * (dgs[1] + 1) + i] = i;
        }
    }

    init = EX_getInitial(poly1);
    tail = EX_GetPolyTail(poly1);
    zerop = CreateZeroPoly();
    
    EX_Poly_Print(poly1);
    EX_Poly_Print(init);
    EX_Poly_Print(tail);

    EX_FreeOnePoly(init);
    EX_FreeOnePoly(tail);
    EX_FreeOnePoly(poly1);
    EX_FreeOnePoly(zerop);
}


void regcd() {
    sfixn p = 40961;
    sfixn dgs1[3] = {0, 2, 3};
    sfixn dgs2[3] = {0, 2, 2};
    MONTP_OPT2_AS_GENE prime;
    MONTP_OPT2_AS_GENE *pPtr = &prime;
    TriSet *T;
    sfixn M = 2;
    sfixn N = 3;
    sfixn dgbound = 3;
    LinkedQueue *resQueue;

    EX_MontP_Init_OPT2_AS_GENE(pPtr, p);
    T = EX_initRandomTriSet(N, M, pPtr);

    preFFTRep *f1 = EX_randomPoly(M, dgs1, p);
    preFFTRep *f2 = EX_randomPoly(M, dgs2, p);
    preFFTRep *R = EX_Resultant_Multi(f1, f2, M, pPtr);
    
    EX_FreeOnePoly(ELEMI(T, 1));
    ELEMI(T, 1) = R;
    BDSI(T, 1) = BUSZSI(R, 1) - 1; // main degree of R is BUSZSI(R, 1) 
    

    resQueue = EX_RegularGcd_Wrapped(f1, f2, T, M, pPtr);

    EX_FreeOnePoly(f1);
    EX_FreeOnePoly(f2);
    EX_freeTriSet(T);
    EX_LinkedQueue_Free(resQueue, EX_RegularPair_Free);
}




// testing Univariate Multiplication (Plain, FFT, TFT).
void
UniMul(){
  sfixn p;
  int isOne1, isOne2;
  sfixn degbound, degA, *A, degB, *B, degRES, *RES1, *RES2, *RES3, e, n;
  MONTP_OPT2_AS_GENE  prime;

  printf("Testing univariate multiplication ... ");
  srand(getSeed());
  p=FP[(rand()%100)];
  EX_MontP_Init_OPT2_AS_GENE(&prime,  p);
  degbound=2000;
  degA= rand()%degbound;
  degB= rand()%degbound;
  while (degA<2) degA = rand()%degbound;
  while (degB<2) degB = rand()%degbound;

  A = EX_RandomUniPolyCoeffsVec(degA, p);
  B = EX_RandomUniPolyCoeffsVec(degB, p);
  degRES =degA+degB;

  RES1 = (sfixn *)my_calloc(degRES+1, sizeof(sfixn));
  // plain univariate multiplicaton.
  EX_Mont_PlainMul_OPT2_AS_GENE(degRES, RES1, degA, A, degB, B, &prime);

  RES2 = (sfixn *)my_calloc(degRES+1, sizeof(sfixn));
  // TFT univariate multiplicaton.
  EX_Mont_TFTMul_OPT2_AS_GENE(RES2, degA, A, degB, B, &prime);

  RES3 = (sfixn *)my_calloc(degRES+1, sizeof(sfixn));
  // FFT univariate multiplicaton.
  e=logceiling(degRES+1);
  n=1<<e;
  EX_Mont_FFTMul_OPT2_AS_GENE(n, e, degRES, RES3, degA, A, degB, B, &prime);


  
  isOne1 = compareVec(degRES, RES1, RES2);
  isOne2 = compareVec(degRES, RES2, RES3); 





  if (isOne1 != 1){
    printf("TFT-based Univeraiate Multiplication breaks!\n");
    fflush(stdout);
    my_free(A); my_free(B);
    my_free(RES1); my_free(RES2); my_free(RES3);
    Throw 10000;
  }

  if (isOne2 != 1){
    printf("FFT-based Univeraiate Multiplication breaks!\n");
    fflush(stdout);
    my_free(A); my_free(B);
    my_free(RES1); my_free(RES2); my_free(RES3);
    Throw 10000;
  }

  my_free(A); my_free(B);
  my_free(RES1); my_free(RES2); my_free(RES3); 

  printf("done.\n");

}






void 
UniDiv(){

  sfixn p;
  int isOne, i;
  sfixn degbound, degA, *A, degB, *B, degRES, *RES, degRem, *Rem, degQuo, *Quo;
  MONTP_OPT2_AS_GENE  prime;

  printf("Testing univariate division ... ");

  srand(getSeed());
  p=FP[(rand()%100)];
  EX_MontP_Init_OPT2_AS_GENE(&prime,  p);
  degbound=10;
  degA= rand()%degbound;
  degB= rand()%degbound;
  while (degA<2) degA = rand()%degbound;
  while (degB>degA) degB = rand()%degbound;



  A = EX_RandomUniPolyCoeffsVec(degA, p);
  B = EX_RandomUniPolyCoeffsVec(degB, p);
  
  // the divisor needs to be monic when using fast division.
  B[degB] = 2000;

  Rem = EX_UniRem(&degRem, degA, A, degB, B, &prime);
  Quo = EX_UniQuo(&degQuo, degA, A, degB, B, &prime);

  degRES = degB+degQuo;

  RES = (sfixn *)my_calloc(degRES+1, sizeof(sfixn));

  EX_Mont_PlainMul_OPT2_AS_GENE(degRES, RES, degB, B, degQuo, Quo, &prime);


  for(i=0; i<=degRem; i++) RES[i]=AddMod(RES[i], Rem[i],p);

  isOne = compareVec(degA, A, RES);  
  

  if(isOne != 1){
    printf("Univariate Euclidean division breaks!\n");
    fflush(stdout);
    my_free(A); my_free(B); my_free(RES);
    my_free(Rem); my_free(Quo);
    Throw 10000;
  }
  
  my_free(A); my_free(B); my_free(RES);
  my_free(Rem); my_free(Quo);

  printf("done.\n");

}






int ExtendedGcd(){
  sfixn p,dbound, *A, *B, *CX, *DX, *GX;
  sfixn  dA, dB, dCX, dDX, dGX;
  sfixn dAA, *AA, dBB, *BB, dGcd, *Gcd;
  int isOne;
  MONTP_OPT2_AS_GENE  prime;

  printf("Testing Extend Gcd ... ");

  srand(getSeed());
  p=FP[(rand()%100)];
  EX_MontP_Init_OPT2_AS_GENE(&prime,  p);

  dbound=1000;

  dAA = rand()%dbound;
  while(dAA<1) dAA = rand()%dbound;
  dBB = rand()%dbound;
  while(dBB>dAA) dBB = rand()%dbound;
  dGcd = rand() %dbound;
  while(dGcd<1) dGcd = rand()%dbound;


  dA = dAA+dGcd;
  dB = dBB+dGcd;


  A = (sfixn *) my_calloc(dA+1, sizeof(sfixn));
  B = (sfixn *) my_calloc(dB+1, sizeof(sfixn));

  
  CX = (sfixn *) my_calloc(dA+1, sizeof(sfixn));
  DX = (sfixn *) my_calloc(dA+1, sizeof(sfixn));
  GX = (sfixn *) my_calloc(dA+1, sizeof(sfixn));


  AA = EX_RandomUniPolyCoeffsVecMonic(dAA, p);
  BB = EX_RandomUniPolyCoeffsVecMonic(dBB, p);  
  Gcd = EX_RandomUniPolyCoeffsVecMonic(dGcd, p);   
 
 

  EX_Mont_PlainMul_OPT2_AS_GENE(dA, A, dAA, AA, dGcd, Gcd, &prime);
  EX_Mont_PlainMul_OPT2_AS_GENE(dB, B, dBB, BB, dGcd, Gcd, &prime);

  ExGcd_Uni(CX, &dCX, DX, &dDX, GX, &dGX,  A, dA, B, dB, &prime);

  isOne = compareVec(dGcd, Gcd, GX);
 
  

  if(isOne != 1){
    printf("Extend Euclidean  breaks!\n");
    fflush(stdout);
  my_free(A);
  my_free(B);
  my_free(CX);
  my_free(DX);
  my_free(GX);
  my_free(AA);
  my_free(BB);
  my_free(Gcd);
    Throw 10000;
  }



  my_free(A);
  my_free(B);
  my_free(CX);
  my_free(DX);
  my_free(GX);
  my_free(AA);
  my_free(BB);
  my_free(Gcd);

  printf("done.\n");


  return 0;
}










void
Reduce( ){
  preFFTRep *f1, *f2, *input, *output1, *output2;
  TriSet *ts;
  TriRevInvSet *tris;
  sfixn  N, dgbound, p;
  MONTP_OPT2_AS_GENE  prime;
  int isOne;

  printf("Testing Normalform ... ");

  srand(getSeed());
  p=FP[(rand()%100)];
  EX_MontP_Init_OPT2_AS_GENE(&prime,  p);

  N=4;
  dgbound=4;
  ts = EX_initRandomTriSet(N, dgbound, &prime);


  tris = EX_getRevInvTriSet(N,  ts,  &prime);



  f1 = EX_randomPoly(N, BDS(ts), p);
  f2 = EX_randomPoly(N, BDS(ts), p);
  output1 = EX_InitOnePoly(N, BDS(ts));
  output2 = EX_InitOnePoly(N, BDS(ts));


  input = EX_EX_mulPoly(N, f1, f2, &prime);



  MultiMod_DF(N, output1, input, ts, tris, &prime);

  MultiMod_BULL(N, output2, input, ts, tris, &prime);

  isOne = EX_IsEqualPoly(output1, output2);

  if(isOne != 1){
    printf("MultiMod()  breaks!\n");
    fflush(stdout);
    EX_FreeOnePoly(f1);
    EX_FreeOnePoly(f2);
    EX_FreeOnePoly(input);
    EX_FreeOnePoly(output1);
    EX_FreeOnePoly(output2);
    EX_freeTriSet(ts);
    EX_freeTriRevInvSet(tris);
   
  }


  EX_FreeOnePoly(f1);
  EX_FreeOnePoly(f2);
  EX_FreeOnePoly(input);
  EX_FreeOnePoly(output1);
  EX_FreeOnePoly(output2);
  EX_freeTriSet(ts);
  EX_freeTriRevInvSet(tris);
  
  printf("done.\n");

}






void Lifting(){ 

  int i, j;
  TriSet * ts;
  sfixn N=3, p;
  MONTP_OPT2_AS_GENE * pPtr=(MONTP_OPT2_AS_GENE *)my_malloc(sizeof(MONTP_OPT2_AS_GENE));
  sfixn *dgs=(sfixn *)my_calloc(3, sizeof(sfixn));
  sfixn Y;
  sfixn y0;
  sfixn *inDGS; sfixn *inSIZS; sfixn *inCOEFS;
  sfixn *GNS;
  int offset=0, totSiz=1;
  sfixn MDAS[39*3+43*3]={0,16,0,0,10,0,2,2,2,5,2,3,4,1,4,0,31,0,1,2,0,5,6,7,1,3,0,5,8,9,4,5,10,0,51,0,1,2,0,5,12,13,1,1,0,5,14,15,4,11,16,0,77,0,1,2,0,5,18,19,4,17,20,0,95,0,2,3,2,5,22,23,4,21,24,1,3,0,1,1,0,5,26,27,4,25,28,1,3,0,4,29,30,0,55,0,2,1,2,5,32,33,4,31,34,0,28,0,1,1,0,5,36,37,4,35,38,0,43,0,0,30,0,2,2,2,5,2,3,4,1,4,0,27,0,1,2,0,5,6,7,1,3,0,5,8,9,4,5,10,0,15,0,1,2,0,5,12,13,1,1,0,5,14,15,4,11,16,0,59,0,1,2,0,5,18,19,4,17,20,0,96,0,2,3,2,5,22,23,4,21,24,0,72,0,1,3,0,5,26,27,1,1,0,5,28,29,4,25,30,0,87,0,1,3,0,5,32,33,4,31,34,0,47,0,2,1,2,5,36,37,4,35,38,0,90,0,1,1,0,5,40,41,4,39,42};

  sfixn esti_dy=32;
  sfixn esti_siz=esti_dy*(dgs[1]+1)+esti_dy*(dgs[1]+1)*(dgs[2]+1);
  sfixn *outPDGVECS=(sfixn *) my_calloc(esti_siz, sizeof(sfixn));
  sfixn *outCOEFVECS=(sfixn *)my_calloc(esti_siz, sizeof(sfixn));
  int iter=0;

  printf("Testing lifting ... ");



  dgs[0]=1;
  dgs[1]=4;
  dgs[2]=1;
 
  srand(getSeed());
  p=FP[rand()%100];
  p = 469762049;
  
  EX_MontP_Init_OPT2_AS_GENE(pPtr,  p);


  // vec_slg = example_1_PolyVec_SLG();



  ts=(TriSet *)my_malloc(sizeof(TriSet));
  init_example_1_DenseTriSetForLifting_y0_10(N, dgs, ts, pPtr);

  


  Y=1;  y0=10;

  //final_ts = EX_UniNewtonLift(Y, y0, vec_slg, ts, N, pPtr);

  

  inDGS = (sfixn *) my_calloc(N*N,sizeof(sfixn));
  for(i=1; i<=N; i++){
    for(j=1; j<=N; j++){
      inDGS[(i-1)*N + j-1]=BUSZSI(ELEMI(ts, i), j);
    }
  }

  inSIZS = (sfixn *) my_calloc(N,sizeof(sfixn));
  for(i=1; i<=N; i++){
      inSIZS[i-1]=SIZ(ELEMI(ts, i));
  }



  for(i=0; i<N; i++){
    totSiz*=(inSIZS[i]);
  }

  inCOEFS= (sfixn *) my_calloc(totSiz,sizeof(sfixn));

  for(i=1; i<=N; i++){
    for(j=0; j<inSIZS[i-1]; j++){
      inCOEFS[offset+j]=(DAT(ELEMI(ts, i)))[j];
    }
    offset+=inSIZS[i-1];
  }



  GNS=(sfixn *)my_calloc(2, sizeof(sfixn));
  GNS[0]=39;
  GNS[1]=43;


  printf("before lifting\n");

  iter = NewtonLiftUniCN(outPDGVECS, outCOEFVECS, Y, y0, N, GNS, MDAS,  dgs, inDGS, inSIZS, inCOEFS, p);

  printf("after lifting ... ");

  my_free(outPDGVECS); my_free(outCOEFVECS);  
  my_free(GNS);  
  my_free(inDGS); my_free(inSIZS); my_free(inCOEFS);

  EX_freeTriSet(ts);

  my_free(dgs);
  my_free(pPtr);

  printf("done.\n");



}

extern int Interrupted;

int main(int argc, char *argv[]){

    //FILE *file;
    //file = fopen ("TEST-RESULT","a");
    //UniMul();
    //fprintf(file, "Univariate plain/FFT/TFT multiplication is correct.\n");
    //fclose(file);
    //if(Interrupted) return 0;
    //fflush(stdout);
    
    //file = fopen ("TEST-RESULT","a");
    //UniDiv();
    //fprintf(file, "Univariate plain/fast Euclidean division is correct.\n");
    //fclose(file);
    
    //if(Interrupted) return 0;
    //fflush(stdout);
    
    //file = fopen ("TEST-RESULT","a");
    //ExtendedGcd();
    //fprintf(file, "Univariate Extended Euclidean algorithm is correct.\n");
    //fclose(file);
    //if(Interrupted) return 0;
    //fflush(stdout);
    
    //file = fopen ("TEST-RESULT","a");
    //Reduce( );
    //fprintf(file, "Univariate NormalForm is correct.\n");
    //fclose(file);
    //if(Interrupted) return 0;
    //fflush(stdout);
    
    //file = fopen ("TEST-RESULT","a");
    //Lifting();
    //fprintf(file, "Univariate Lifting is correct.\n");
    //fclose(file);
    
    //if(Interrupted) return 0;
    //fflush(stdout);
    //poly_tst();
    solve2_tst();
    //random_dense_solve2(argc, argv);
    //hass_solve2(argc, argv);
    //HGCD_tst(argc, argv);
    //uniquo_tst(argc, argv);
    //resultant3_tst(argc, argv);
    //hgcd_bug();
    return 0;
}

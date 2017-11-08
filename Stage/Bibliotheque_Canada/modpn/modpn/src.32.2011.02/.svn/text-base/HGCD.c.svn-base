/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#include "HGCD.h"
#define max(x, y) ((x)>(y)?(x):(y))
#define min(x, y) ((x)>(y)?(y):(x))


typedef struct{
    sfixn degrees[4];
    sfixn *polynomials0;
    sfixn *polynomials1;
    sfixn *polynomials2;
    sfixn *polynomials3;
} matrix;


/* Utility: printing a polynomial */
void printIt(char *s, sfixn *A, sfixn dA){
    sfixn i;
    printf(s);
    if (dA == -1){
        printf("0;\n");
        return;
    }
    for (i = 0; i < dA; i++)
        printf("%ld*x^%ld +", A[i], i);
    printf("%ld*x^%ld;\n", A[dA], dA);
}

// negation of a polynomial
static void neg(sfixn* C, sfixn dC, sfixn PPP){
    sfixn i;
    for (i = 0; i <= dC; i++)
        C[i] = NegMod(C[i],PPP);     // use NegMod here
}

// copy of a polynomial
static void copy(sfixn* C, sfixn *dC, sfixn *A, sfixn dA){
    sfixn i;
    for (i = 0; i <= dA; i++) C[i] = A[i];
    dC[0] = dA;
}

// addition of polynomials
static 
void add(sfixn* C, sfixn *dC, sfixn* A, sfixn dA, sfixn* B, sfixn dB, sfixn PPP){
    sfixn i;
    if (dA < dB){ // if dA < dB, switch polynomials
        add(C, dC, B, dB, A, dA, PPP);
        return;
    }
    // assume dA >= dB
    for (i = 0; i <= dB; i++)
        C[i] = AddMod(A[i], B[i], PPP);    // use AddMod here!!!
    for (i = dB+1; i <= dA; i++)
        C[i] = A[i];
    for (i = dA; i >= 0; i--)
        if (C[i] != 0){
            dC[0] = i;
            return;
        }
    dC[0] = -1;
}

// subtraction of polynomials  C = A-B
static 
void sub(sfixn* C, sfixn *dC, sfixn* A, sfixn dA, sfixn* B, sfixn dB, sfixn PPP){
    sfixn i;
    if (dA < dB){ // dB > dA
        for (i = 0; i <= dA; i++){
            C[i] = SubMod(A[i],B[i],PPP);    
        }
        for (i = dA+1; i <= dB; i++)
            C[i] = NegMod(B[i], PPP);     // SubMod
        for (i = dB; i >= 0; i--)
            if (C[i] != 0){
                dC[0] = i;
                return;
            }
        dC[0] = -1;
        return;
    }
    else{ // dB <= dA
        for (i = 0; i <= dB; i++){
            C[i] = SubMod(A[i],B[i],PPP);    // use SubMod here!!!
        }
        for (i = dB+1; i <= dA; i++)
            C[i] = A[i];
        for (i = dA; i >= 0; i--)
            if (C[i] != 0){
                dC[0] = i;
                return;
            }
        dC[0] = -1;
        return;
    }
}

// naive multiplication
static void mul(sfixn* C, sfixn *dC, sfixn* A, sfixn dA, sfixn* B, sfixn dB,
        MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn e,n;
    if (dA == -1 || dB == -1){
        dC[0] = -1;
        return;
    }

    dC[0] = dA+dB;

    e=logceiling(dC[0]+1);
    n=1<<e;
    cleanVec(dC[0], C);
    if((dA<16)||(dB<16))  
        EX_Mont_PlainMul_OPT2_AS_GENE(dC[0], C, dA, A, dB, B, pPtr);
    else 
        EX_Mont_FFTMul_OPT2_AS_GENE(n, e, dC[0],  C, dA, A, dB, B, pPtr);

}


// quotient in the Euclidean division
static 
void quo(sfixn* Q, sfixn *dQ, sfixn* A, sfixn dA, sfixn* B, sfixn dB, sfixn PPP){
    sfixn i, j;
    sfixn *tmp, u, v;

    tmp = (sfixn *) my_calloc((dA+1), sizeof(sfixn));
    for (i = 0; i <= dA; i++)
        tmp[i] = A[i];

    dQ[0] = dA-dB;
    u = inverseMod(B[dB], PPP);

    for (i = dA-dB; i >= 0; i--){
        if (tmp[dB+i] != 0){
            Q[i] = MulMod(tmp[dB+i],u,PPP);
            for (j = 0; j <= dB; j++){
                v = MulMod(Q[i],B[j],PPP);   // MulMod
                tmp[i+j] = SubMod(tmp[i+j], v, PPP); // SubMod
            }
        }
    }
    my_free(tmp);
}

void EX_QUO_Uni_Wrapper(sfixn *Q, sfixn *dQ, sfixn *A, sfixn dA, sfixn *B, 
        sfixn dB, MONTP_OPT2_AS_GENE *pPtr)
{
    sfixn i, invu, *UB, p = pPtr->P;
    if (dA - dB >= 128) {
        dQ[0] = dA - dB;
        if (B[dB] != 1) {
            invu = inverseMod(B[dB], p);
            UB = (sfixn *)my_malloc(sizeof(sfixn)*(dB + 1));
            for (i = 0; i <= dB; ++i) { UB[i] = MulMod(B[i], invu, p); }
            fastQuo(*dQ, Q, dA, A, dB, UB, pPtr);
            dQ[0] = shrinkDegUni(dQ[0], Q);
            for (i = 0; i <= dQ[0]; ++i) { Q[i] = MulMod(Q[i], invu, p); }
            my_free(UB);
        } else {
            fastQuo(*dQ, Q, dA, A, dB, B, pPtr);
            dQ[0] = shrinkDegUni(dQ[0], Q);
        }
    } else {
        quo(Q, dQ, A, dA, B, dB, p);
    }
}


// R = AB
static 
matrix multiply(matrix* A, matrix* B, MONTP_OPT2_AS_GENE * pPtr){
    sfixn *tmp, dTmp, *tmp2, dTmp2;
    sfixn d, dA, dB;
    matrix R;

    dA = A->degrees[0];
    dA = max(dA, A->degrees[1]);
    dA = max(dA, A->degrees[2]);
    dA = max(dA, A->degrees[3]);

    dB = B->degrees[0];
    dB = max(dB, B->degrees[1]);
    dB = max(dB, B->degrees[2]);
    dB = max(dB, B->degrees[3]);

    d = dA+dB+2;
    tmp = (sfixn *)my_calloc(d, sizeof(sfixn));
    tmp2 = (sfixn *)my_calloc(d, sizeof(sfixn));

    R.polynomials0 = (sfixn *) my_calloc(d, sizeof(sfixn));
    R.polynomials1 = (sfixn *) my_calloc(d, sizeof(sfixn));
    R.polynomials2 = (sfixn *) my_calloc(d, sizeof(sfixn));
    R.polynomials3 = (sfixn *) my_calloc(d, sizeof(sfixn));

    mul(tmp2, &dTmp2, A->polynomials0, A->degrees[0], B->polynomials0, B->degrees[0], pPtr);
    mul(tmp,  &dTmp,  A->polynomials1, A->degrees[1], B->polynomials2, B->degrees[2], pPtr);
    add(R.polynomials0, &(R.degrees[0]), tmp2, dTmp2, tmp, dTmp, pPtr->P);

    mul(tmp2, &dTmp2, A->polynomials0, A->degrees[0], B->polynomials1, B->degrees[1],pPtr );
    mul(tmp,  &dTmp,  A->polynomials1, A->degrees[1], B->polynomials3, B->degrees[3],pPtr );
    add(R.polynomials1, &(R.degrees[1]), tmp2, dTmp2, tmp, dTmp, pPtr->P);

    mul(tmp2, &dTmp2, A->polynomials2, A->degrees[2], B->polynomials0, B->degrees[0], pPtr);
    mul(tmp,  &dTmp,  A->polynomials3, A->degrees[3], B->polynomials2, B->degrees[2], pPtr);
    add(R.polynomials2, &(R.degrees[2]), tmp2, dTmp2, tmp, dTmp, pPtr->P);

    mul(tmp2, &dTmp2, A->polynomials2, A->degrees[2], B->polynomials1, B->degrees[1], pPtr);
    mul(tmp,  &dTmp,  A->polynomials3, A->degrees[3], B->polynomials3, B->degrees[3], pPtr);
    add(R.polynomials3, &(R.degrees[3]), tmp2, dTmp2, tmp, dTmp, pPtr->P);

    my_free(tmp);
    my_free(tmp2);
    return R;
}

static 
void setIdentityMatrix(matrix *U){
    U->polynomials0 = (sfixn*) my_malloc(1*sizeof(sfixn));
    U->polynomials1 = (sfixn*) my_malloc(0*sizeof(sfixn));
    U->polynomials2 = (sfixn*) my_malloc(0*sizeof(sfixn));
    U->polynomials3 = (sfixn*) my_malloc(1*sizeof(sfixn));
    U->degrees[0] = 0;
    U->polynomials0[0] = 1;
    U->degrees[1] = -1;  // degree of the zero polynomial is -1???
    U->degrees[2] = -1;
    U->degrees[3] = 0;
    U->polynomials3[0] = 1;
}

static 
void freeMatrix(matrix *U){
    my_free(U->polynomials0);
    my_free(U->polynomials1);
    my_free(U->polynomials2);
    my_free(U->polynomials3);
}


static 
void HGCD_slow(matrix *T, sfixn *A, sfixn dA, sfixn *B, sfixn dB, MONTP_OPT2_AS_GENE * pPtr){
    sfixn dTmp2, dTmp, dQ, dR1, dR2, dA1, dA2, dB1, dB2, m = (dA+1)/2;
    sfixn *tmp2, *tmp, *Q, *R1, *R2, *A1, *A2, *B1, *B2;




    if (dB < m){
        setIdentityMatrix(T);
        return;
    }

    R1 = (sfixn *) my_calloc(dA+1, sizeof(sfixn));
    R2 = (sfixn *) my_calloc(dA+1, sizeof(sfixn));
    A1 = (sfixn *) my_calloc(dA+1, sizeof(sfixn));
    A2 = (sfixn *) my_calloc(dA+1, sizeof(sfixn));
    B1 = (sfixn *) my_calloc(dA+1, sizeof(sfixn));
    B2 = (sfixn *) my_calloc(dA+1, sizeof(sfixn));
    tmp2 = (sfixn *) my_calloc(dA+1, sizeof(sfixn));
    tmp = (sfixn *) my_calloc(dA+1, sizeof(sfixn));
    Q = (sfixn *) my_calloc(dA+1, sizeof(sfixn));

    copy(R1, &dR1, A, dA);
    copy(R2, &dR2, B, dB);
    A1[0] = 1;
    dA1 = 0;
    dA2 = -1;
    dB1 = -1;
    B2[0] = 1;
    dB2 = 0;

    while (dR2 >= m){
        EX_QUO_Uni_Wrapper(Q, &dQ, R1, dR1, R2, dR2, pPtr);

        copy(tmp, &dTmp, R2, dR2);
        mul(tmp2, &dTmp2, R2, dR2, Q, dQ, pPtr);
        sub(R2, &dR2, R1, dR1, tmp2, dTmp2, pPtr->P);
        copy(R1, &dR1, tmp, dTmp);

        copy(tmp, &dTmp, A2, dA2);
        mul(tmp2, &dTmp2, A2, dA2, Q, dQ, pPtr);
        sub(A2, &dA2, A1, dA1, tmp2, dTmp2, pPtr->P);
        copy(A1, &dA1, tmp, dTmp);

        copy(tmp, &dTmp, B2, dB2);
        mul(tmp2, &dTmp2, B2, dB2, Q, dQ, pPtr);
        sub(B2, &dB2, B1, dB1, tmp2, dTmp2, pPtr->P);
        copy(B1, &dB1, tmp, dTmp);
    }

    T->polynomials0 = (sfixn *) my_calloc((dA1+1), sizeof(sfixn));
    copy(T->polynomials0, &(T->degrees[0]), A1, dA1);

    T->polynomials2 = (sfixn *) my_calloc((dA2+1), sizeof(sfixn));
    copy(T->polynomials2, &(T->degrees[2]), A2, dA2);

    T->polynomials1 = (sfixn *) my_calloc((dB1+1), sizeof(sfixn));
    copy(T->polynomials1, &(T->degrees[1]), B1, dB1);

    T->polynomials3 = (sfixn *) my_calloc((dB2+1), sizeof(sfixn));
    copy(T->polynomials3, &(T->degrees[3]), B2, dB2);

    my_free(R1);
    my_free(R2);
    my_free(A1);
    my_free(A2);
    my_free(B1);
    my_free(B2);
    my_free(tmp2);
    my_free(tmp);
    my_free(Q);
}

//-----------------------------------------------------------
// Recursive HGCD algorithm
//-----------------------------------------------------------
void HGCD(matrix *T, sfixn *A, sfixn dA, sfixn *B, sfixn dB, MONTP_OPT2_AS_GENE * pPtr){
    sfixn *tA, *tB, *tmp, *tmp1, *tmp2;
    sfixn dTa, dTb, k, d, dA0, dB0, m, dTmp1, dTmp2, dTmp;
    matrix R, S, U, V, W;


    if (dB >= dA){
        printf("HGCD: error in the degrees dA = %d, dB = %d.\n", dA, dB);
        exit(-1);
    } // we can assume dA > dB
    
    if (dA <= 200){
        HGCD_slow(T, A, dA, B, dB, pPtr);
        return;
    } // handle the low degrees

    m = (1+dA) / 2;

    if (dB < m){
        setIdentityMatrix(T);
        return;
    }

    d = dA+dB+2;  // a bound on the number of coefficients 
    tmp = (sfixn *) my_malloc(d*sizeof(sfixn)); 
    tmp1 = (sfixn *) my_malloc(d*sizeof(sfixn));
    tmp2 = (sfixn *) my_malloc(d*sizeof(sfixn));
    tA = (sfixn *) my_malloc(d*sizeof(sfixn));
    tB = (sfixn *) my_malloc(d*sizeof(sfixn));

    // recursive call
    dA0 = dA-m;
    dB0 = dB-m;
    HGCD(&R, A+m, dA0, B+m, dB0, pPtr);

    // apply the HGCD matrix to the input polynomials
    // tmp1 = R[0]*A
    mul(tmp1, &dTmp1, R.polynomials0, R.degrees[0], A, dA, pPtr);
    // tmp2 = R[1]*B
    mul(tmp2, &dTmp2, R.polynomials1, R.degrees[1], B, dB, pPtr);
    // tmp = R0+R1
    add(tmp, &dTmp, tmp1, dTmp1, tmp2, dTmp2, pPtr->P);

    // tmp1 = R[2]*A
    mul(tmp1, &dTmp1, R.polynomials2, R.degrees[2], A, dA, pPtr);
    // tmp2 = R[3]*B
    mul(tmp2, &dTmp2, R.polynomials3, R.degrees[3], B, dB, pPtr);
    // B = tmp1+tmp2
    add(tB, &dTb, tmp1, dTmp1, tmp2, dTmp2, pPtr->P);
    // A = tmp
    copy(tA, &dTa, tmp, dTmp);

    // early exit, if degred(B) < m
    if (dTb < m){
        T->polynomials0 = (sfixn *) my_malloc((1+R.degrees[0])*sizeof(sfixn));
        T->polynomials1 = (sfixn *) my_malloc((1+R.degrees[1])*sizeof(sfixn));
        T->polynomials2 = (sfixn *) my_malloc((1+R.degrees[2])*sizeof(sfixn));
        T->polynomials3 = (sfixn *) my_malloc((1+R.degrees[3])*sizeof(sfixn));

        copy(T->polynomials0, &(T->degrees[0]), R.polynomials0, R.degrees[0]);
        copy(T->polynomials1, &(T->degrees[1]), R.polynomials1, R.degrees[1]);
        copy(T->polynomials2, &(T->degrees[2]), R.polynomials2, R.degrees[2]);
        copy(T->polynomials3, &(T->degrees[3]), R.polynomials3, R.degrees[3]);

        my_free(tmp);
        my_free(tmp1);
        my_free(tmp2);
        my_free(tA);
        my_free(tB);
        freeMatrix(&R);
        return;
    }

    // Euclidean division
    // tmp2 = Q
    EX_QUO_Uni_Wrapper(tmp2, &dTmp2, tA, dTa, tB, dTb, pPtr);
    neg(tmp2, dTmp2, pPtr->P);
    // tmp = A
    copy(tmp, &dTmp, tA, dTa);
    // A = B
    copy(tA, &dTa, tB, dTb);  
    // tmp1 = QB
    mul(tmp1, &dTmp1, tmp2, dTmp2, tB, dTb, pPtr);
    // B = tmp - tmp1 = A_old + QB
    add(tB, &dTb, tmp, dTmp, tmp1, dTmp1, pPtr->P);

    // recursive call  
    k=2*m-dTa;
    dA0 = dTa-k;
    dB0 = dTb-k;
    HGCD(&S, tA+k, dA0, tB+k, dB0, pPtr);

    // matrix products
    U.polynomials0 = (sfixn*) my_calloc(1, sizeof(sfixn));
    U.polynomials1 = (sfixn*) my_calloc(1, sizeof(sfixn));
    U.polynomials2 = (sfixn*) my_calloc(1, sizeof(sfixn));
    U.polynomials3 = (sfixn*) my_calloc((1+dTmp2), sizeof(sfixn));
    U.degrees[0] = -1;
    U.degrees[1] = 0;
    U.polynomials1[0] = 1;
    U.degrees[2] = 0;
    U.polynomials2[0] = 1;
    copy(U.polynomials3, &(U.degrees[3]), tmp2, dTmp2);

    V=multiply(&U, &R, pPtr);
    W=multiply(&S, &V, pPtr);

    // T = W
    T->polynomials0 = (sfixn *) my_calloc((1+(W.degrees[0])), sizeof(sfixn));
    T->polynomials1 = (sfixn *) my_calloc((1+(W.degrees[1])), sizeof(sfixn));
    T->polynomials2 = (sfixn *) my_calloc((1+(W.degrees[2])), sizeof(sfixn));
    T->polynomials3 = (sfixn *) my_calloc((1+(W.degrees[3])), sizeof(sfixn));

    copy(T->polynomials0, &(T->degrees[0]), W.polynomials0, W.degrees[0]);
    copy(T->polynomials1, &(T->degrees[1]), W.polynomials1, W.degrees[1]);
    copy(T->polynomials2, &(T->degrees[2]), W.polynomials2, W.degrees[2]);
    copy(T->polynomials3, &(T->degrees[3]), W.polynomials3, W.degrees[3]);

    my_free(tmp);
    my_free(tmp1);
    my_free(tmp2);
    freeMatrix(&U);
    freeMatrix(&S);
    freeMatrix(&R);
    freeMatrix(&V);
    freeMatrix(&W);
    my_free(tA);
    my_free(tB);
}

//-----------------------------------------------------------
// Recursive GCD algorithm, using HGCD
//-----------------------------------------------------------
void recursiveGCD(matrix *T, sfixn *A, sfixn dA, sfixn *B, sfixn dB, MONTP_OPT2_AS_GENE * pPtr){
    sfixn *tA, *tB, *tmp, *tmp1, *tmp2;
    sfixn dTa, dTb, d, dTmp1, dTmp2, dTmp;
    matrix R, S, U, V, W;

    if (dB < 0){
        setIdentityMatrix(T);
        return;
    }

    // call HGCD
    HGCD(&R, A, dA, B, dB, pPtr);
    d = dA+dB+2;  // a bound on the number of coefficients 
    tmp = (sfixn *) my_malloc(d*sizeof(sfixn)); 
    tmp1 = (sfixn *) my_malloc(d*sizeof(sfixn));
    tmp2 = (sfixn *) my_malloc(d*sizeof(sfixn));
    tA = (sfixn *) my_malloc(d*sizeof(sfixn));
    tB = (sfixn *) my_malloc(d*sizeof(sfixn));

    // apply the HGCD matrix to the input polynomials
    // tmp1 = R[0]*A
    mul(tmp1, &dTmp1, R.polynomials0, R.degrees[0], A, dA, pPtr);
    // tmp2 = R[1]*B
    mul(tmp2, &dTmp2, R.polynomials1, R.degrees[1], B, dB, pPtr);
    // tmp = tmp1+tmp2
    add(tmp, &dTmp, tmp1, dTmp1, tmp2, dTmp2, pPtr->P);

    // tmp1 = R[2]*A
    mul(tmp1, &dTmp1, R.polynomials2, R.degrees[2], A, dA, pPtr);
    // tmp2 = R[3]*B
    mul(tmp2, &dTmp2, R.polynomials3, R.degrees[3], B, dB, pPtr);
    // B = tmp1+tmp2
    add(tB, &dTb, tmp1, dTmp1, tmp2, dTmp2, pPtr->P);
    // A = tmp
    copy(tA, &dTa, tmp, dTmp);

    // if B=0, finished
    if (dTb < 0){
        T->polynomials0 = (sfixn *) my_malloc((1+R.degrees[0])*sizeof(sfixn));
        T->polynomials1 = (sfixn *) my_malloc((1+R.degrees[1])*sizeof(sfixn));
        T->polynomials2 = (sfixn *) my_malloc((1+R.degrees[2])*sizeof(sfixn));
        T->polynomials3 = (sfixn *) my_malloc((1+R.degrees[3])*sizeof(sfixn));

        copy(T->polynomials0, &(T->degrees[0]), R.polynomials0, R.degrees[0]);
        copy(T->polynomials1, &(T->degrees[1]), R.polynomials1, R.degrees[1]);
        copy(T->polynomials2, &(T->degrees[2]), R.polynomials2, R.degrees[2]);
        copy(T->polynomials3, &(T->degrees[3]), R.polynomials3, R.degrees[3]);

        my_free(tmp);
        my_free(tmp1);
        my_free(tmp2);
        my_free(tA);
        my_free(tB);
        freeMatrix(&R);
        return;
    }

    // Euclidean division
    // tmp2 = Q
    EX_QUO_Uni_Wrapper(tmp2, &dTmp2, tA, dTa, tB, dTb, pPtr);
    neg(tmp2, dTmp2, pPtr->P);
    // tmp = A
    copy(tmp, &dTmp, tA, dTa);
    // A = B
    copy(tA, &dTa, tB, dTb);  
    // tmp1 = QB
    mul(tmp1, &dTmp1, tmp2, dTmp2, tB, dTb, pPtr);
    // B = tmp - tmp1 = A_old + QB
    add(tB, &dTb, tmp, dTmp, tmp1, dTmp1, pPtr->P);

    // recursive call
    recursiveGCD(&S, tA, dTa, tB, dTb, pPtr);

    // matrix multiplications
    U.polynomials0 = (sfixn*) my_calloc(1, sizeof(sfixn));
    U.polynomials1 = (sfixn*) my_calloc(1, sizeof(sfixn));
    U.polynomials2 = (sfixn*) my_calloc(1, sizeof(sfixn));
    U.polynomials3 = (sfixn*) my_calloc((1+dTmp2), sizeof(sfixn));
    U.degrees[0] = -1;
    U.degrees[1] = 0;
    U.polynomials1[0] = 1;
    U.degrees[2] = 0;
    U.polynomials2[0] = 1;
    copy(U.polynomials3, &(U.degrees[3]), tmp2, dTmp2);
    V=multiply(&U, &R, pPtr);
    W=multiply(&S, &V, pPtr);

    // copy the output
    T->polynomials0 = (sfixn *) my_calloc((1+(W.degrees[0])), sizeof(sfixn));
    T->polynomials1 = (sfixn *) my_calloc((1+(W.degrees[1])), sizeof(sfixn));
    T->polynomials2 = (sfixn *) my_calloc((1+(W.degrees[2])), sizeof(sfixn));
    T->polynomials3 = (sfixn *) my_calloc((1+(W.degrees[3])), sizeof(sfixn));

    copy(T->polynomials0, &(T->degrees[0]), W.polynomials0, W.degrees[0]);
    copy(T->polynomials1, &(T->degrees[1]), W.polynomials1, W.degrees[1]);
    copy(T->polynomials2, &(T->degrees[2]), W.polynomials2, W.degrees[2]);
    copy(T->polynomials3, &(T->degrees[3]), W.polynomials3, W.degrees[3]);

    // cleaning
    my_free(tmp);
    my_free(tmp1);
    my_free(tmp2);
    freeMatrix(&U);
    freeMatrix(&S);
    freeMatrix(&R);
    freeMatrix(&V);
    freeMatrix(&W);
    my_free(tA);
    my_free(tB);
}

//-----------------------------------------------------------
// XGCD algorithm, using HGCD
// C, D, G must have been initialized before
//-----------------------------------------------------------
/**
 * ExGcd_Uni:
 * @C: Coefficient vector for Bezout coefficent 1.
 * @dC: Pointer for the degree of Bezout coefficent 1.
 * @D: Coefficient vector for Bezout coefficent 2.
 * @dD: Pointer for the degree of Bezout coefficent 2.
 * @G: Coefficient vector for GCD.
 * @dG: Pointer for the degree of GCD.
 * @A: Coefficient vector for the input polynomial 1.
 * @dA: Degree of input polynomial 1.
 * @B: Coefficient vector for the input polynomail 2.
 * @dB: Degree of input polynomial 2.
 * @pPtr: Info for prime number.
 * 
 *  Given A and B, to compute GCD(A, B) and Bezout coefficients.
 *  Extended Euclidean Algorithm.
 *      C * A  +  D * B  = G.
 * Return value: 
 **/

void XGCD(sfixn *C, sfixn *dC, sfixn *D, sfixn *dD, sfixn *G, sfixn *dG, 
        sfixn *A, sfixn dA, sfixn *B, sfixn dB, MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn *tA=NULL, *tB=NULL, *tmp=NULL, *tmp1=NULL, *tmp2=NULL;
    sfixn dTa, dTb, d, dTmp1, dTmp2, dTmp;
    matrix S, U, V;

    if (dB < 0){
        dC[0] = 0;
        C[0] = 1;
        dD[0] = -1;
        copy(G, dG, A, dA); // size of G is dA + 1
        return;
    }
    if (dB > dA){
        XGCD(D, dD, C, dC, G, dG, B, dB, A, dA, pPtr);
        return;
    }

    d = dA+dB+3;
    tmp = (sfixn *) my_malloc(d*sizeof(sfixn));
    tmp1 = (sfixn *) my_malloc(d*sizeof(sfixn));
    tmp2 = (sfixn *) my_malloc(d*sizeof(sfixn));
    tA = (sfixn *) my_malloc(d*sizeof(sfixn));
    tB = (sfixn *) my_malloc(d*sizeof(sfixn));

    // tmp2 = Q
    EX_QUO_Uni_Wrapper(tmp2, &dTmp2, A, dA, B, dB, pPtr);
    neg(tmp2, dTmp2, pPtr->P);
    // tmp = A
    copy(tmp, &dTmp, A, dA);
    // A = B
    copy(tA, &dTa, B, dB);
    // tmp1 = QB
    mul(tmp1, &dTmp1, tmp2, dTmp2, B, dB, pPtr);
    // B = tmp - tmp1 = A_old + QB
    add(tB, &dTb, tmp, dTmp, tmp1, dTmp1, pPtr->P);
 
    recursiveGCD(&S, tA, dTa, tB, dTb, pPtr);

    U.polynomials0 = (sfixn*) my_calloc(1, sizeof(sfixn));
    U.polynomials1 = (sfixn*) my_calloc(1, sizeof(sfixn));
    U.polynomials2 = (sfixn*) my_calloc(1, sizeof(sfixn));
    U.polynomials3 = (sfixn*) my_calloc((1+dTmp2), sizeof(sfixn));
    U.degrees[0] = -1;
    U.degrees[1] = 0;
    U.polynomials1[0] = 1;
    U.degrees[2] = 0;
    U.polynomials2[0] = 1;
    copy(U.polynomials3, &(U.degrees[3]), tmp2, dTmp2);
    V=multiply(&S, &U, pPtr);

    copy(C, dC, V.polynomials0, V.degrees[0]);
    copy(D, dD, V.polynomials1, V.degrees[1]);
    
    // tmp1 = R[0]*A
    mul(tmp1, &dTmp1, C, dC[0], A, dA, pPtr);
    // tmp2 = R[1]*B
    mul(tmp2, &dTmp2, D, dD[0], B, dB, pPtr);
    // tmp = R0+R1
    //add(G, dG, tmp1, dA, tmp2, dA, pPtr->P); ---> Fixed by Eric on Dec. 14, 07.
    add(G, dG, tmp1, dTmp1, tmp2, dTmp2, pPtr->P);

    my_free(tmp);
    my_free(tmp1);
    my_free(tmp2);
    my_free(tA);
    my_free(tB);
    freeMatrix(&U);
    freeMatrix(&V);
    freeMatrix(&S);
}

/**
 * Given A and B, to compute GCD(A, B) = G 
 *
 * @G: Coefficient vector for GCD
 * @dG: Pointer for the degree of GCD
 * @A: Coefficient vector for the input polynomial 1
 * @dA: Degree of input polynomial 1
 * @B: Coefficient vector for the input polynomail 2
 * @dB: Degree of input polynomial 2
 * @pPtr: prime number
 * 
 * Added by WP on Jan 10, 2011.
 */
sfixn* HalfGCD(sfixn *dG, sfixn *A, sfixn dA, sfixn *B, sfixn dB, 
    MONTP_OPT2_AS_GENE *pPtr) 
{
    sfixn i, *C, dC, *D, dD, *G, *GCD;

    if (dA < dB) return HalfGCD(dG, B, dB, A, dA, pPtr);

    // In this file, zero polynomial has degree -1.
    // But in other files of modpn, zero polynomial has degree 0.
    // In this function, we use the latter.
    
    // B is nonzero constant
    if (dB == 0 && B[0] != 0) {
        G = (sfixn *)my_malloc(sizeof(sfixn));
        G[0] = 1, *dG = 0;
        return G;
    }

    // B is zero
    if (dB == 0 && B[0] == 0) {
        G = (sfixn *)my_malloc((dA + 1)*sizeof(sfixn));
        for (i = 0; i <= dA; ++i) G[i] = A[i];
        *dG = dA;
        return G;
    }

    // preallocate C, D and G
    C = (sfixn *)my_malloc((dA + dB + 1) * sizeof(sfixn));
    D = (sfixn *)my_malloc((dA + dB + 1) * sizeof(sfixn));
    G = (sfixn *)my_malloc((dA + dB + 1) * sizeof(sfixn));
    //G = (sfixn *)my_malloc((dB + 1) * sizeof(sfixn));

    dC = dB, dD = dA, dG[0] = 0;

    XGCD(C, &dC, D, &dD, G, dG, A, dA, B, dB, pPtr);

    my_free(C);
    my_free(D);
    if (dG[0] == dB) { return G; }
    GCD = (sfixn *)my_malloc((dG[0]+ 1)*sizeof(sfixn));
    for (i = 0; i <= dG[0]; ++i) GCD[i] = G[i];
    my_free(G);
    return GCD;
}

sfixn* EX_GCD_Uni_Wrapper(sfixn *dG, sfixn *A, sfixn dA, sfixn *B, sfixn dB,
    MONTP_OPT2_AS_GENE *pPtr) 
{
    sfixn *G = NULL;
    if (dA >= 20000) {
        // BUG in HGCD
        // when solving {x^49 + y + 1,  x + y^41 + 1} 
        // with p = 469762049
        G = HalfGCD(dG, A, dA, B, dB, pPtr);
    } else {
        G = EX_GCD_UNI(dG, A, dA, B, dB, pPtr);
    }
    return G;
}

//////////////////////// END OF FILE //////////////////////////////////////////

#include "solve2.h"

/**
 * Solving bivariate polynomial systems.
 *
 * WP, Sat Jan 1 21:15:07 EST 2011
 */

/**
 * next_regular_subres_index
 * 
 * @scube, the subresultant chain 
 * @i, the current index
 * 
 * Compute the smallest index j >= i, such that S[j] is a regular 
 * subresultant, w if no such a regular subresultant exists. 
 *
 * Assumption:
 *
 * (1) i >= 1 and i <= w
 */
sfixn next_regular_subres_index(sfixn i, SCUBE *scube, MONTP_OPT2_AS_GENE *pPtr)
{
    preFFTRep *init=NULL;
    sfixn w, j;

    w = EX_WidthInSCUBE(scube);
    // the input polynomial with smaller degree is always regarded 
    // as a regular subresultant
    if (i == w) return w;
    for (j = i; j < w; ++j) {
        init = EX_IthDthCoeff(j, j, scube, pPtr);
        if (!zeroPolyp(init)) { return j; } 
    }
    return j;
}

/**
 * Specialized regular chain structure for the bivariate solver.
 */
regular_chain2 *EX_RegularChain2_Init(preFFTRep *f1, preFFTRep *f2) {
    regular_chain2 *T;
    T = (regular_chain2 *) my_malloc(sizeof(regular_chain2));
    T->poly0 = f1;
    T->poly1 = f2;
    return T;
}

regular_chain2 *EX_RegularChain2_Copy_Init(preFFTRep *f1, preFFTRep *f2) {
    regular_chain2 *T;
    T = (regular_chain2 *) my_malloc(sizeof(regular_chain2));
    T->poly0 = (f1 != NULL) ? EX_CopyOnePoly(f1) : NULL; 
    T->poly1 = (f2 != NULL) ? EX_CopyOnePoly(f2) : NULL; 
    return T;
}

void EX_RegularChain2_Free(void *element) {
    regular_chain2 *T = (regular_chain2 *)element;
    if (T != NULL) {
        EX_FreeOnePoly(T->poly0);
        EX_FreeOnePoly(T->poly1);
        my_free(T);
    }
}

void EX_RegularChain2_Print(void *element) {
    regular_chain2 *T = (regular_chain2*) element;
    if (T == NULL) printf("NULL regular_chain2\n");
    printf("\n{");
    printPoly(T->poly0);
    printf(", ");
    printPoly(T->poly1);
    printf("}\n");
}

/**
 * Check if F is a multiple of T.
 *
 * @F, univariate polynomial of degree df
 * @T, univariate polynomial of degree dT
 * 
 * returns 0 if not a multiple of T, 1 otherwise.
 *
 * Assumptions:
 *
 * (1) df is the true degree of F
 * (2) dT is the true degree of T
 * (3) df, dT > 0, nonconstant
 *
 */
sfixn is_divided_by(sfixn dF, sfixn *F, sfixn dT, sfixn *T, 
    MONTP_OPT2_AS_GENE *pPtr) 
{
    sfixn *R, dR, result;
    if (dF < dT) return 0;
    if (DEBUG) { assert(dF >= 0 && dT > 0 && F[dF] && T[dT]); }

    R = EX_UniRem(&dR, dF, F, dT, T, pPtr);
 
    // if dR = 0, either R is zero or R is a constant.
    if (dR > 0) {
        result = 0;
    } else {
        result = (R[0] == 0) ? 1 : 0;
    }

    my_free(R);
    return result;
}

/**
 * modular_generic_solve2
 *
 * Input: F1, F2 in k[x, y] and g, h in k[x].
 * Output: regular chains (A1, B1), ..., (Ae, Be) such that
 *         V(F1, F2, g) \ V(h) = V(A1, B1) union ... union V(Ae, Be). 
 *  
 * The initials of F1 and F2 MAY NOT BE regular modulo the resultant R.
 * But one of the initials is regular modulo R. 
 *
 * Assumption:
 *
 * - mvar(F1) = mvar(F2).
 * - mdeg(F1) >= mdeg(F2).
 * - F1 and F2 are shrinked. 
 * - the resultant R of F1 and F2 in y is not zero. 
 *
 */
LinkedQueue *modular_generic_solve2(preFFTRep *F1, preFFTRep *F2, 
    preFFTRep *g, preFFTRep *h, SCUBE *scube, MONTP_OPT2_AS_GENE *pPtr) 
{
    preFFTRep *R, *init, *initF2, *poly;
    sfixn *Rs, dRs;
    sfixn *hs, dhs;
    sfixn  *Q, dQ;
    sfixn  *T, dT;
    sfixn  *G, dG;
    sfixn w, i, j, dinit;
    LinkedQueue *resQueue;

    ///////////////////////////////////////////////////////////////////////////
    // Initialization & Setup
    ///////////////////////////////////////////////////////////////////////////

    // R is the resultant of F1 and F2, a univariate polynomial in x
    R = EX_IthDthCoeff(0, 0, scube, pPtr);

    if (DEBUG) {
        assert(N(R) == 1);
        assert(N(F1) == 2);
        assert(N(F2) == 2);
        assert(N(h) == 1);
        assert(N(g) == 1);
        assert(!zeroPolyp(R));
        assert(BUSZSI(R, 1) == shrinkDeg(BUSZSI(R, 1), DAT(R), CUMI(R, 1)));
        assert(BUSZSI(h, 1) == shrinkDeg(BUSZSI(h, 1), DAT(h), CUMI(h, 1)) );
        assert(BUSZSI(g, 1) == shrinkDeg(BUSZSI(g, 1), DAT(g), CUMI(g, 1)) );
        assert(shrinkDeg(BUSZSI(F1, 2), DAT(F1), CUMI(F1, 2)) >=  
               shrinkDeg(BUSZSI(F2, 2), DAT(F2), CUMI(F2, 2)));
    }
    
    // Rs is the squarefree part of R as a plain coefficient array 
    // the degree is given by dRs
    //
    // hs is the squarefree part of h as a plain coefficient array 
    // the degree is given by dhs
    
    Rs = SquarefreePart(&dRs, BUSZSI(R, 1), DAT(R), pPtr);
    hs = SquarefreePart(&dhs, BUSZSI(h, 1), DAT(h), pPtr);

    // Q = Rs / hs, the degree is given by dQ
    // T = gcd(Q, g), the degree is given by dT
    Q = EX_UniQuo(&dQ, dRs, Rs, dhs, hs, pPtr);
    
    // handle the case that g is zero
    if (zeroPolyp(g)) {
        T = Q, dT = dQ;
    } else {
        T = EX_GCD_Uni_Wrapper(&dT, Q, dQ, DAT(g), BUSZSI(g, 1), pPtr);
        my_free(Q);
    } 

    my_free(Rs);
    my_free(hs);

    ////////////////////////////////////////////////////
    // T is either 1 or non constant
    // if T is 1 then, no regular GCD to be computed.
    ////////////////////////////////////////////////////
    if (DEBUG) assert(dT > 0);

    ///////////////////////////////////////////////////////////////////////////
    // Compute the regular GCDs
    ///////////////////////////////////////////////////////////////////////////
    w = EX_WidthInSCUBE(scube);
    i = 1;
    
    resQueue = EX_LinkedQueue_Init();
    initF2 = EX_getInitial(F2);

    while (dT > 0) {

        while (i <= w) {
            j = next_regular_subres_index(i, scube, pPtr);
            // the initial of the j-th regular subresultant
            // it must not be zero;
            if  (j < w) {
                init = EX_IthDthCoeff(j, j, scube, pPtr);
            } else {
                init = initF2;
            }

            // check if init is a multiple of T
            dinit = shrinkDeg(BUSZSI(init, 1), DAT(init), CUMI(init, 1));
            if (!is_divided_by(dinit, DAT(init), dT, T, pPtr)) break;
            ++i;
        }
        
        if (i > w) {
            EX_LinkedQueue_Enqeue(resQueue, 
                EX_RegularChain2_Init(
                    CreateUniPoly(dT, T), EX_CopyOnePoly(F1)));
            my_free(T);
            EX_FreeOnePoly(initF2);
            return resQueue;
        }

        G = EX_GCD_Uni_Wrapper(&dG, T, dT, DAT(init), dinit, pPtr);

        if (j < w)  {
            poly = EX_CopyOnePoly(EX_IthSubres(j, scube, pPtr));
        } else {
            poly = EX_CopyOnePoly(F2);
        }

        if (dG == 0) { 
            EX_LinkedQueue_Enqeue(resQueue, 
                EX_RegularChain2_Init(CreateUniPoly(dT, T), poly));
            my_free(T);
            EX_FreeOnePoly(initF2);
            return resQueue;
        }

        // Q = R / G
        Q = EX_UniQuo(&dQ, dT, T, dG, G, pPtr);
        EX_LinkedQueue_Enqeue(resQueue, 
            EX_RegularChain2_Init(CreateUniPoly(dQ, Q), poly));

        my_free(Q);
        my_free(T);
        T = G, dT = dG, i = j + 1;
    }
    EX_FreeOnePoly(initF2);
    return resQueue;
}

/**
 * Inner function for solving bivariate polynomial systems.
 *
 * @F1, bivariate polynomial in x, y
 * @F2, bivariate polynomial in x, y
 * @g,  univariate polynomial in x 
 * @pPtr, Fourier prime structure
 *
 * Assumptions:
 *
 * (1) g is non-constant, univariate 
 * (2) F1 and F2 are bivariate with positive partial degrees in y
 */
LinkedQueue *modular_solve2_inner(preFFTRep *F1, preFFTRep *F2, preFFTRep *g, 
    SCUBE *scube, MONTP_OPT2_AS_GENE *pPtr)
{
    LinkedQueue *resQueue, *resQueue2;
    preFFTRep *h1, *h2, *t1, *t2, *hpoly, *gcdpoly;
    sfixn *h, dh, dh1, dh2;
    sfixn *gcd, dgcd, dg;

    h1 = EX_getInitial(F1); 
    h2 = EX_getInitial(F2); 
    dh1 = shrinkDegUni(BUSZSI(h1, 1), DAT(h1));
    dh2 = shrinkDegUni(BUSZSI(h2, 1), DAT(h2));
    h = EX_GCD_Uni_Wrapper(&dh, DAT(h1), dh1, DAT(h2), dh2, pPtr);
    hpoly = CreateUniPoly(dh, h);
    resQueue = modular_generic_solve2(F1, F2, g, hpoly, scube, pPtr);
    
    if (dh == 0) { 
        EX_FreeOnePoly(h1);
        EX_FreeOnePoly(h2);
        EX_FreeOnePoly(hpoly);
        my_free(h);
        return resQueue; 
    }

    t1 = EX_GetPolyTail(F1);
    t2 = EX_GetPolyTail(F2);
    dg = shrinkDeg(BUSZSI(g, 1), DAT(g), CUMI(g, 1));
    gcd = EX_GCD_Uni_Wrapper(&dgcd, DAT(g), dg, h, dh, pPtr);
    gcdpoly = CreateUniPoly(dgcd, gcd);

    resQueue2 = modular_solve2(t1, t2, gcdpoly, pPtr);
    EX_LinkedQueue_Concat_1(resQueue, resQueue2);

    EX_FreeOnePoly(h1);
    EX_FreeOnePoly(h2);
    EX_FreeOnePoly(hpoly);
    EX_FreeOnePoly(t1);
    EX_FreeOnePoly(t2);
    EX_FreeOnePoly(gcdpoly);
    my_free(h);
    my_free(gcd);
    EX_LinkedQueue_Free(resQueue2, EX_RegularChain2_Free);

    return resQueue;
}

/**
 * With option to select scube type.
 */
LinkedQueue *modular_solve2_select_inner(sfixn method, preFFTRep *F1, 
    preFFTRep *F2, preFFTRep *g, SCUBE *scube, MONTP_OPT2_AS_GENE *pPtr)
{
    LinkedQueue *resQueue, *resQueue2;
    preFFTRep *h1, *h2, *t1, *t2, *hpoly, *gcdpoly;
    sfixn *h, dh, dh1, dh2;
    sfixn *gcd, dgcd, dg;

    h1 = EX_getInitial(F1); 
    h2 = EX_getInitial(F2); 
    dh1 = shrinkDegUni(BUSZSI(h1, 1), DAT(h1));
    dh2 = shrinkDegUni(BUSZSI(h2, 1), DAT(h2));
    h = EX_GCD_Uni_Wrapper(&dh, DAT(h1), dh1, DAT(h2), dh2, pPtr);
    hpoly = CreateUniPoly(dh, h);
    resQueue = modular_generic_solve2(F1, F2, g, hpoly, scube, pPtr);
    
    if (dh == 0) { 
        EX_FreeOnePoly(h1);
        EX_FreeOnePoly(h2);
        EX_FreeOnePoly(hpoly);
        my_free(h);
        return resQueue; 
    }

    t1 = EX_GetPolyTail(F1);
    t2 = EX_GetPolyTail(F2);
    dg = shrinkDeg(BUSZSI(g, 1), DAT(g), CUMI(g, 1));
    gcd = EX_GCD_Uni_Wrapper(&dgcd, DAT(g), dg, h, dh, pPtr);
    gcdpoly = CreateUniPoly(dgcd, gcd);

    resQueue2 = modular_solve2_select(method, t1, t2, gcdpoly, pPtr);
    EX_LinkedQueue_Concat_1(resQueue, resQueue2);

    EX_FreeOnePoly(h1);
    EX_FreeOnePoly(h2);
    EX_FreeOnePoly(hpoly);
    EX_FreeOnePoly(t1);
    EX_FreeOnePoly(t2);
    EX_FreeOnePoly(gcdpoly);
    my_free(h);
    my_free(gcd);
    EX_LinkedQueue_Free(resQueue2, EX_RegularChain2_Free);

    return resQueue;
}

/**
 * Add a univariate polynomial f1 to a bivariate polynomial f2,
 * resulting a bivariate polynomial.
 */
preFFTRep *add_poly_uni_bi(preFFTRep *f1, preFFTRep *f2, sfixn p) 
{
    sfixn dgs[3], i, j, dx1, dx2, dy2, sz;
    preFFTRep *result;

    if (DEBUG) { assert(N(f1) == 1 && N(f2) == 2); }
    dx1 = shrinkDeg(BUSZSI(f1, 1), DAT(f1), CUMI(f1, 1));
    dx2 = shrinkDeg(BUSZSI(f2, 1), DAT(f2), CUMI(f2, 1));
    dy2 = shrinkDeg(BUSZSI(f2, 2), DAT(f2), CUMI(f2, 2));

    dgs[0] = 0;
    dgs[1] = (dx1 > dx2) ? dx1 : dx2;
    dgs[2] = dy2;
    
    if (DEBUG) assert(dy2 > 0);
    result = EX_InitOnePoly(2, dgs);

    sz = dgs[1] + 1;
    for (i = 0; i <= dy2; ++i) {
        for (j = 0; j <= dx2; ++j) {
            (result->data)[i * sz + j] = (f2->data)[i * dx2 + j];
        }
    }

    for (i = 0; i <= dx1; ++i) {
        (result->data)[i] = AddMod((result->data)[i], (f1->data)[i], p);
    }

    return EX_NormalizePoly(result);
}

/**
 * Handle corner cases for bivariate solving.
 * 
 * Assumptions:
 *
 * (1) F1 is a univariate or bivariate polynomial
 * (2) F2 is a univariate or bivariate polynomial
 * (3) g is a univariate polynomial 
 */
LinkedQueue *modular_solve2(preFFTRep *F1, preFFTRep *F2, preFFTRep *g, 
    MONTP_OPT2_AS_GENE *pPtr)
{
    sfixn d1, d2, d3, k;
    SCUBE *scube;
    preFFTRep *R, *gcdpoly, *F3, *zpoly, *Sk; 
    sfixn *gcd, *gcd2, dgcd, dgcd2;
    LinkedQueue *resQueue = NULL;

    // the case that g is nonzero constant
    if (!zeroPolyp(g) && constantPolyp(g)) {
        if (DEBUG) printf("g is a nonzero constant\n");
        return EX_LinkedQueue_Init();
    }
    
    // the case that both F1 and F2 are univariate polynomials in x
    if (N(F1) == N(F2) && N(F2) == 1) { 
        if (DEBUG) printf("both F1 and F2 are univariate polynomials\n");
        d1 = shrinkDeg(BUSZSI(F1, 1), DAT(F1), CUMI(F1, 1));
        d2 = shrinkDeg(BUSZSI(F2, 1), DAT(F2), CUMI(F2, 1));

        resQueue = EX_LinkedQueue_Init();
        gcd = EX_GCD_Uni_Wrapper(&dgcd, DAT(F1), d1, DAT(F2), d2, pPtr);
        if (dgcd == 0) {
            my_free(gcd);
            return resQueue;  
        } 

        if (zeroPolyp(g)) {
            gcdpoly = CreateUniPoly(dgcd, gcd);
            EX_LinkedQueue_Enqeue(resQueue, 
                EX_RegularChain2_Init(gcdpoly, NULL));
            my_free(gcd);
            return resQueue;
        }
        
        d3 = shrinkDeg(BUSZSI(g, 1), DAT(g), CUMI(g, 1));
        gcd2 = EX_GCD_Uni_Wrapper(&dgcd2, gcd, dgcd, DAT(g), d3, pPtr);
        
        if (dgcd2 == 0) {
            my_free(gcd);
            my_free(gcd2);
            return resQueue;  
        }

        gcdpoly = CreateUniPoly(dgcd2, gcd2);
        EX_LinkedQueue_Enqeue(resQueue, EX_RegularChain2_Init(gcdpoly, NULL));

        my_free(gcd);
        my_free(gcd2);
        return resQueue;
    }

    if (N(F1) == 1) { 
        if (DEBUG) printf("F1 is a univariate polynomial\n");
        F3 = add_poly_uni_bi(F1, F2, pPtr->P);
        resQueue = modular_solve2(F3, F2, g, pPtr);
        EX_FreeOnePoly(F3);
        return resQueue;
    }

    if (N(F2) == 1) { 
        if (DEBUG) printf("F2 is a univariate polynomial\n");
        F3 = add_poly_uni_bi(F2, F1, pPtr->P);
        resQueue = modular_solve2(F1, F3, g, pPtr);
        EX_FreeOnePoly(F3);
        return resQueue;
    }
    
    // The case both F1 and F2 are bivariate
    if (DEBUG) printf("F1 and F2 are bivariate polynomials\n");
    d1 = shrinkDeg(BUSZSI(F1, 2), DAT(F1), CUMI(F1, 2));
    d2 = shrinkDeg(BUSZSI(F2, 2), DAT(F2), CUMI(F2, 2));
    if (d1 < d2) { return modular_solve2(F2, F1, g, pPtr); }
    
    // printf("d1 = %d, d2 = %d\n", d1, d2);
    // EX_Poly_Print(F1);
    // EX_Poly_Print(F2);
    
    // build the scube
    scube = EX_SubResultantChainOpt(F1, F2, 2, pPtr);
    R = EX_IthDthCoeff(0, 0, scube, pPtr);
    if (zeroPolyp(R)) { 
        if (DEBUG) printf("The resultant is zero\n"); 
        // S_k is the last nonzero regular subresultant of F1 and F2;
        k = next_regular_subres_index(1, scube, pPtr);
        zpoly = CreateZeroPoly();

        if (k == EX_WidthInSCUBE(scube)) {
            resQueue = modular_solve2(F2, g, zpoly, pPtr);
        } else {
            Sk = EX_IthSubres(k, scube, pPtr);
            resQueue = modular_solve2(Sk, g, zpoly, pPtr);
        }
        EX_SCUBE_Free(scube);
        EX_FreeOnePoly(zpoly);
        return resQueue;
    }

    // generic case
    resQueue = modular_solve2_inner(F1, F2, g, scube, pPtr);
    //EX_SCUBE_Print(scube);
    EX_SCUBE_Free(scube);
    return resQueue;
}

/**
 * With option to select scube type
 */
LinkedQueue *modular_solve2_select(sfixn method, preFFTRep *F1, preFFTRep *F2,
    preFFTRep *g, MONTP_OPT2_AS_GENE *pPtr)
{
    sfixn d1, d2, d3, k;
    SCUBE *scube;
    preFFTRep *R, *gcdpoly, *F3, *zpoly, *Sk; 
    sfixn *gcd, *gcd2, dgcd, dgcd2;
    LinkedQueue *resQueue;

    // the case that g is nonzero constant
    if (!zeroPolyp(g) && constantPolyp(g)) {
        if (DEBUG) printf("g is a nonzero constant\n");
        return EX_LinkedQueue_Init();
    }
    
    // the case that both F1 and F2 are univariate polynomials in x
    if (N(F1) == N(F2) && N(F2) == 1) { 
        if (DEBUG) printf("both F1 and F2 are univariate polynomials\n");
        d1 = shrinkDeg(BUSZSI(F1, 1), DAT(F1), CUMI(F1, 1));
        d2 = shrinkDeg(BUSZSI(F2, 1), DAT(F2), CUMI(F2, 1));

        resQueue = EX_LinkedQueue_Init();
        gcd = EX_GCD_Uni_Wrapper(&dgcd, DAT(F1), d1, DAT(F2), d2, pPtr);
        if (dgcd == 0) {
            my_free(gcd);
            return resQueue;  
        } 

        if (zeroPolyp(g)) {
            gcdpoly = CreateUniPoly(dgcd, gcd);
            EX_LinkedQueue_Enqeue(resQueue, 
                EX_RegularChain2_Init(gcdpoly, NULL));
            my_free(gcd);
            return resQueue;
        }
        
        d3 = shrinkDeg(BUSZSI(g, 1), DAT(g), CUMI(g, 1));
        gcd2 = EX_GCD_Uni_Wrapper(&dgcd2, gcd, dgcd, DAT(g), d3, pPtr);
        
        if (dgcd2 == 0) {
            my_free(gcd);
            my_free(gcd2);
            return resQueue;  
        }

        gcdpoly = CreateUniPoly(dgcd2, gcd2);
        EX_LinkedQueue_Enqeue(resQueue, EX_RegularChain2_Init(gcdpoly, NULL));

        my_free(gcd);
        my_free(gcd2);
        return resQueue;
    }

    if (N(F1) == 1) { 
        if (DEBUG) printf("F1 is a univariate polynomial\n");
        F3 = add_poly_uni_bi(F1, F2, pPtr->P);
        resQueue = modular_solve2_select(method, F3, F2, g, pPtr);
        EX_FreeOnePoly(F3);
        return resQueue;
    }

    if (N(F2) == 1) { 
        if (DEBUG) printf("F2 is a univariate polynomial\n");
        F3 = add_poly_uni_bi(F2, F1, pPtr->P);
        resQueue = modular_solve2_select(method, F1, F3, g, pPtr);
        EX_FreeOnePoly(F3);
        return resQueue;
    }
    
    // the case both F1 and F2 are bivariate
    if (DEBUG) printf("F1 and F2 are bivariate polynomials\n");
    d1 = shrinkDeg(BUSZSI(F1, 2), DAT(F1), CUMI(F1, 2));
    d2 = shrinkDeg(BUSZSI(F2, 2), DAT(F2), CUMI(F2, 2));
    if (d1 < d2) { return modular_solve2_select(method, F2, F1, g, pPtr); }
    
    // build the scube with various types of code
    // method == 0 ==> gpu_fft + subprodtree
    // method == 1 ==> cpu_fft + subprodtree
    scube = EX_SubResultantChainSelect(method, F1, F2, 2, pPtr);

    R = EX_IthDthCoeff(0, 0, scube, pPtr);
    if (zeroPolyp(R)) {
        if (DEBUG) printf("The resultant is zero\n"); 
        // S_k is the last nonzero regular subresultant of F1 and F2;
        k = next_regular_subres_index(1, scube, pPtr);
        zpoly = CreateZeroPoly();

        if (k == EX_WidthInSCUBE(scube)) {
            resQueue = modular_solve2_select(method, F2, g, zpoly, pPtr);
        } else {
            Sk = EX_IthSubres(k, scube, pPtr);
            resQueue = modular_solve2_select(method, Sk, g, zpoly, pPtr);
        }
        EX_FreeOnePoly(zpoly);
        EX_SCUBE_Free(scube);
        return resQueue;
    }

    resQueue = modular_solve2_select_inner(method, F1, F2, g, scube, pPtr);
    EX_SCUBE_Free(scube);
    return resQueue;
}

/**
 * Wrapper function, exported 
 * 
 * @F1, poly in Zp[x, y]
 * @F2, poly in Zp[x, y]
 * @p, fourier prime number
 * 
 * @return, a list of regular_chain2 objects stored in a linked-queue.
 *
 * A possible output is 
 *
 * [(A_1, B_1),  (A_2, NULL), (A_3, B_3), (NULL, B_4)]
 *
 * where A_i are non-constant univariate polynomials and
 * B_i are bivariate polynomials with positive degree in y.
 */
LinkedQueue *EX_ModularSolve2(preFFTRep *F1, preFFTRep *F2, sfixn p) {
    MONTP_OPT2_AS_GENE prime;
    MONTP_OPT2_AS_GENE *pPtr = &prime;
    preFFTRep *g = CreateZeroPoly();
    LinkedQueue *resQueue = NULL;
    EX_MontP_Init_OPT2_AS_GENE(pPtr, p);
    //EX_Poly_Print(F1);
    //EX_Poly_Print(F2);
    resQueue = modular_solve2(F1, F2, g, pPtr);
    EX_Poly_Free(g);
    return resQueue;
}
///////////////////////////END OF FILE/////////////////////////////////////////

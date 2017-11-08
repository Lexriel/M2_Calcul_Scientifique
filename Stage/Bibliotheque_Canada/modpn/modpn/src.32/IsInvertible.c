/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#include "IsInvertible.h"

#ifdef WINDOWS
#define TRYNOPATCHD 1
#else
#ifdef MAC64
#define TRYNOPATCHD 1
#endif
#endif

extern int Interrupted;

#ifdef TRYNOPATCHD

// output chain in the pairs are all reduced and monic.
LinkedQueue *isInvertible_zeroDim(preFFTRep *poly, TriSet *ts, 
    MONTP_OPT2_AS_GENE *pPtr) 
{
    int invertibility;
    sfixn M, N, dA, dB, dG, *G, dQ, *Q, dnewQ;
    preFFTRep *Gpoly, *newGpoly, *Qqpoly, *newQqpoly, *tmppoly, *newpoly, *ts_M, *res, 
        *gcd, *new_ts_M, *Qpoly, *thenewpoly;
    LinkedQueue *resQueue, *newQueue, *gcdQueue;
    TriSet *zeroChain, *cofactorChain=NULL, *newChain, *newts, *tmpTs;
    SCUBE *scube;
    RegularPair *pair, *gcdPair;
    int bool1, bool2;
    LinkedQueue* tmpResQueue;

    signal(SIGINT,catch_intr);
    if(Interrupted==1){ return NULL; }
       
    resQueue=EX_LinkedQueue_Init();
    M = N(poly);
    N = N(ts);
    
    // #########################
    // # Normalizing the input #
    // #########################
    
    tmppoly=EX_NormalizePoly(poly);
    invertibility=MonicizeTriSet_1(N, ts, pPtr);
    if(invertibility==-1){
        printf("Error: input triangular set can not be normalized!\n");
        fflush(stdout);
        Interrupted=1;
        return NULL;
    }
    
    newpoly = EX_EY_ForNormalForm(tmppoly, ts, pPtr);
    EX_FreeOnePoly(tmppoly);
    
    // #################
    // # Trivial cases #
    // #################
    if(constantPolyp(newpoly)==1){
        EX_LinkedQueue_Enqeue(resQueue, EX_RegularPair_Init(newpoly, EX_CopyOneTriSet(ts)));
        return resQueue;
    }

    //##############
    //# Setting up #
    //##############
    M = N(newpoly);
    ts_M = ELEMI(ts, M);

    //###################
    //# Univariate case #
    //###################
    if (M==1){
        dA = shrinkDegUni(BUSZSI(newpoly, M), DAT(newpoly));
        dB = shrinkDegUni(BUSZSI(ts_M, M), DAT(ts_M));
        G = EX_GCD_UNI(&dG, DAT(newpoly), dA, DAT(ts_M), dB, pPtr);
        if (dG == 0){
            my_free(G);
            EX_LinkedQueue_Enqeue(resQueue, EX_RegularPair_Init(newpoly, EX_CopyOneTriSet(ts)));
        } else{
            zeroChain = EX_ExchangeOnePoly(CreateUniPoly(dG, G), ts, pPtr);
            EX_LinkedQueue_Enqeue(resQueue, EX_RegularPair_Init(CreateZeroPoly(), zeroChain));    
            Q = EX_UniQuo(&dQ, dB, DAT(ts_M), dG, G, pPtr); 
            my_free(G);
            if(dQ>0){
                cofactorChain = EX_ExchangeOnePoly(CreateUniPoly(dQ, Q), ts, pPtr);
                tmpResQueue=isInvertible_zeroDim(newpoly, cofactorChain, pPtr);
                if(tmpResQueue !=NULL){
                    resQueue=EX_LinkedQueue_Concat_1(resQueue, tmpResQueue);
	            }
            }
            EX_FreeOnePoly(newpoly);
            EX_freeTriSet(cofactorChain);
            my_free(Q);
        }
        return resQueue;
    }
   
    bool1 = IsAllNumberCoeffs(newpoly);
    bool2 = IsAllNumberCoeffs(ts_M);
    if(bool1&&bool2){
        Gpoly = GcdAsUni(newpoly, ts_M, pPtr);
        dG =  shrinkDeg(BUSZSI(Gpoly, M), DAT(Gpoly), CUMI(Gpoly, M));
        if (dG == 0){
            EX_FreeOnePoly(Gpoly);
            EX_LinkedQueue_Enqeue(resQueue, EX_RegularPair_Init(newpoly, EX_CopyOneTriSet(ts)));
        } else{
            newGpoly=EX_NormalizePoly(Gpoly);
            zeroChain = EX_ExchangeOnePoly(newGpoly, ts, pPtr);
            EX_LinkedQueue_Enqeue(resQueue, EX_RegularPair_Init(CreateZeroPoly(), zeroChain));    
            Qqpoly = QuoAsUni(ts_M, Gpoly, pPtr);
            EX_FreeOnePoly(Gpoly);
            newQqpoly=EX_NormalizePoly(Qqpoly);
            dQ = shrinkDeg(BUSZSI(Qqpoly, N(Qqpoly)), DAT(Qqpoly), CUMI(Qqpoly, N(Qqpoly)));
            EX_FreeOnePoly(Qqpoly);
            if(dQ>0){
                cofactorChain = EX_ExchangeOnePoly(newQqpoly, ts, pPtr);
                tmpResQueue=isInvertible_zeroDim(newpoly, cofactorChain, pPtr);
                if(tmpResQueue != NULL){
                    resQueue=EX_LinkedQueue_Concat_1(resQueue, tmpResQueue);
	            }
            }
            EX_FreeOnePoly(newpoly);
            EX_freeTriSet(cofactorChain);
        }
        return resQueue;
    }
    // #########################################################
    // # Case where either p_rc or rc_v is not univariate in v #
    // #########################################################
    scube = EX_SubResultantChain(newpoly, ts_M, M, pPtr);
    res = EX_ResultantFromChain(scube, pPtr);
    newts = EX_getLowerTriSet(M-1, ts);
    newQueue = isInvertible_zeroDim(res, newts, pPtr);
    EX_freeTriSet(newts);
    EX_FreeOnePoly(res);
    while  ( ! EX_LinkedQueue_IsEmpty(newQueue)){
        if(Interrupted==1) {
            EX_SCUBE_Free(scube);
            EX_LinkedQueue_Free(newQueue, EX_RegularPair_Free);
            EX_FreeOnePoly(newpoly);
	        return resQueue; 
        }
        pair=(RegularPair *)EX_LinkedQueue_Deqeue(newQueue);
        if (zeroPolyp(pair->poly)){
            gcdQueue = EX_RegularGcd(newpoly, ts_M, pair->ts, scube, pPtr);
            while (! EX_LinkedQueue_IsEmpty(gcdQueue)){
                gcdPair=(RegularPair *)EX_LinkedQueue_Deqeue(gcdQueue);
                gcd = EX_EY_Normalize(gcdPair->poly, gcdPair->ts, pPtr);
                new_ts_M = EX_EY_Normalize(ts_M, gcdPair->ts, pPtr);
	            zeroChain= EX_MergeTriSet(N, M, gcd, ts, gcdPair->ts, pPtr);
	            Qpoly = EX_QuoMulti(new_ts_M, gcd, M, pPtr, 0);
                EX_FreeOnePoly( gcd );
                EX_FreeOnePoly( new_ts_M );

                EX_LinkedQueue_Enqeue(resQueue, EX_RegularPair_Init(CreateZeroPoly(), zeroChain));    
                if(! constantPolyp(Qpoly)){
	                newChain = EX_MergeTriSet(N, M, Qpoly, ts, gcdPair->ts, pPtr);                    
	                tmpResQueue=isInvertible_zeroDim(newpoly, newChain, pPtr);
                    if(tmpResQueue != NULL){
	 	                resQueue=EX_LinkedQueue_Concat_1(resQueue, tmpResQueue);
	                }
                    EX_freeTriSet(newChain);
	            }   
                EX_FreeOnePoly(Qpoly);
                EX_RegularPair_Free((void *)gcdPair);
	        }
            EX_LinkedQueue_Free(gcdQueue, EX_RegularPair_Free);
        }else{
            tmpTs=EX_MergeTriSet(N(ts), M, ts_M, ts, pair->ts, pPtr);
            EX_LinkedQueue_Enqeue(resQueue, EX_RegularPair_Init( EX_CopyOnePoly(newpoly), tmpTs));
        }
        EX_RegularPair_Free((void *)pair);
    }

    EX_SCUBE_Free(scube);
    EX_LinkedQueue_Free(newQueue, EX_RegularPair_Free);
    EX_FreeOnePoly(newpoly);
   return resQueue;
}

#else

LinkedQueue *EX_RegularizeInitial(preFFTRep *InPoly, TriSet *InTs, 
    MONTP_OPT2_AS_GENE *pPtr) 
{
    LinkedQueue *resQueue, *taskQueue, *tmpQueue;
    RegularPair *taskregpair, *tmpPair;
    preFFTRep  *tmpInit, *rPoly;
    resQueue = EX_LinkedQueue_Init();
    taskQueue = EX_LinkedQueue_Init();

    EX_LinkedQueue_Enqeue(taskQueue, 
        EX_RegularPair_Init(EX_CopyOnePoly(InPoly), EX_CopyOneTriSet(InTs)));

    while( !EX_LinkedQueue_IsEmpty(taskQueue) ) {
        taskregpair = (RegularPair *) EX_LinkedQueue_Deqeue(taskQueue);
        if (constantPolyp(taskregpair->poly)==1) {
            EX_LinkedQueue_Enqeue(resQueue, taskregpair);
        } else {
	        tmpInit = EX_getInitial(taskregpair->poly);
	        tmpQueue = isInvertible_zeroDim(tmpInit, taskregpair->ts, pPtr);
            EX_FreeOnePoly(tmpInit);
            while (!EX_LinkedQueue_IsEmpty(tmpQueue)){
                tmpPair = (RegularPair *)EX_LinkedQueue_Deqeue(tmpQueue);
                if (zeroPolyp(tmpPair->poly)){
		            rPoly = EX_GetPolyTail(taskregpair->poly);
                    EX_LinkedQueue_Enqeue(taskQueue, 
                        EX_RegularPair_Init(rPoly, 
                            EX_CopyOneTriSet(tmpPair->ts)));
                } else {
		            EX_LinkedQueue_Enqeue(resQueue, 
                        EX_RegularPair_Init(
                            EX_CopyOnePoly(taskregpair->poly), 
                            EX_CopyOneTriSet(tmpPair->ts)));
	            }
                EX_RegularPair_Free(tmpPair);
	        }
            EX_LinkedQueue_Free(tmpQueue, EX_RegularPair_Free);     
            EX_RegularPair_Free((void *)taskregpair);
        }
    }
    EX_LinkedQueue_Free(taskQueue, EX_RegularPair_Free);
    return resQueue;
}

///////////////////////////////////////////////////////////////////////////////
// TRDzerodim_regularizeInitial := proc(in_p, in_rc, R_rc)
// local Results, Tasks, p, p_rc, pair_p_uc, rc, task, uc, 
//     R_modpn, p_modpn, rc_modpn,  pairs_p_uc;
//     if TRDis_empty_regular_chain(in_rc, R_rc) then
//         Results := [[in_p, in_rc]];
//         Tasks := [];
//     else
//         Tasks := [[in_p, in_rc]];
//         Results := [];
//     end if;
//     while (nops(Tasks) <> 0) do
//         task := Tasks[1];
//         Tasks := Tasks[2..-1];
//         p := task[1];
//         rc := task[2];
//         if TRDis_constant(p, R_rc) then
//             Results := [op(Results), task];
//             next;
//         end if;
//         pairs_p_uc := TRDis_invertible_modpn(TRDinit(p, R_rc), rc, R_rc);
//         for pair_p_uc in  pairs_p_uc  do
//             p_rc := pair_p_uc[1];
//             uc := pair_p_uc[2];
//             if (p_rc = 0) then
//                 Tasks := [op(Tasks), [TRDtail(p, R_rc), uc]];
//             else
//                 Results := [op(Results), [p, uc]];
//             end if;
//         end do;
//     end do;
//     return (Results);
// end proc;
///////////////////////////////////////////////////////////////////////////////

LinkedQueue* isInvertible_zeroDim_Inner(preFFTRep *poly, TriSet *ts, 
    MONTP_OPT2_AS_GENE *pPtr);

/**
 * The input polynomial might not have regular initials.
 */
LinkedQueue* isInvertible_zeroDim(preFFTRep *poly, TriSet *ts, 
    MONTP_OPT2_AS_GENE *pPtr)
{
    LinkedQueue *PairList, *tmpResQueue, *resQueue;
    RegularPair *tmpPair;
    preFFTRep *nf;

    if (constantPolyp(poly) == 1) {
        resQueue = EX_LinkedQueue_Init();
        EX_LinkedQueue_Enqeue(resQueue, 
            EX_RegularPair_Init(EX_CopyOnePoly(poly), EX_CopyOneTriSet(ts)));
        return resQueue;
    }
    
    nf = EX_EY_ForNormalForm(poly, ts, pPtr);
    PairList = EX_RegularizeInitial(nf, ts, pPtr);
    EX_FreeOnePoly(nf);

    resQueue = EX_LinkedQueue_Init();
    while ( !EX_LinkedQueue_IsEmpty(PairList) ) {
        tmpPair = (RegularPair *)EX_LinkedQueue_Deqeue(PairList);
        tmpResQueue = isInvertible_zeroDim_Inner(tmpPair->poly, tmpPair->ts, pPtr);
        if (tmpResQueue != NULL) {
            resQueue = EX_LinkedQueue_Concat_1(resQueue, tmpResQueue);
        }
        EX_RegularPair_Free(tmpPair);
   }
   EX_LinkedQueue_Free(PairList, EX_RegularPair_Free);
   return resQueue;
}

// output chain in the pairs are all reduced and monic.
LinkedQueue* isInvertible_zeroDim_Inner(preFFTRep *poly, TriSet *ts, 
    MONTP_OPT2_AS_GENE *pPtr) 
{
    int invertibility;
    sfixn M, N, dA, dB, dG, *G, dQ, *Q, dnewQ;
    preFFTRep *Gpoly, *newGpoly, *Qqpoly, *newQqpoly, *tmppoly, *newpoly, 
        *ts_M, *res, *gcd, *new_ts_M, *Qpoly, *thenewpoly, *nres;
    LinkedQueue *resQueue, *newQueue, *gcdQueue;
    TriSet *zeroChain, *cofactorChain=NULL, *newChain, *newts, *tmpTs;
    SCUBE *scube;
    RegularPair *pair, *gcdPair;
    int bool1, bool2;
    LinkedQueue* tmpResQueue;

    signal(SIGINT,catch_intr);
    if (Interrupted==1) { return NULL; }

    resQueue = EX_LinkedQueue_Init();
    M = N(poly);
    N = N(ts);
 
    // #########################
    // # Normalizing the input #
    // #########################
    tmppoly = EX_NormalizePoly(poly);
    invertibility = MonicizeTriSet_1(N, ts, pPtr);
    if (invertibility == -1) {
        printf("Error: input triangular set can not be normalized!\n");
        fflush(stdout);
        Interrupted=1;
        return NULL;
    }
    newpoly = EX_EY_ForNormalForm(tmppoly, ts, pPtr);
    EX_FreeOnePoly(tmppoly);

    //#################
    //# Trivial cases #
    //#################
    if (constantPolyp(newpoly)==1) {
        EX_LinkedQueue_Enqeue(resQueue, 
            EX_RegularPair_Init(newpoly, EX_CopyOneTriSet(ts)));
        return resQueue;
    }

    //##############
    //# Setting up #
    //##############
    M = N(newpoly);
    ts_M = ELEMI(ts, M);
    
    //###################
    //# Univariate case #
    //###################
    if (M == 1) {
        dA = shrinkDegUni(BUSZSI(newpoly, M), DAT(newpoly));
        dB = shrinkDegUni(BUSZSI(ts_M, M), DAT(ts_M));
        G = EX_GCD_UNI(&dG, DAT(newpoly), dA, DAT(ts_M), dB, pPtr);

        if (dG == 0){
            my_free(G);
            EX_LinkedQueue_Enqeue(resQueue, 
                EX_RegularPair_Init(newpoly, EX_CopyOneTriSet(ts)));
        } else {
            zeroChain = EX_ExchangeOnePoly(CreateUniPoly(dG, G), ts, pPtr);
            EX_LinkedQueue_Enqeue(resQueue, 
                EX_RegularPair_Init(CreateZeroPoly(), zeroChain));
            Q = EX_UniQuo(&dQ, dB, DAT(ts_M), dG, G, pPtr); 

            my_free(G);
            if (dQ > 0) {
                cofactorChain = EX_ExchangeOnePoly(CreateUniPoly(dQ, Q), ts, pPtr);
                tmppoly = EX_CopyOnePoly(newpoly);
                tmpResQueue = isInvertible_zeroDim(tmppoly, cofactorChain, pPtr);
                EX_FreeOnePoly(tmppoly);
                if (tmpResQueue != NULL) {
                    resQueue = EX_LinkedQueue_Concat_1(resQueue, tmpResQueue);
	            }
            }
            EX_FreeOnePoly(newpoly);
            EX_freeTriSet(cofactorChain);
            my_free(Q);
        }
        return resQueue;
    }

    bool1 = IsAllNumberCoeffs(newpoly);
    bool2 = IsAllNumberCoeffs(ts_M);

    // Both ts_m and newpoly are univariate polynomials
    if (bool1 && bool2) {
        Gpoly = GcdAsUni(newpoly, ts_M, pPtr);
        dG = shrinkDeg(BUSZSI(Gpoly, M), DAT(Gpoly), CUMI(Gpoly, M));
        if (dG == 0){
            EX_FreeOnePoly(Gpoly);
            EX_LinkedQueue_Enqeue(resQueue, 
                EX_RegularPair_Init(EX_CopyOnePoly(newpoly), EX_CopyOneTriSet(ts)));
        } else{
            newGpoly = EX_NormalizePoly(Gpoly);
            zeroChain = EX_ExchangeOnePoly(newGpoly, ts, pPtr);
            EX_LinkedQueue_Enqeue(resQueue, 
                EX_RegularPair_Init(CreateZeroPoly(), zeroChain));    
            Qqpoly = QuoAsUni(ts_M, Gpoly, pPtr);
            EX_FreeOnePoly(Gpoly);

            newQqpoly = EX_NormalizePoly(Qqpoly);
            dQ = shrinkDeg(BUSZSI(Qqpoly, N(Qqpoly)), DAT(Qqpoly), CUMI(Qqpoly, N(Qqpoly)));
            EX_FreeOnePoly(Qqpoly);

            if (dQ > 0) {
                cofactorChain = EX_ExchangeOnePoly(newQqpoly, ts, pPtr);
                tmpResQueue = isInvertible_zeroDim(newpoly, cofactorChain, pPtr);
                if (tmpResQueue != NULL){
                    resQueue = EX_LinkedQueue_Concat_1(resQueue, tmpResQueue);
	            }
            }

            EX_FreeOnePoly(newpoly);
            EX_freeTriSet(cofactorChain);
        }
        return resQueue;
    }

    // #########################################################
    // # Case where either p_v or rc_v is not univariate in v #
    // #########################################################
    scube = EX_SubResultantChain(newpoly, ts_M, M, pPtr);
    res = EX_ResultantFromChain(scube, pPtr);
    nres = EX_EY_ForNormalForm(res, ts, pPtr);
    newts = EX_getLowerTriSet(M - 1, ts);
    newQueue = isInvertible_zeroDim(nres, newts, pPtr);

    EX_freeTriSet(newts);
    EX_FreeOnePoly(res);
    EX_FreeOnePoly(nres);

    while ( !EX_LinkedQueue_IsEmpty(newQueue) ) {
        if (Interrupted == 1) {
            EX_SCUBE_Free(scube);
            EX_LinkedQueue_Free(newQueue, EX_RegularPair_Free);
            EX_FreeOnePoly(newpoly);
	        return resQueue; 
        }
        pair = (RegularPair *)EX_LinkedQueue_Deqeue(newQueue);
        if (zeroPolyp(pair->poly)) {
            gcdQueue = EX_RegularGcdNew(newpoly, ts_M, pair->ts, scube, pPtr);
            //gcdQueue = EX_RegularGcd(newpoly, ts_M, pair->ts, scube, pPtr);
            while (!EX_LinkedQueue_IsEmpty(gcdQueue)) {
                gcdPair = (RegularPair *)EX_LinkedQueue_Deqeue(gcdQueue);
                gcd = EX_EY_Normalize(gcdPair->poly, gcdPair->ts, pPtr);
                new_ts_M = EX_EY_Normalize(ts_M, gcdPair->ts, pPtr);
	            zeroChain = EX_MergeTriSet(N, M, gcd, ts, gcdPair->ts, pPtr);
	            Qpoly = EX_QuoMulti(new_ts_M, gcd, M, pPtr, 0);
                EX_FreeOnePoly(gcd);
                EX_FreeOnePoly(new_ts_M);
                EX_LinkedQueue_Enqeue(resQueue, 
                    EX_RegularPair_Init(CreateZeroPoly(), zeroChain)); 
                if (!constantPolyp(Qpoly)) {
	                newChain = EX_MergeTriSet(N, M, Qpoly, ts, gcdPair->ts, pPtr); 
	                tmpResQueue = isInvertible_zeroDim(newpoly, newChain, pPtr);

                    if (tmpResQueue != NULL) {
	 	                resQueue=EX_LinkedQueue_Concat_1(resQueue, tmpResQueue);
	                }
                    EX_freeTriSet(newChain);
	            }
                EX_FreeOnePoly(Qpoly);
                EX_RegularPair_Free((void *)gcdPair);
	        }
            EX_LinkedQueue_Free(gcdQueue, EX_RegularPair_Free);
        } else {
            tmpTs = EX_MergeTriSet(N(ts), M, ts_M, ts, pair->ts, pPtr);
            EX_LinkedQueue_Enqeue(resQueue, 
                EX_RegularPair_Init(EX_CopyOnePoly(newpoly), tmpTs));
        }
        EX_RegularPair_Free((void *)pair);
    }

    EX_SCUBE_Free(scube);
    EX_LinkedQueue_Free(newQueue, EX_RegularPair_Free);

    EX_FreeOnePoly(newpoly);
    return resQueue;
}
#endif

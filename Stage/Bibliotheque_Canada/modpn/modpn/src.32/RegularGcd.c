/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#include "RegularGcd.h"

///////////////////////////////////////////////////////////////////////////////
extern int Interrupted;

/**
 * Regularize a list of polynomials w.r.t a triangular set.
 */
LinkedQueue *EX_RegularizeList_1(LinkedQueue *RegQueue, 
    LinkedQueue *ToCheckQueue, TriSet *ts, MONTP_OPT2_AS_GENE *pPtr) 
{
    preFFTRep *poly;
    preFFTRep **arrayPoly;
    LinkedQueue *queue, *resQueue, *tmpResQueue, *tmpRegQueue, *ToCheckQueueCopy;
    RegularPair *pair;

    signal(SIGINT, catch_intr);
    if (Interrupted==1) { return NULL; }

    resQueue = EX_LinkedQueue_Init();
    ToCheckQueueCopy = EX_LinkedQueue_Copy(ToCheckQueue, EX_CopyPoly);
    if (EX_LinkedQueue_IsEmpty(ToCheckQueueCopy)) {
        arrayPoly = (preFFTRep **)LinkedQueue2Array(RegQueue, EX_CopyPoly);

        EX_LinkedQueue_Enqeue(resQueue, EX_RegularListPair_Init(RegQueue->count, 
            arrayPoly, EX_CopyOneTriSet(ts)));

    } else {
        poly = (preFFTRep *)EX_LinkedQueue_Deqeue(ToCheckQueueCopy);
        queue = isInvertible_zeroDim(poly, ts, pPtr);
        while( ! EX_LinkedQueue_IsEmpty(queue)) {
            pair = (RegularPair *) EX_LinkedQueue_Deqeue(queue);
            tmpRegQueue = EX_LinkedQueue_Copy(RegQueue, EX_CopyPoly);
            EX_LinkedQueue_Enqeue(tmpRegQueue, (void *)(pair->poly));
            tmpResQueue = EX_RegularizeList_1(tmpRegQueue, ToCheckQueueCopy, pair->ts, pPtr);
            EX_LinkedQueue_Free(tmpRegQueue, EX_Poly_Free);
            EX_LinkedQueue_Concat_1(resQueue, tmpResQueue);
            my_free(tmpResQueue);
            EX_freeTriSet(pair->ts);
        }
        EX_FreeOnePoly(poly);
        EX_LinkedQueue_Free(queue, EX_RegularPair_Free);
    }
    EX_LinkedQueue_Free(ToCheckQueueCopy, EX_Poly_Free);
    return resQueue;
}

/**
 * Find the smallest index next in a polynomial array such that 
 *
 * (1) ary[next] is not NULL, and
 * (2) start < next <= before
 *
 * returns -1 if all are NULL.
 */
sfixn findNextIndex(preFFTRep **ary, sfixn start, sfixn before) {
    sfixn i, next = -1;
    for(i = start + 1; i <= before; i++) {
        if (ary[i] != NULL) return i;
    }   
    return next;
}

/**
 * Interpolate the i-th subresultant from an scube.
 */
preFFTRep *ithSubresSPTree(sfixn i, SPTreeChain_t *scube, 
    MONTP_OPT2_AS_GENE *pPtr)
{
    preFFTRep *Si;
    sfixn w = scube->w;
    sfixn N = scube->dim - 1; 

    assert(i >= 0 && i < w);

    if ((scube->SPolys)[i] == NULL) {
        Si = interpIthSlice(i, N, N - 1, w, (scube->Ssize/w), scube->Sdims,
            scube->Ssize, scube->Sdata, scube->points_trees, pPtr);
        (scube->SPolys)[i] = Si;
    } else {
        Si = (scube->SPolys)[i];
    }
    return Si;
}

preFFTRep *ithSubres(sfixn i, SCUBE *scube, MONTP_OPT2_AS_GENE *pPtr) {
    assert(scube != NULL);
    if (scube->cType == TreeChain) {
        return ithSubresSPTree(i, (scube->cPtr).sptPtr, pPtr);
    } else if (scube->cType == DftChain) {
        // Todo
    }
    return NULL;
}

/**
 * Interpolate the i-th principle subresultant from an scube. That is, 
 * the cofficient of S_i in X^i. 
 */
preFFTRep *ithPsubresSPTree(sfixn i, SPTreeChain_t *scube, 
    MONTP_OPT2_AS_GENE *pPtr)
{
    preFFTRep *psi, *r;
    sfixn w = scube->w;
    sfixn N = scube->dim - 1; 
    sfixn tmp;

    assert(i >= 0 && i < w);
    if ((scube->SLcs)[i] == NULL) {
        tmp = scube->Sdims[N];
        scube->Sdims[N] = 1;

        r = interpIthDthSlice(i, i, N, N - 1, w, scube->Ssize/w/w, scube->Sdims,
            scube->Ssize, scube->Sdata, scube->points_trees, pPtr);

        scube->Sdims[N] = tmp;
        psi = EX_NormalizePoly(r);
        EX_FreeOnePoly(r);

        (scube->SLcs)[i] = psi;
    } else {
        psi = (scube->SLcs)[i];
    }

    return psi;
}

preFFTRep *ithPsubres(sfixn i, SCUBE *scube, MONTP_OPT2_AS_GENE *pPtr) {
    assert(scube != NULL);
    if (scube->cType == TreeChain) {
        return ithPsubresSPTree(i, (scube->cPtr).sptPtr, pPtr);
    } else if (scube->cType == DftChain) {
        // Todo
    }
    return NULL;
}

sfixn widthInSCUBE(SCUBE *scube) {
    assert(scube != NULL);
    if (scube->cType == TreeChain) {
        return ((scube->cPtr).sptPtr)->w;
    } else if (scube->cType == DftChain) {
        return ((scube->cPtr).dftPtr)->w;
    } else {
        printf("Not implemented\n");
        return 0;
    }
}

preFFTRep *ithDthCoeffSPTree(sfixn i, sfixn d, SPTreeChain_t *scube, 
    MONTP_OPT2_AS_GENE *pPtr) 
{
    preFFTRep *cid, *r;
    sfixn w = scube->w;
    sfixn N = scube->dim - 1; 
    sfixn tmp;

    assert(i >= 0 && i < w);
    assert(d >= 0 && d <= i);

    tmp = scube->Sdims[N];
    scube->Sdims[N] = 1;
    r = interpIthDthSlice(i, d, N, N - 1, w, scube->Ssize/w/w, scube->Sdims,
        scube->Ssize, scube->Sdata, scube->points_trees, pPtr);
    scube->Sdims[N] = tmp;

    cid = EX_NormalizePoly(r);
    EX_FreeOnePoly(r);

    return cid;
}

/**
 * Interpolate the coeff(S_i, y^d) from the SCUBE.
 **/
preFFTRep *ithDthCoeff(sfixn i, sfixn d, SCUBE *scube, MONTP_OPT2_AS_GENE *pPtr)
{
    assert(scube != NULL);
    if (scube->cType == TreeChain) {
        return ithDthCoeffSPTree(i, d, (scube->cPtr).sptPtr, pPtr);
    } else {
        printf("Not implemented\n");
        return 0;
    }
}

/**
 * Find largest index i such that S_i is not in sat(ts). We assume that ts is 
 * a zero-dimensional regular chain. Hence only check the normal form of S_i
 * is zero or not mudolo ts.
 *
 * Return -1 if all resubresultant are in sat(ts), otherwise returns 
 * the smallest index i such S_i \notin sat(ts).
 */
sfixn startingNonzeroIndex(SCUBE *scube, TriSet *ts, MONTP_OPT2_AS_GENE *pPtr) 
{
    preFFTRep *cid, *nf;
    sfixn i, d, w = widthInSCUBE(scube);

    for (i = 0; i < w; ++i) {
        for (d = i; d >= 0; --d) { 
            cid = ithDthCoeff(i, d, scube, pPtr);
            if (DEBUG) assert(cid != NULL);
            assert(cid->N <= ts->N);
            nf = EX_EY_ForNormalForm(cid, ts, pPtr);
            EX_FreeOnePoly(cid);
            if (!zeroPolyp(nf)) { 
                EX_FreeOnePoly(nf);
                return i;
            }
            EX_FreeOnePoly(nf);
        }
    }

    return -1;
}

/**
 * Compute the regular GCD of two polynomials w.r.t a zero-dim regular chain.
 * 
 * @f1, polynomial in (K[X])[y]
 * @f2, polynomial in (K[X])[y]
 * @ts, regular chain in K[X]
 * @scube, a subresultant chain of f1 and f2 in y.
 *
 * Assume that both init(f1) and init(f2) are regular modulo ts.
 */
LinkedQueue *EX_RegularGcdNew(preFFTRep *f1, preFFTRep *f2, TriSet *ts, 
    SCUBE *scube, MONTP_OPT2_AS_GENE *pPtr) 
{
    sfixn N, d1, d2, i, d, j, k, w;
    LinkedQueue *resQueue;  
    LinkedQueue *taskQueue;  
    LinkedQueue *candQueue;
    LinkedQueue *regQueue;

    RegularPair *pair;
    preFFTRep *sid, *sk;
    TaskPair2 *taskpair2;
    TaskPair *taskpair;
    TaskPair *candpair;

    signal(SIGINT, catch_intr);
    if (Interrupted == 1) { return NULL; }

    N = N(f1);
    assert(N == N(f2));

    d1 = shrinkDeg(BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    d2 = shrinkDeg(BUSZSI(f2, N), DAT(f2), CUMI(f2, N));

    if (d1 < d2) { return EX_RegularGcdNew(f2, f1, ts, scube, pPtr); }
 
    // Initialize the result queue
    resQueue = EX_LinkedQueue_Init();

    // Find the smallest index i such that S_i is not in sat(ts).
    i = startingNonzeroIndex(scube, ts, pPtr);
    // if i = -1 then all subresultants are zero modulo ts.
    // Hence directly return pair [f2, ts] as the output.
    if (i == -1) {
        EX_LinkedQueue_Enqeue(resQueue, 
            EX_RegularPair_Init(EX_CopyOnePoly(f2), EX_CopyOneTriSet(ts)));
        return resQueue;
    }

    // We know i >= 0 and there exists a nonzero subresultant modulo ts.
    taskQueue = EX_LinkedQueue_Init();  
    EX_LinkedQueue_Enqeue(taskQueue, EX_TaskPair2_Init(i, i, EX_CopyOneTriSet(ts)));

    // Starts to build the candidates queue. The procedure is the following
    //
    // Tasks = {[i, i, ts]}
    //
    // While nops(Tasks) > 0 do
    //    Take and remove an item [i, d, C] from Tasks
    //    sid = coeff(Si, y^d)
    //    if (i = w) {
    //         Results = Results U {[f2, C]};
    //         next;
    //    } 
    //    for D in Regularize(sid, C) do
    //        if sid is in sat(D) then
    //           if (d = 0) {
    //              Tasks = Tasks U {i + 1, i + 1, D}
    //           } else {
    //              Tasks = Tasks U {i, d - 1, D}
    //           }
    //        else
    //           Candidate = Candidate U {i, D}
    //        end if
    //     end for 
    // end while
    //
   
    w = widthInSCUBE(scube);
    if (DEBUG) assert(w == d2);

    candQueue = EX_LinkedQueue_Init();
    while (!EX_LinkedQueue_IsEmpty(taskQueue)) {
        taskpair2 = (TaskPair2 *)EX_LinkedQueue_Deqeue(taskQueue);
        i = taskpair2->i;
        d = taskpair2->d;
        if (i == d2) {
            EX_LinkedQueue_Enqeue(resQueue, 
                EX_RegularPair_Init(EX_CopyOnePoly(f2), 
                    EX_CopyOneTriSet(taskpair2->ts)));
            EX_TaskPair2_Free(taskpair2);
            continue;
        }

        if (DEBUG) assert(i >= 0 && i < w && d >= 0 && d <= i);
        sid = ithDthCoeff(i, d, scube, pPtr);

        regQueue = isInvertible_zeroDim(sid, taskpair2->ts, pPtr);

        while (!EX_LinkedQueue_IsEmpty(regQueue)) {
            pair = (RegularPair *) EX_LinkedQueue_Deqeue(regQueue);
            if (zeroPolyp(pair->poly)) {
                if (d == 0) {
                    EX_LinkedQueue_Enqeue(taskQueue, EX_TaskPair2_Init(i + 1, i + 1, 
                            EX_CopyOneTriSet(pair->ts)));
                } else {
                    EX_LinkedQueue_Enqeue(taskQueue, EX_TaskPair2_Init(i, d - 1, 
                        EX_CopyOneTriSet(pair->ts)));
                }
            } else {
                EX_LinkedQueue_Enqeue(candQueue, 
                    EX_TaskPair_Init(i, EX_CopyOneTriSet(pair->ts)));
            }
            EX_RegularPair_Free(pair);
        }
        EX_LinkedQueue_Free(regQueue, EX_RegularPair_Free);
        EX_TaskPair2_Free(taskpair2);
        EX_FreeOnePoly(sid);
    }

    //printf("++++++++++++++++++++++++++canQueue++++++++++++++++++++\n");
    //EX_LinkedQueue_Print(candQueue, EX_TaskPair_Print);
    //printf("++++++++++++++++++++++++++resQueue++++++++++++++++++++\n");
    //EX_LinkedQueue_Print(resQueue, EX_RegularPair_Print);
    //printf("++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
    ///////////////////////////////////////////////////////////////////////////
    // The checking phrase
    //
    // while nops(Candidates) > 0 do
    //    Take and remove an item [i, C] from Candidates
    //    Tasks = {[i + 1, C]}
    //    while nop(Tasks) > 0 do
    //        Take and remove an item [j, D] out of Tasks
    //        if j = mdeg(f2) = d2 then
    //            result = result U {Si, D}
    //        else
    //            Find smallest k > j s.t. sk = coeff(Sk, y^k) not in sat(D)
    //            for E in Regularize(sk, D) do
    //               Tasks = Tasks U {[j + 1, E]}
    //            end for
    //        end if
    //   end while
    // end while
    ///////////////////////////////////////////////////////////////////////////
    while (!EX_LinkedQueue_IsEmpty(candQueue)) {
        candpair = (TaskPair *)EX_LinkedQueue_Deqeue(candQueue);
        i = candpair->index;
        taskQueue = EX_LinkedQueue_Init();
        EX_LinkedQueue_Enqeue(taskQueue, 
            EX_TaskPair_Init(i + 1, EX_CopyOneTriSet(candpair->ts)));

        while (!EX_LinkedQueue_IsEmpty(taskQueue)) {
            taskpair = (TaskPair *)EX_LinkedQueue_Deqeue(taskQueue);
            j = taskpair->index;
            if (j == d2) {
                EX_LinkedQueue_Enqeue(resQueue, 
                    EX_RegularPair_Init(
                        EX_CopyOnePoly(ithSubres(i, scube, pPtr)), 
                        EX_CopyOneTriSet(taskpair->ts)));
            } else {
                for (k = j; k < d2; ++k) {
                    // sk = coeff(Sk, y^k)
                    sk = ithPsubres(k, scube, pPtr);
                    if (zeroPolyp(sk)) { 
                        EX_LinkedQueue_Enqeue(taskQueue, 
                            EX_TaskPair_Init(j + 1, EX_CopyOneTriSet(taskpair->ts)));
                        continue;
                    } 
                    // regularize sk
                    regQueue = isInvertible_zeroDim(sk, taskpair->ts, pPtr);

                    while (!EX_LinkedQueue_IsEmpty(regQueue)) {
                       pair = (RegularPair *) EX_LinkedQueue_Deqeue(regQueue);
                       EX_LinkedQueue_Enqeue(taskQueue, 
                               EX_TaskPair_Init(j + 1, EX_CopyOneTriSet(taskpair->ts)));
                       EX_RegularPair_Free(pair);
                    }
                    EX_LinkedQueue_Free(regQueue, EX_RegularPair_Free);
                }
            }
            EX_TaskPair_Free(taskpair);
        }
        EX_LinkedQueue_Free(taskQueue, EX_TaskPair_Free);
        EX_TaskPair_Free(candpair);
    }
    return resQueue;
}

///////////////////////////////////////////////////////////////////////////////

LinkedQueue *EX_RegularGcd(preFFTRep *f1, preFFTRep *f2, TriSet *ts, 
    SCUBE *in_scube, MONTP_OPT2_AS_GENE *pPtr)
{
    LinkedQueue *taskQueue, *taskQueue2, *taskQueue3, *resQueue, *regQueue, 
        *regPolyQueue, *toCheckQueue, *regularQueue, *queue2;

    TaskPair *taskpair, *taskpair2;
    int start, start2, startbackup, next, next2, bool1, bool2, taskinx;
    sfixn N, tmp, a, b, c, d1, d2;
    preFFTRep *LC, *LC2, *poly, *TmpLC, *TmpLC2;
    RegularListPair *listpair;
    RegularPair *regpair, *regpair2;
    preFFTRep *init_f1, *init_f2;

    /////////////////////////////////////////////////////////////
    // only subproduct tree based method implemented
    // //////////////////////////////////////////////////////////
    SPTreeChain_t *scube = (in_scube->cPtr).sptPtr;
    /////////////////////////////////////////////////////////////

    signal(SIGINT, catch_intr);
    if (Interrupted==1) { return NULL; }

    N = N(f1);
    assert(N == N(f2));

    d1 = shrinkDeg(BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    d2 = shrinkDeg(BUSZSI(f2, N), DAT(f2), CUMI(f2, N));

    if (d1 < d2) { return EX_RegularGcd(f2, f1, ts, in_scube, pPtr); }

    init_f1 = EX_getInitial(f1);
    init_f2 = EX_getInitial(f2);
    resQueue = EX_LinkedQueue_Init();
    taskQueue = EX_LinkedQueue_Init();  
    EX_LinkedQueue_Enqeue(taskQueue, EX_TaskPair_Init(0, EX_CopyOneTriSet(ts)));
    while (!EX_LinkedQueue_IsEmpty(taskQueue)) {

        taskpair = (TaskPair *) EX_LinkedQueue_Deqeue(taskQueue);
        start = taskpair->index;
        a = (scube->w)-1;
        b = scube->doneBfr;
        c = a;
        if (b<c) c=b;

        tmp = (scube->Sdims)[(scube->dim)-1];
        (scube->Sdims)[(scube->dim)-1] = 1;
        TmpLC = interpNextCandidateSliceLCDefective(&next, start, N, N-1, scube->w, 
            ((scube->Ssize)/(scube->w))/(scube->w), scube->Sdims, 
            scube->Ssize, scube->Sdata, scube->points_trees, pPtr);

        (scube->Sdims)[(scube->dim)-1] = tmp;

        if(TmpLC != NULL ){
            LC = EX_NormalizePoly(TmpLC);
            EX_FreeOnePoly(TmpLC);
        } else { next = -1; }

        if (next ==-1){
            next = findNextIndex(scube->SLcs, start, c);
	        if(next == -1){
	            tmp = (scube->Sdims)[(scube->dim)-1];
	            (scube->Sdims)[(scube->dim)-1] = 1;
	            TmpLC = interpNextCandidateSliceLC(&next, start, N, N-1, scube->w, 
                    ((scube->Ssize)/(scube->w))/(scube->w), scube->Sdims, 
                    scube->Ssize, scube->Sdata, scube->points_trees, pPtr);
	        
                (scube->Sdims)[(scube->dim)-1] = tmp;

                if(TmpLC != NULL ){
		            LC = EX_NormalizePoly(TmpLC);
		            EX_FreeOnePoly(TmpLC);
		            (scube->SLcs)[next] = LC;
		            (scube->doneBfr)=next;
	            } else { next = -1; }
	        } else {
	            LC = (scube->SLcs)[next];
	        }
        }

        start=next;
        if (start == -1) {
            regPolyQueue = EX_LinkedQueue_Init();
            toCheckQueue = EX_LinkedQueue_Init();
            EX_LinkedQueue_Enqeue(toCheckQueue, EX_CopyOnePoly(init_f1));
            EX_LinkedQueue_Enqeue(toCheckQueue, EX_CopyOnePoly(init_f2));
        
            regQueue = EX_RegularizeList_1(regPolyQueue, toCheckQueue, taskpair->ts, pPtr);
       
            while(! EX_LinkedQueue_IsEmpty(regQueue)){
	            listpair = (RegularListPair *) EX_LinkedQueue_Deqeue(regQueue);
                bool1 = zeroPolyp((listpair->polyList)[0]);
                bool2 = zeroPolyp((listpair->polyList)[1]);
	            if (bool1==1){
	                if (bool2==1) {
	                    EX_LinkedQueue_Enqeue(resQueue, EX_RegularPair_Init(CreateZeroPoly(), listpair->ts));
	                } else{
                        EX_LinkedQueue_Enqeue(resQueue, EX_RegularPair_Init(EX_CopyOnePoly(f2), listpair->ts));
	                }
	            } else {
	                if (bool2==1) {
                        EX_LinkedQueue_Enqeue(resQueue, EX_RegularPair_Init(EX_CopyOnePoly(f1), listpair->ts)); 
	                } else{
	                    EX_LinkedQueue_Enqeue(resQueue, EX_RegularPair_Init(EX_CopyOnePoly(f2), listpair->ts));
	                }
	            }
                EX_RegularListPair_List_Free(listpair);
            }

            EX_LinkedQueue_Free(regQueue, EX_RegularListPair_Free);
            EX_LinkedQueue_Free(regPolyQueue, EX_Poly_Free);
            EX_LinkedQueue_Free(toCheckQueue, EX_Poly_Free);
            EX_TaskPair_Free((void *)taskpair);
            continue;
        }

        regularQueue = isInvertible_zeroDim(LC, taskpair->ts, pPtr);

        while(! EX_LinkedQueue_IsEmpty(regularQueue)){
            regpair = (RegularPair *) EX_LinkedQueue_Deqeue(regularQueue);
	        if(zeroPolyp(regpair->poly)==1){
	            EX_LinkedQueue_Enqeue(taskQueue, EX_TaskPair_Init(start, regpair->ts));
	        }else{
	            startbackup=start;
                if((scube->SPolys)[start] == NULL){
                    poly = interpIthSlice(start, N, N-1, scube->w, ((scube->Ssize)/scube->w), 
                        scube->Sdims, scube->Ssize, scube->Sdata,  scube->points_trees, pPtr);

                    (scube->SPolys)[start] = poly;
                } else {
                    poly = (scube->SPolys)[start];
                }

                taskQueue2=EX_LinkedQueue_Init();
	            taskinx=0; 
                EX_LinkedQueue_Enqeue(taskQueue2, EX_TaskPair_Init(taskinx, EX_CopyOneTriSet(regpair->ts)));
                start2=startbackup;

                while(!EX_LinkedQueue_IsEmpty(taskQueue2)) {
			        tmp = (scube->Sdims)[(scube->dim)-1];
			        (scube->Sdims)[(scube->dim)-1] = 1;
			        TmpLC2 = interpNextCandidateSliceLC(&next2, start2, N, N-1, scube->w, 
                        ((scube->Ssize)/(scube->w))/(scube->w), scube->Sdims, scube->Ssize, 
                        scube->Sdata, scube->points_trees, pPtr);

			        (scube->Sdims)[(scube->dim)-1] = tmp;

			        if(TmpLC2 != NULL ){
			            LC2 = EX_NormalizePoly(TmpLC2);
			            EX_FreeOnePoly(TmpLC2);
			        } else{ next2 = -1; } 

                    start2=next2;
                    if(start2==-1){
                        while(! EX_LinkedQueue_IsEmpty(taskQueue2) ) {
                            taskpair2 = (TaskPair *) EX_LinkedQueue_Deqeue(taskQueue2);
                            taskinx--;
                            EX_LinkedQueue_Enqeue(resQueue, EX_RegularPair_Init(EX_CopyOnePoly(poly), EX_CopyOneTriSet(taskpair2->ts)));
                            EX_TaskPair_Free((void *)taskpair2);
			            } 
                    } else { 
                        taskQueue3=EX_LinkedQueue_Init();
                        while(! EX_LinkedQueue_IsEmpty(taskQueue2) ){
                            taskpair2 = (TaskPair *) EX_LinkedQueue_Deqeue(taskQueue2);
                            taskinx--;
                            queue2 = isInvertible_zeroDim(LC2, taskpair2->ts, pPtr);
                            while(! EX_LinkedQueue_IsEmpty(queue2)){
                                regpair2 = (RegularPair *) EX_LinkedQueue_Deqeue(queue2);
                                EX_LinkedQueue_Enqeue(taskQueue3, EX_TaskPair_Init(++taskinx, EX_CopyOneTriSet(regpair2->ts)));
                                EX_RegularPair_List_Free((void *)regpair2);
				            }
                            EX_TaskPair_Free((void *)taskpair2);
                            EX_LinkedQueue_Free(queue2, EX_RegularPair_Free);
			            }
                        EX_LinkedQueue_Free(taskQueue2, EX_TaskPair_Free);
                        taskQueue2=taskQueue3;
			        }
	            }
                EX_LinkedQueue_Free(taskQueue2, EX_TaskPair_Free);
	        }
	        EX_RegularPair_List_Free((void *)regpair);
        }
        EX_LinkedQueue_Free(regularQueue, EX_RegularPair_Free);
        EX_TaskPair_Free((void *)taskpair);
    }
  
    EX_LinkedQueue_Free(taskQueue, EX_TaskPair_Free);
    EX_FreeOnePoly(init_f1);
    EX_FreeOnePoly(init_f2);
    return resQueue;
}

LinkedQueue * EX_RegularGcd_Wrapped(preFFTRep *f1, preFFTRep *f2, TriSet *ts,
    sfixn M, MONTP_OPT2_AS_GENE *pPtr) 
{
    int invertibility;
    LinkedQueue *resQueue;
    SCUBE *scube;
    sfixn N, M1, M2;
    TriSet *newts;
    preFFTRep *res;
    
    // does not handle univariate case.
    assert(M>1);
    M1 = N(f1);
    M2 = N(f2);
    assert(M1 == M2);
    M = M1;
    N = N(ts);
    assert(N >= M-1);

    invertibility = MonicizeTriSet_1(N, ts, pPtr);
    if (invertibility==-1) { Interrupted=1; return NULL; }

    printf("invertibility = %d\n", invertibility);

    scube = EX_SubResultantChain(f1, f2, M, pPtr);  
    res = EX_ResultantFromChain(scube, pPtr);
    
    printf("resultant is : ");
    EX_Poly_Print(res);

    if (zeroPolyp(res)){
        newts = EX_getLowerTriSet(M-1, ts);
        //resQueue = EX_RegularGcd(f1, f2, newts, scube, pPtr);
        resQueue = EX_RegularGcdNew(f1, f2, newts, scube, pPtr);
        EX_freeTriSet(newts);
        EX_FreeOnePoly(res);
    }else{
        resQueue = EX_LinkedQueue_Init();
        EX_LinkedQueue_Enqeue(resQueue, EX_RegularPair_Init(res, NULL));
        resQueue->count = 0;
    }
    EX_SCUBE_Free(scube);
    return resQueue;
}

//////////////////////////////  END OF FILE ////////////////////////////////////

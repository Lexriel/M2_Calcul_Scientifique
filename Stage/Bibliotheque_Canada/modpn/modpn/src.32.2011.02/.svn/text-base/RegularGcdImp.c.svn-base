#include "RegularGcd.h"
/**
 * Improved implementation of regular GCD algorithms.
 * 
 * For the classic Triade regular GCD definition, there are two steps
 *
 * (1) Searching phase, candidate regular GCDs 
 * (2) Checking phase, turn candidate regular GCDs into true ones.
 *
 * Regular GCD definition could be relaxed somehow. Doing so, candidate 
 * regular GCDs are sufficient for a number of nontrivial tasks.
 *
 * - Input
 *
 * @P: polynomial in K[X][y] with main degree mdeg(P) = deg(P, y) = p
 * @Q: polynomial in K[X][y] with main degree mdeg(Q) = deg(Q, y) = q
 * @T: regular chain in K[X], zero-dimensional most of time
 * @S: SCUBE of P and Q
 *
 * - Assumptions
 * 
 * (a)  p >= q > 0,
 * (b)  init(P) and (or) init(Q) are regular modulo sat(T).
 *
 * - Output
 *
 * A sequence of pairs [G_i, T_i] for i = 1 .. s, with G_i being a 
 * regular GCD of P and Q modulo sat(T_i).
 *
 * - Implementation
 *
 * Two phases can be pseudo-coded as follows 
 *
 * (1) searching
 * 
 * tasks = {[0, 0, T]};
 * result = {};
 * candidates = {};
 *
 * while nops(tasks) > 0 do
 *     Take and remove an item [i, d, C] from tasks
 *     sid = subres_coeff(S, i, d);
 *
 *     if (i = q) then
 *         result = result union {[Q, C]};
 *         to handle the next pair;
 *     end if
 *
 *     for D in regularize(sid, C) do
 *         if sid is in sat(D) then
 *              if (d = 0) then
 *                  tasks = tasks union {[i + 1, i + 1, D]};
 *              else
 *                  tasks = tasks union {[i, d - 1, D]};
 *              end if
 *         else
 *              candidates = candidates union {[i, D]};
 *         end if
 *     end for 
 *
 * end while
 * 
 * After step (1), we feed data in the set candidates and 
 * result for step (2).
 *
 * (2) checking
 *
 * while nops(candidates) > 0 do
 *     take and remove an item [i, C] from candidates;
 *     tasks = {[i + 1, C]};
 *     
 *     while nop(tasks) > 0 do
*         take and remove an item [j, D] out of tasks
*         if j = q then
*             result = result union {[S_i, D]};
*         else
*             sjj = subres_coeff(S, j, j);
*             for E in regularize(sjj, D) do
*                 tasks = tasks union {[j + 1, E]};
*             end for
*         end if
*     end while
*
* end while
*
* We break this whole algorithm into several function calls.
*
*/

extern int Interrupted;

/**
 * Compute the candidate regular GCD of f1, f2 w.r.t ts.  
 *
 * The result will be stored in resQueue or candQueue.
 *  
 * This subroutine should be only called by EX_RegularGcdImp!!
 */
void regularGcdImp_candidates(LinkedQueue *resQueue, LinkedQueue *candQueue, 
        preFFTRep *f1, sfixn d2, preFFTRep *f2, TriSet *ts, SCUBE *scube, 
        MONTP_OPT2_AS_GENE *pPtr) 
{
    LinkedQueue *taskQueue; // tasks 
    TaskPair2 *taskpair2;       
    preFFTRep *poly; // subresultant coefficient
    sfixn i, d; 

    // initializations
    taskQueue = EX_LinkedQueue_Init();  

    EX_LinkedQueue_Enqeue(taskQueue, 
        EX_TaskPair2_Init(0, 0, EX_CopyOneTriSet(ts)));

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

        // use coeff(S_i, y^d) of f1 and f2, and then regularize it
        poly = subres_coeff(i, d, scube, pPtr);
        regQueue = isInvertible_zeroDim(poly, taskpair2->ts, pPtr);

        while (!EX_LinkedQueue_IsEmpty(regQueue)) {
            pair = (RegularPair *) EX_LinkedQueue_Deqeue(regQueue);
            if (zeroPolyp(pair->poly)) {
                if (d == 0) {
                    EX_LinkedQueue_Enqeue(taskQueue, 
                        EX_TaskPair2_Init(i + 1, i + 1, 
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
        EX_FreeOnePoly(poly);
    }

    EX_LinkedQueue_Free(taskQueue, EX_TaskPair2_Free);
}

/**
 * Checking the candidate regular GCDs of f1, f2 w.r.t ts.  
 *
 * The result will be stored in resQueue.
 *  
 * This subroutine should be only called by EX_RegularGcdImp!!
 */
void regularGcdImp_checking(LinkedQueue *resQueue, LinkedQueue *candQueue,
        sfixn d2, SCUBE *scube, MONTP_OPT2_AS_GENE *pPtr) 
{
    TaskPair    *candpair;  // a candidate pair [i, ts]
    LinkedQueue *taskQueue; // tasks 
    LinkedQueue *regQueue;
    preFFTRep *sk;
    sfixn i, j, k;

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
                // using ith subresultant
                EX_LinkedQueue_Enqeue(resQueue, 
                    EX_RegularPair_Init(
                        EX_CopyOnePoly(ithSubres(i, scube, pPtr)), 
                            EX_CopyOneTriSet(taskpair->ts)));
            } else {
                for (k = j; k < d2; ++k) {
                    // sk = coeff(S_k, y^k)
                    sk = subres_coeff(k, k, scube, pPtr);
                    if (zeroPolyp(sk)) { 
                        EX_LinkedQueue_Enqeue(taskQueue, EX_TaskPair_Init(j + 1,
                            EX_CopyOneTriSet(taskpair->ts)));
                        continue;
                    }

                    // regularize sk
                    regQueue = isInvertible_zeroDim(sk, taskpair->ts, pPtr);
                    while (!EX_LinkedQueue_IsEmpty(regQueue)) {
                        pair = (RegularPair *) EX_LinkedQueue_Deqeue(regQueue);
                        EX_LinkedQueue_Enqeue(taskQueue, 
                            EX_TaskPair_Init(j + 1, 
                                EX_CopyOneTriSet(taskpair->ts)));
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
}

LinkedQueue *EX_RegularGcdImp(preFFTRep *f1, preFFTRep *f2, TriSet *ts, 
        SCUBE *scube, MONTP_OPT2_AS_GENE *pPtr) 
{
    sfixn N, d1, d2, i, d;
    LinkedQueue *resQueue;   // result
    LinkedQueue *candQueue;  // candidates
    RegularPair *pair;

    signal(SIGINT, catch_intr);
    if (Interrupted == 1) { return NULL; }

    N = N(f1); 
    assert(N == N(f2));
    d1 = shrinkDeg(BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    d2 = shrinkDeg(BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
    if (d1 < d2) { return EX_RegularGcdImp(f2, f1, ts, scube, pPtr); }

    resQueue = EX_LinkedQueue_Init();
    candQueue = EX_LinkedQueue_Init();
    regularGcdImp_candidates(resQueue, candQueue, f1, d2, f2, ts, scube, pPtr);
    regularGcdImp_checking(resQueue, candQueue, d2, scube, pPtr);
    EX_LinkedQueue_Free(candQueue, EX_TaskPair_Free);

    return resQueue;
}


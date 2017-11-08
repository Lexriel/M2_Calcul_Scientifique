#ifndef _SOLVE_2_H_
#define _SOLVE_2_H_

#include "IteratedResultant.h"
#include "RegularGcd.h"
#include "Types.h"
#include "Factorization.h"
#include "GCD.h"
#include "HGCD.h"
#include "MPMMTS.h"
#include <time.h>

regular_chain2 *EX_RegularChain2_Init(preFFTRep *f1, preFFTRep *f2);

regular_chain2 *EX_RegularChain2_Copy_Init(preFFTRep *f1, preFFTRep *f2);

void EX_RegularChain2_Free(void *element);

void EX_RegularChain2_Print(void *element);

LinkedQueue *modular_generic_solve2(preFFTRep *F1, preFFTRep *F2, 
    preFFTRep *g, preFFTRep *h, SCUBE *scube, MONTP_OPT2_AS_GENE *pPtr);

LinkedQueue *modular_solve2_select(sfixn method, preFFTRep *F1, preFFTRep *F2,
    preFFTRep *g, MONTP_OPT2_AS_GENE *pPtr);

LinkedQueue *modular_solve2(preFFTRep *F1, preFFTRep *F2, preFFTRep *g, 
    MONTP_OPT2_AS_GENE *pPtr);

LinkedQueue *EX_ModularSolve2(preFFTRep *F1, preFFTRep *F2, sfixn p);

#endif /* solve2.h */

/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */

#ifndef __UniHensel_h
#define __UniHensel_h 

//#include "object.h"
#include "Types.h"
#include "generalFuncs.h"
#include "FDIV.h"
#include "MPMMTS.h"
#include "HashTable.h"
#include "FINTERP.h"
#include "Factorization.h"
#include <assert.h>


SLG *randomSLG(int GN, sfixn n, sfixn N);
void freeSLB(SLG * slg);
void printSLG(SLG * slg);
void printOperand( operand oper);
void printOperandG( operand oper);
void fprintOperandG( FILE *file, operand oper);

SLG *randomSLG(int GN, sfixn n, sfixn N);

SLG * getDerivOfG_OLD0(SLG *slg, operand oper2);

SLG * shrinkG(SLG *slg, int newGN);

SLG * removRedun(SLG * slg);

void printSLG_Line(SLG * slg);

void printSLG_Line_to(int n, SLG * slg);

void freeSLG(SLG *slg);

void freeSLG_nNodes(SLG *slg, int n);


SLG * newSLG(int GN);

preFFTRep * SLG2POLY( SLG * slg, TriSet *ts, TriRevInvSet * tris, sfixn N, MONTP_OPT2_AS_GENE * pPtr);

preFFTRep * SLG2POLY_ROOT( operand root, SLG * slg, TriSet *ts, TriRevInvSet * tris, sfixn N, MONTP_OPT2_AS_GENE * pPtr);


preFFTRep *
degDirevOfPoly(preFFTRep *poly, int VarNo, TriSet * ts, TriRevInvSet * tris,sfixn N, MONTP_OPT2_AS_GENE * pPtr);



void fprintSLG_Line(FILE *file, SLG *slg);

// y0 is the value to shift to.


SLG * generateSLG_example_1_F1();

SLG * generateSLG_example_1_F2();

RFuncTriSet *UniNewtonLift(int *iterAddr, sfixn Y, POLYVECTOR_SLG * vec_slg, TriSet * ts, 
                   sfixn N, MONTP_OPT2_AS_GENE * pPtr);

TriSet *EX_UniNewtonLift(int *iterAddr, sfixn Y, sfixn y0, POLYVECTOR_SLG * vec_slg, TriSet * ts, 
                   sfixn N, MONTP_OPT2_AS_GENE * pPtr);


SLG *
createOneRowOfJMatrix_For_Lifting(operand *roots, int i, POLYMATRIX *mat, 
                          POLYVECTOR_SLG *polyVec_SLG, 
                          operand *vars, TriSet *ts, TriRevInvSet * tris, 
				  sfixn N, MONTP_OPT2_AS_GENE * pPtr);



SLG *
createWholeJMatrix_For_Lifting(operand *roots, int i, POLYMATRIX *mat, 
                          POLYVECTOR_SLG *polyVec_SLG, 
                          operand *vars, TriSet *ts, TriRevInvSet * tris, 
				  sfixn N, MONTP_OPT2_AS_GENE * pPtr);


SLG*
createWholeJMatrix_For_Lifting_Hashing(operand *roots,
                          int i, POLYMATRIX *mat, 
                          POLYVECTOR_SLG *polyVec_SLG, 
                          operand *vars, TriSet *ts, TriRevInvSet * tris, 
				       sfixn N, MONTP_OPT2_AS_GENE * pPtr);

void fprintSLG_Line_to(FILE *file, int n, SLG * slg);


RFuncTriSet * RFR_for_TriSet(TriSet *ts, sfixn d, MONTP_OPT2_AS_GENE * pPtr);



TriSet *
EvalRFTriSetAtZeroForSmallestVarAndMakeThefirstOneBeY(TriSet *tsnum, TriSet *tsden, MONTP_OPT2_AS_GENE *pPtr, sfixn pt);

TriSet *
EvalRFTriSetAtZeroForSmallestVarAndMakeThefirstOneBeYAndMonicize(TriSet *tsnum, TriSet *tsden, MONTP_OPT2_AS_GENE *pPtr, sfixn pt);


void freeRFT(RFuncTriSet *srft);


TriSet *
EvalTriSetAtZeroForSmallestVarAndMakeThefirstOneBeY(TriSet *ts, MONTP_OPT2_AS_GENE *pPtr, sfixn pt);

SLG * evalVar2pt(SLG *slg, int varno, sfixn val);

TriSet *
RemoveDenominators(RFuncTriSet * rfts,  MONTP_OPT2_AS_GENE *pPtr);

void shiftPolynomial(sfixn degg, sfixn *g, sfixn degf, sfixn *f, sfixn c, MONTP_OPT2_AS_GENE * pPtr);


int
isInputSystemConsistent(sfixn N, POLYVECTOR_SLG * vec_slg,   TriSet *ts, MONTP_OPT2_AS_GENE * pPtr, sfixn y0);

void
clearTmpMarks(SLG *slg, int markNo);

operand
copy_operand_insert(SLG * slg, int * slotIn, operand oper, unsigned short int * IDsAry2);

#endif

/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#ifndef __MapleCConverter_h
#define __MapleCConverter_h 

#include "Types.h"
#include "generalFuncs.h"
#include "UniHensel.h"
#include "MPMMTS.h"
#include "FINTERP.h"
#include "LinkedList.h"
#include "IteratedResultant.h"
#include "IsInvertible.h"
#include <stdlib.h>
#include <string.h>

#ifdef _mcompile_
#include <maplec.h>
#endif

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
is_cuda_tag_enabled(); 

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
enable_cuda_tag(sfixn tag); 

////////////////////////////////////////////////////////////////////////////////

preFFTRep *createOneWrapperPoly(sfixn N, sfixn *dgs, sfixn *data);

SLG * Maple2C_DAG2DAG(int GN, sfixn *MDA);

void create_pdeg_coef_Vec(sfixn *pdegVec, sfixn *coefVec, preFFTRep *poly);

preFFTRep *inverse_create_pdeg_coef_Vec(sfixn N, sfixn *dgs, sfixn *pdegVec, sfixn*coefVec);

//ALGEB M_DECL MyIdentity( MKernelVector kv, ALGEB *args );

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
TestMDag2CDag(sfixn GN,  sfixn *MDA);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall 
#else
void
#endif
TestRecden2C(int N, sfixn  sz, sfixn *dgs,  sfixn *MDA);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
TestC2Recden(sfixn pdVdeg, sfixn *pdegVec, sfixn cVdeg, sfixn *coefVec);

void getSMPfromC(sfixn size, sfixn *buffer);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
MulPolyTFTFFTCN(sfixn N, sfixn *rdgs, sfixn rBsz, sfixn *resBuffer, sfixn *p1dgs, 
        sfixn p1sz, sfixn *p1Buffer, sfixn *p2dgs, sfixn p2sz, sfixn *p2Buffer, 
        sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn p);

int estimatePartialDegVecSize(sfixn *dgs, sfixn n);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
MulPolyTFTFFTCNC(sfixn N,  sfixn *dgs1, sfixn p1dgssz, sfixn *p1dgs, sfixn p1sz, sfixn *p1Buffer, 
        sfixn *dgs2, sfixn p2dgssz, sfixn *p2dgs, sfixn p2sz, sfixn *p2Buffer, 
        sfixn *rdgs, sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn p);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
TFTFFTUNIC(sfixn dr, sfixn *resPtr, sfixn d1, sfixn *v1Ptr, sfixn d2, sfixn *v2Ptr, sfixn p);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
FASTDIVC(sfixn degR, sfixn *RPtr, sfixn degQ, sfixn *QPtr, sfixn degA, sfixn *APtr, sfixn degB,
        sfixn *BPtr, sfixn p);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
PLAINDIVC(sfixn degR, sfixn *RPtr,sfixn degQ, sfixn *QPtr, sfixn degA, sfixn *APtr, sfixn degB,
        sfixn *BPtr, sfixn p);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
PLAINGCDUNIC(sfixn ud, sfixn *uPtr, sfixn vd, sfixn *vPtr, sfixn gd, sfixn *gcdPtr, sfixn dA,
        sfixn *APtr, sfixn dB, sfixn *BPtr, sfixn p);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
FASTGCDUNIC(sfixn ud, sfixn *uPtr, sfixn vd, sfixn *vPtr, sfixn gd, sfixn *gcdPtr, sfixn dA,
        sfixn *APtr, sfixn dB, sfixn *BPtr, sfixn p);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall 
#else
void
#endif
subProdTreeCreWrapC(sfixn h, sfixn levels, sfixn *W, sfixn *NoNodes, sfixn *Bases, sfixn totSZ,
        sfixn *data, sfixn itemNo, sfixn itemSz, sfixn p);

void subProdTreeFreeWrapC(subProdTree * tree);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
FastEvalWrapC(sfixn n, sfixn *EvalPts, sfixn degf, sfixn *fPtr, sfixn h, sfixn *W, sfixn *NoNodes,
        sfixn *Bases, sfixn *data, sfixn p);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
FastInterpWrapC(sfixn n, sfixn *InterpedPts, sfixn *EvaluatingPts, sfixn *EvaluatedPts, sfixn h,
        sfixn *W, sfixn *NoNodes, sfixn *Bases, sfixn *data, sfixn p);


// Creating the pts_tree, and return the result data back to Maple.
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
PTRTREESCRECN(sfixn *Nmp, sfixn *ptPHWDSZ, sfixn *bounds, sfixn *dims1, sfixn *dims2,
        sfixn *pts_s, sfixn *h_s, sfixn *W_s, sfixn *NoNodes_s, sfixn *Bases_s,
        sfixn *data_s, sfixn *p1dgs, sfixn p1sz, sfixn *p1Buffer, sfixn *p2dgs,
        sfixn p2sz, sfixn *p2Buffer);

// Creating the pts_tree by using the data passed from Maple.
PTS_TREE* createWrapperPTS_TREE(sfixn N, sfixn m, sfixn *bounds, sfixn pts_sSz, sfixn *pts_s,
        sfixn h_sSz, sfixn *h_s, sfixn WNB_sSz, sfixn *W_s, sfixn *NoNodes_s,
        sfixn *Bases_s, sfixn data_sSz, sfixn *data_s, sfixn p);

subProdTree* createWrapperTree(sfixn h, sfixn *W, sfixn *NoNodes, sfixn *Bases, sfixn *data);

void freeWrapperPTS_TREE(PTS_TREE *pts_tree);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
FastEvalMultiWrapCN(sfixn *Nmp, sfixn *ptPHWDSZ, sfixn *bounds, sfixn *pts_s, sfixn *h_s,
        sfixn *W_s, sfixn *NoNodes_s, sfixn *Bases_s, sfixn *data_s, sfixn *dims, 
        sfixn Esz, sfixn *E, sfixn *fdgs, sfixn fsz, sfixn *fBuffer);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
FastInterpMultiWrapCN(sfixn *Nmp, sfixn *ptPHWDSZ, sfixn *bounds, sfixn *pts_s,
        sfixn *h_s, sfixn *W_s, sfixn *NoNodes_s, sfixn *Bases_s,
        sfixn *data_s, sfixn *dims, sfixn Esz, sfixn *E, sfixn dVsz,
        sfixn *pdegVec, sfixn cVsz, sfixn *coefVec);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
SubResultantChains(sfixn N, sfixn w, sfixn Ssz, sfixn *S, sfixn *Edims1, sfixn E1sz, sfixn *E1,
        sfixn *Edims2, sfixn E2sz, sfixn*E2, sfixn p);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
InterpIthDthMultiWrapCN(sfixn *Nmp, sfixn *ptPHWDSZ, sfixn ith, sfixn dth, sfixn *bounds,
        sfixn *pts_s, sfixn *h_s, sfixn *W_s, sfixn *NoNodes_s,
        sfixn *Bases_s, sfixn *data_s, sfixn w, sfixn subslicesz,
        sfixn *subslicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
        sfixn cVsz, sfixn *coefVec);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
InterpIthMultiWrapCN(sfixn *Nmp, sfixn *ptPHWDSZ, sfixn ith, sfixn *bounds, sfixn *pts_s,
        sfixn *h_s, sfixn *W_s, sfixn *NoNodes_s, sfixn *Bases_s, sfixn *data_s,
        sfixn w, sfixn slicesz, sfixn *slicedims, sfixn Ssz, sfixn *S,
        sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
InterpNextLCMultiWrapCN(sfixn *Nmp, sfixn *ptPHWDSZ,sfixn start, sfixn *bounds, sfixn *pts_s,
        sfixn *h_s, sfixn *W_s, sfixn *NoNodes_s, sfixn *Bases_s, sfixn *data_s,
        sfixn w, sfixn subslicesz, sfixn *subslicedims, sfixn Ssz, sfixn *S,
        sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
InterpNextDefectiveLCMultiWrapCN(sfixn *Nmp, sfixn *ptPHWDSZ,sfixn start,
        sfixn *bounds, sfixn *pts_s, sfixn *h_s, sfixn *W_s,
        sfixn *NoNodes_s, sfixn *Bases_s, sfixn *data_s,
        sfixn w, sfixn subslicesz, sfixn *subslicedims, 
        sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
        sfixn cVsz, sfixn *coefVec);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
DftMultiWrapCN(sfixn *Nmp, sfixn *es, sfixn *dims, sfixn Esz, sfixn *E, sfixn *fdgs, sfixn fsz,
        sfixn *fBuffer, sfixn *rootsPtr);


#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
InvDftMultiWrapCN(sfixn *Nmp, sfixn *es, sfixn *dims, sfixn Esz, sfixn *E, sfixn dVsz,
        sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
InvDftIthDthMultiWrapCN(sfixn *Nmp, sfixn ith, sfixn dth, sfixn w, sfixn subslicesz,
        sfixn *subslicees, sfixn *subslicedims, sfixn Ssz, sfixn *S, sfixn dVsz,
        sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
InvDftIthMultiWrapCN(sfixn *Nmp, sfixn ith, sfixn w, sfixn slicesz, sfixn *slicees,
        sfixn *slicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
        sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
InvDftNextLCMultiWrapCN(sfixn *Nmp, sfixn start, sfixn w, sfixn subslicesz, sfixn *subslicees,
        sfixn *subslicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
        sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
createGoodRootsCN(sfixn N, sfixn M, sfixn *f1dgs, sfixn *f1Buffer, sfixn *f2dgs,
        sfixn *f2Buffer, sfixn *es, sfixn *dims, sfixn *rootsPtr, sfixn p);


#ifdef WINDOWS
__declspec(dllexport) void __stdcall 
#else
void
#endif
PLAINRFRUNIC(sfixn d, sfixn vd, sfixn *vPtr, sfixn gd, sfixn *gcdPtr, sfixn dA,
        sfixn *APtr, sfixn dB, sfixn *BPtr, sfixn p);


#ifdef WINDOWS
__declspec(dllexport) int __stdcall
#else
int
#endif
NewtonLiftUniCN(sfixn *outPDGVECS, sfixn *outCOEFVECS, sfixn Y, sfixn y0, sfixn N,
        sfixn *GNS, sfixn *MDAS, sfixn *TS_DGS, sfixn *inDGS, sfixn *inSIZS,
        sfixn *inCOEFS, sfixn p);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
MultiModCN(sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn N,
        sfixn *fdgs, sfixn *fBuffer, sfixn *TS_DGS, sfixn *inDGS, sfixn *inSIZS,
        sfixn *inCOEFS, sfixn p, sfixn opt);


#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
NormalizeCN(sfixn *outPDGVECS, sfixn *outCOEFVECS, sfixn N, sfixn *TS_DGS, sfixn *inDGS,
        sfixn *inSIZS, sfixn *inCOEFS, sfixn p);


#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
GetQuotientCN(sfixn N, sfixn dd, sfixn Ssz, sfixn *S, sfixn* Edims1, sfixn E1sz, sfixn *E1,
        sfixn *Edims2, sfixn E2sz, sfixn*E2, sfixn p, sfixn opt);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
ReduceCoeffCN(sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn N,
        sfixn *fdgs, sfixn *fBuffer, sfixn *TS_DGS, sfixn *inDGS, sfixn *inSIZS,
        sfixn *inCOEFS, sfixn p);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
QuotientModTriSetCN(sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn N,
        sfixn *fdgs1, sfixn *fBuffer1, sfixn *fdgs2, sfixn *fBuffer2, 
        sfixn *TS_DGS, sfixn *inDGS, sfixn *inSIZS, sfixn *inCOEFS, sfixn p);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
FastInterpRFRWrapC(sfixn *np, sfixn *InterpedPtsNum, sfixn *InterpedPtsDen, sfixn *EvaluatingPts,
        sfixn *EvaluatedPts, sfixn h, sfixn *W, sfixn *NoNodes, sfixn *Bases, sfixn *data);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
IterResOneDimCN(sfixn *outVec, sfixn M, sfixn *fdgs, sfixn *fBuffer, sfixn N, sfixn *TS_DGS,
        sfixn *inDGS, sfixn *inSIZS, sfixn *inCOEFS, sfixn bound, sfixn freeVarNo, sfixn p);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
IterResZeroDimCN(sfixn M, sfixn *fdgs, sfixn *fBuffer, sfixn N, sfixn *TS_DGS, sfixn *inDGS,
        sfixn *inSIZS, sfixn *inCOEFS,  sfixn p);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif 
isInvertableCN(sfixn N, sfixn *fdgs, sfixn *fBuffer, sfixn *TS_DGS,
        sfixn *inDGS, sfixn *inSIZS, sfixn *inCOEFS, sfixn p);


#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
IsInvertibleChainCN(sfixn *Ns, sfixn *outPolyPDGVECS, sfixn *outPolyCOEFVECS,
        sfixn *outTsPDGVECS, sfixn *outTsCOEFVECS, sfixn N, sfixn *fdgs, sfixn *fBuffer,
        sfixn *TS_DGS, sfixn *inDGS, sfixn *inSIZS, sfixn *inCOEFS, sfixn p);

TriSet* createWrapperDenseTriSet(sfixn N, sfixn * dgs);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
RegularGcdChainCN(sfixn *Ns, sfixn *outPolyPDGVECS, sfixn *outPolyCOEFVECS, sfixn *outTsPDGVECS,
        sfixn *outTsCOEFVECS, sfixn N, sfixn M, sfixn *fdgs1, sfixn *fBuffer1,
        sfixn *fdgs2, sfixn *fBuffer2, sfixn *TS_DGS, sfixn *inDGS, sfixn *inSIZS,
        sfixn *inCOEFS, sfixn p);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall 
#else
sfixn
#endif
ResultantMultivariateCN(sfixn *Nnew, sfixn *rdgs, sfixn *resBuffer, sfixn N, sfixn *p1dgs,
        sfixn *p1Buffer, sfixn *p2dgs, sfixn *p2Buffer, sfixn p);

/********************************************/ 
/*    Exporting TFT based methods 2009      */
/********************************************/ 
#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
TFTMultiWrapCN(sfixn *Nmp, sfixn *es, sfixn *dims, sfixn Esz, sfixn *E, sfixn *fdgs,
        sfixn fsz, sfixn *fBuffer, sfixn *rootsPtr);

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
InvTFTMultiWrapCN(sfixn *Nmp, sfixn *es, sfixn *dims, sfixn Esz, sfixn *E, sfixn dVsz,
        sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
SubResultantChainsTFT(sfixn N, sfixn M, sfixn w, sfixn Ssz, sfixn *S, sfixn *Edims1,
        sfixn E1sz, sfixn *E1, sfixn *Edims2, sfixn E2sz, sfixn*E2, sfixn p);

// Maple connector function of interpolate a coefficient from the sub-resultant via inverse TFT
#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
InvTFTIthDthMultiWrapCN(sfixn *Nmp, sfixn ith, sfixn dth, sfixn w, sfixn subslicesz,
        sfixn *subslicees, sfixn *subslicedims, sfixn Ssz, sfixn *S,
        sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr);

// Maple connector function of interpolate a polynomial from the sub-resultant via inverse FFT
#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
InvTFTIthMultiWrapCN(sfixn *Nmp, sfixn ith, sfixn w, sfixn slicesz, sfixn *slicees,
        sfixn *slicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
        sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr);

// Maple connector function of interpolate next leading coefficient from the sub-resultant via inverse TFT
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
InvTFTNextDefectiveLCMultiWrapCN(sfixn *Nmp, sfixn start, sfixn w, sfixn subslicesz, sfixn *subslicees,
        sfixn *subslicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
        sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
InvTFTNextLCMultiWrapCN(sfixn *Nmp, sfixn start, sfixn w, sfixn subslicesz, sfixn *subslicees,
        sfixn *subslicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
        sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr);

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
InvDftNextDefectiveLCMultiWrapCN(sfixn *Nmp, sfixn start, sfixn w,
        sfixn subslicesz, sfixn *subslicees,
        sfixn *subslicedims, sfixn Ssz, sfixn *S,
        sfixn dVsz, sfixn *pdegVec, sfixn cVsz,
        sfixn *coefVec, sfixn *rootsPtr);

/******************************************************************************/
/*                        Moving M-Array to C-Array                           */
/******************************************************************************/

#ifdef _mcompile_

/* MapleGcArray memory management */
#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
MapleGcArray_ALGEB(MKernelVector kv, ALGEB *args);


/*  TFT based SCUBE construction  */

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
SubResultantChainsTFT_ALGEB(MKernelVector kv, ALGEB *args);
// the input parameters are :
// sfixn N, sfixn M, sfixn w, sfixn Ssz, sfixn *S, sfixn *Edims1, sfixn E1sz, 
// sfixn *E1, sfixn *Edims2, sfixn E2sz, sfixn *E2, sfixn p;
// return : sfixn

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
TFTMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args);
// the input parameters are :
// sfixn *Nmp, sfixn *es, sfixn *dims, sfixn Esz, sfixn *E,
// sfixn *fdgs, sfixn fsz, sfixn *fBuffer, sfixn *rootsPtr;
// return : void

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
InvTFTMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args);
// the input parameters are :
// sfixn *Nmp, sfixn *es, sfixn *dims, sfixn Esz, sfixn *E, sfixn dVsz,
// sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr;
// return : void

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
InvTFTIthDthMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args);
// the input parameters are :
// sfixn *Nmp, sfixn ith, sfixn dth, sfixn w, sfixn subslicesz,
// sfixn *subslicees, sfixn *subslicedims, sfixn Ssz, sfixn *S,
// sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr;
// return : void

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
InvTFTIthMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args);
// the input parameters are :
// sfixn *Nmp, sfixn ith, sfixn w, sfixn slicesz, sfixn *slicees,
// sfixn *slicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
// sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr;
// return : void

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
InvTFTNextDefectiveLCMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args);
// the input parameters are :
// sfixn *Nmp, sfixn start, sfixn w, sfixn subslicesz, sfixn *subslicees,
// sfixn *subslicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
// sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr;
// return : sfixn 

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
InvTFTNextLCMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args);
// the input parameters are :
// sfixn *Nmp, sfixn start, sfixn w, sfixn subslicesz, sfixn *subslicees,
// sfixn *subslicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
// sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr;
// return : sfixn

/* FFT based SCUBE construction  */

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
SubResultantChains_ALGEB(MKernelVector kv, ALGEB *args);
// the input parameters are :
// sfixn N, sfixn w, sfixn Ssz, sfixn *S, sfixn *Edims1, sfixn E1sz,
// sfixn *E1, sfixn *Edims2, sfixn E2sz, sfixn*E2, sfixn p
// return : sfixn
// MaplePointers : S, E1, E2

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
DftMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args);
// the input parameters are :
// sfixn *Nmp, sfixn *es, sfixn *dims, sfixn Esz, sfixn *E,
// sfixn *fdgs, sfixn fsz, sfixn *fBuffer, sfixn *rootsPtr
// return : void
// MaplePointers : E

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
// the input parameters are :
// sfixn *Nmp, sfixn *es, sfixn *dims, sfixn Esz, sfixn *E,
// sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr
// return : void
// MaplePointers : E
InvDftMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args);

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
// the input parameters are :
// sfixn *Nmp, sfixn ith, sfixn dth, sfixn w,
// sfixn subslicesz, sfixn *subslicees,
// sfixn *subslicedims, sfixn Ssz, sfixn *S,
// sfixn dVsz, sfixn *pdegVec, sfixn cVsz,
// sfixn *coefVec, sfixn *rootsPtr
// return : void
// MaplePointers : S
InvDftIthDthMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args);

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
// the input parameters are :
// sfixn *Nmp, sfixn ith, sfixn w, sfixn slicesz, sfixn *slicees,
// sfixn *slicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
// sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr;
// return : void
// MaplePointers :  S
InvDftIthMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args);

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
// the input parameters are :
// sfixn *Nmp, sfixn start, sfixn w, sfixn subslicesz, sfixn *subslicees,
// sfixn *subslicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
// sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr;
// return : sfixn 
// MaplePointers : S
InvDftNextDefectiveLCMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args);

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
// the input parameters are :
// sfixn *Nmp, sfixn start, sfixn w, sfixn subslicesz, sfixn *subslicees,
// sfixn *subslicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
// sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr;
// return : sfixn
// MaplePointers : S
InvDftNextLCMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args);

/* subproduct tree based SCUBE construction  */

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
// the input parameters are :
// sfixn *Nmp, sfixn *ptPHWDSZ, sfixn *bounds, sfixn *pts_s, sfixn *h_s,
// sfixn *W_s, sfixn *NoNodes_s, sfixn *Bases_s, sfixn *data_s, sfixn *dims, 
// sfixn Esz, sfixn *E, sfixn *fdgs, sfixn fsz, sfixn *fBuffer;
// return : sfixn
// MaplePointers : E
FastEvalMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args);

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
// the input parameters are :
// sfixn *Nmp, sfixn *ptPHWDSZ, sfixn *bounds, sfixn *pts_s,
// sfixn *h_s, sfixn *W_s, sfixn *NoNodes_s, sfixn *Bases_s,
// sfixn *data_s, sfixn *dims, sfixn Esz, sfixn *E, sfixn dVsz,
// sfixn *pdegVec, sfixn cVsz, sfixn *coefVec;
// return : sfixn
// MaplePointers : E
FastInterpMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args);

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
// the input parameters are :
// sfixn *Nmp, sfixn *ptPHWDSZ, sfixn ith, sfixn dth, sfixn *bounds,
// sfixn *pts_s, sfixn *h_s, sfixn *W_s, sfixn *NoNodes_s,
// sfixn *Bases_s, sfixn *data_s, sfixn w, sfixn subslicesz,
// sfixn *subslicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
// sfixn cVsz, sfixn *coefVec;
// return : void 
// MaplePointers : S
InterpIthDthMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args);

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
// the input parameters are :
// sfixn *Nmp, sfixn *ptPHWDSZ, sfixn ith, sfixn *bounds, sfixn *pts_s,
// sfixn *h_s, sfixn *W_s, sfixn *NoNodes_s, sfixn *Bases_s, sfixn *data_s,
// sfixn w, sfixn slicesz, sfixn *slicedims, sfixn Ssz, sfixn *S,
// sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec;
// return : void
// MaplePointers : S
InterpIthMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args); 

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
// the input parameters are :
// sfixn *Nmp, sfixn *ptPHWDSZ,sfixn start, sfixn *bounds, sfixn *pts_s,
// sfixn *h_s, sfixn *W_s, sfixn *NoNodes_s, sfixn *Bases_s, sfixn *data_s,
// sfixn w, sfixn subslicesz, sfixn *subslicedims, sfixn Ssz, sfixn *S,
// sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec;
// return : sfixn
// MaplePointers : S
InterpNextLCMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args);

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
// the input parameters are :
// sfixn *Nmp, sfixn *ptPHWDSZ,sfixn start,
// sfixn *bounds, sfixn *pts_s, sfixn *h_s, sfixn *W_s,
// sfixn *NoNodes_s, sfixn *Bases_s, sfixn *data_s,
// sfixn w, sfixn subslicesz, sfixn *subslicedims, 
// sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
// sfixn cVsz, sfixn *coefVec);
// return : sfixn
// MaplePointers : S
InterpNextDefectiveLCMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args);

#endif // _mcompile_
/* EOF */
#endif

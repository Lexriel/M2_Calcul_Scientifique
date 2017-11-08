/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#include "MapleCConverter.h"

extern int Interrupted; // -->  -10
extern int PrimeError;  // --> -100
extern sfixn FP[];
extern sfixn CUDA_TAG;

double time_sfixn=0, time_var=0, time_biPlus=0, time_biSub=0, time_biProd=0,
       time_total=0, t01=0, t02=0, t03=0, t04=0, t05=0, t06=0, t07=0, t08=0, 
       time_cls=0, time_fft=0, time_mulmod=0;

// GN number of Nodes in G.
// MDA is the Maple array who encodes the polynomial dag.
// MDA must be a GN x 3 2-d array.
// num                              0    [0,  num,   not used]
// var                              1    [1,  varno, not used]
// "^"      var^num                 2    [2,  varno, exponent]
// "^"      addr^num                3    [3,  addr,  exponent]
// "+"  oprands must be addr        4    [4,  addr1, addr2   ]
// "*"  operand must be addr        5    [5,  addr1, addr2   ]
// addr                             6    [6,  addr,  not used]


// This function always returns 1. WP 
// 1 -- for good.
// 0 -- for bad.
int testPrimeNumber(sfixn p){
    int i;
    return 1;
    for (i=0; i<100; i++) { if (p == FP[i]) return 1; }
    return 0;
}

operand binaryExpansion(operand *tmpVec, sfixn e) 
{
    operand opr;
    sfixn e2;
    if ((tmpVec[e] != 0) || (e==1)) return tmpVec[e];
    e2 = e/2;
    if ((e%2) == 0) {
        tmpVec[e] = new_biProd_ini(opr, binaryExpansion(tmpVec, e2), binaryExpansion(tmpVec, e2));
    } else {
        tmpVec[e]=new_biProd_ini(opr,binaryExpansion(tmpVec, e2), binaryExpansion(tmpVec, e2+1));
    }
    return tmpVec[e];
}

int expand2Prods_varpow(SLG *slg, sfixn varno, sfixn e, int startSlot)
{
    operand opr;
    operand *tmpVec;
    int i, j=startSlot;
    if (e<0) Throw 1001;

    tmpVec = (operand *)my_calloc(e+1, sizeof(operand));
    new_var_ini(opr, varno);
    tmpVec[1] = opr;
    binaryExpansion(tmpVec, e);
    id_set(tmpVec[1], startSlot);
    (slg->Nodes)[j++] = tmpVec[1];

    for(i=2; i<=e; i++){
        if (tmpVec[i]!=NULL) {
            id_set(tmpVec[i], j);
            (slg->Nodes)[j++]=tmpVec[i]; 
        }
    }
    my_free(tmpVec);
    return j - 1 - startSlot;
}

int expand2Prods_pow(SLG *slg, sfixn loc, sfixn e, int startSlot){
    operand opr;
    operand *tmpVec;
    int i, j = startSlot;

    if (e<0) Throw 1001;

    tmpVec=(operand *) my_calloc(e+1, sizeof(operand));
    new_biPlus_ini(opr, (slg->Nodes)[0], (slg->Nodes)[loc]);   
    tmpVec[1] = opr;
    id_set(tmpVec[1], startSlot);
    binaryExpansion(tmpVec, e);
    id_set(tmpVec[1], startSlot);
    (slg->Nodes)[j++] = tmpVec[1];

    for (i=2; i<=e; i++) {
        if(tmpVec[i] != NULL) {
            id_set(tmpVec[i], j);
            (slg->Nodes)[j++] = tmpVec[i];  
        }
    }
    my_free(tmpVec);
    return j - 1 - startSlot;
}

//   num                              0
//   var                              1
//   power   var^num =                2
//   power   addr^num=                3
//    +      operands must be addr    4
//    *      operands must be addr    5
//   addr                             6


// convert the intermediate encoding of a DAG polynomial to a C-DAG polynomial.
SLG* Maple2C_DAG2DAG(int GN, sfixn *MDA)
{
    operand opr;
    SLG *slg;
    int i, j;
    int incr = 0;
    int *offsets = (int *)my_calloc(GN+1, sizeof(int));
    slg = (SLG *)my_malloc(sizeof(SLG));
    slg->GN = GN+1; // Padded a "zero" operand at the head to
                    // deal with 0 + operand.
    slg->GE = 0;
    for (j=0; j<GN*3; j+=3) 
    {
        if (MDA[j] == 2 || MDA[j] == 3) incr += logceiling(MDA[j+2]) + 1;
        if (MDA[j] == 6) incr += 1;
    }

    slg->Nodes = (operand *)my_calloc((GN+1+incr), sizeof(operand));
    new_sfixn_ini((slg->Nodes)[0], 0);
    id_set((slg->Nodes)[0], 0);
    offsets[0] = offsets[1] = 1;
    for (i=0; i<GN; i++) {
        switch (MDA[i*3]) {
	    case 0:
	        new_sfixn_ini(opr, MDA[i*3+1]);
            id_set(opr, i+offsets[i]);
            (slg->Nodes)[i+offsets[i]] = opr;
            offsets[i+1] = offsets[i];
            break;
	    case 1:
	        new_var_ini(opr, MDA[i*3+1]);
            id_set(opr, i+offsets[i]);
            (slg->Nodes)[i+offsets[i]] = opr;
            offsets[i+1] = offsets[i];
            break;
	    case 2:
	        if (1) {
	            offsets[i] += expand2Prods_varpow(slg, MDA[i*3+1], MDA[i*3+2], i+offsets[i]);
                offsets[i+1] = offsets[i];
                slg->GN = GN + offsets[i];
	        } else {  
                printf("Not done case!");
                fflush(stdout);
	            Throw (13);
                new_varpow_ini(opr, MDA[i*3+1], MDA[i*3+2]);
	            id_set(opr, i+offsets[i]);
	            (slg->Nodes)[i+offsets[i]] = opr;
                offsets[i+1] = offsets[i];
	        }
            break;
	    case 3:
            offsets[i] += expand2Prods_pow(slg, MDA[i*3+1]-1+offsets[MDA[i*3+1]-1], MDA[i*3+2], i+offsets[i]);
            offsets[i+1] = offsets[i];
            slg->GN = GN+offsets[i];
            break;
	    case 4:
	         // won't increase incr.
          	 new_biPlus_ini(opr, (slg->Nodes)[MDA[i*3+1]-1+offsets[MDA[i*3+1]-1]],
                                 (slg->Nodes)[MDA[i*3+2]-1+offsets[MDA[i*3+2]-1]]);
             id_set(opr, i+offsets[i]);
             (slg->Nodes)[i+offsets[i]] = opr;
             offsets[i+1] = offsets[i];
             break;
  	    case 5:
 	         new_biProd_ini(opr, (slg->Nodes)[MDA[i*3+1]-1+offsets[MDA[i*3+1]-1]],
                                 (slg->Nodes)[MDA[i*3+2]-1+offsets[MDA[i*3+2]-1]]);
             id_set(opr, i+offsets[i]);
             (slg->Nodes)[i+offsets[i]] = opr;
             offsets[i+1] = offsets[i];
             break;
        case 6:  
             new_biPlus_ini(opr, (slg->Nodes)[0], (slg->Nodes)[MDA[i*3+1]-1+offsets[MDA[i*3+1]-1]]);   
             id_set(opr, i+offsets[i]);
     	     (slg->Nodes)[i+offsets[i]] = opr;
             offsets[i+1] = offsets[i];
	         break;
	    default:
		     return(NULL);
	    }
    }
    my_free(offsets);
    return slg;
}

// Same as the InitOnePoly,
// except that the DAT is allocated by Maple GC.
// So freeOneMapleWrapperPoly(), do not try to free the DAT field.
void InitOneMapleWrapperPoly(preFFTRep * rPtr, sfixn N, sfixn * p1dgs, sfixn * theDAT){  
    register int j; 
    N(rPtr) = N;
    BUSZS(rPtr) = (sfixn *)my_calloc(N+1, sizeof(sfixn));
    CUTS(rPtr) = (sfixn *)my_calloc(N+1, sizeof(sfixn));
    SIZ(rPtr) = 1;
    CUM(rPtr) = (sfixn *)my_calloc(N+1, sizeof(sfixn));
    CUMI(rPtr, 1) = 1;
    for (j=1; j<=N; j++) {
        BUSZSI(rPtr, j) = p1dgs[j-1];
        CUTSI (rPtr, j) = p1dgs[j-1];   
        SIZ(rPtr) = SIZ(rPtr)*(BUSZSI(rPtr,j)+1);
        if (j>=2) {
            CUMI(rPtr, j) = CUMI(rPtr, j-1)*(BUSZSI(rPtr, j-1)+1);
        }
    }
    OFST(rPtr) = 0;
    DAT(rPtr) = theDAT;
    DFN(rPtr) = N(rPtr);
    DFSIZ(rPtr) = SIZ(rPtr);
    DEFDAT(rPtr) = DAT(rPtr);
}

void FreeOneMapleWrapperPoly(preFFTRep * x){
    if (BUSZS(x) != NULL) { my_free(BUSZS(x)); BUSZS(x) = NULL; }
    if (CUTS(x)  != NULL) { my_free(CUTS(x));   CUTS(x) = NULL; }
    if (CUM(x)   != NULL) { my_free(CUM(x));     CUM(x) = NULL; }
}

// Maple Dag -> C
#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
TestMDag2CDag(sfixn GN, sfixn *MDA){
    SLG *slg;
    slg = Maple2C_DAG2DAG(GN, MDA);
    freeSLG(slg);
}

void freeOneWrapperPoly(preFFTRep * x){
    if(BUSZS(x) != NULL) {my_free(BUSZS(x)); BUSZS(x) = NULL;}
    if(CUTS(x) != NULL) {my_free(CUTS(x)); CUTS(x) = NULL;}
    if(CUM(x) != NULL) {my_free(CUM(x)); CUM(x) = NULL;}
    // this block belongs to Maple GC.
    //if(DAT(x) !=NULL) {fConnectorModule.mwree(DAT(x)); DAT(x)=NULL;}
    my_free(x);
}

// for recurden <->C
#ifdef WINDOWS
__declspec(dllexport) void __stdcall 
#else
void
#endif
TestRecden2C(int N, sfixn sz, sfixn *dgs,  sfixn *MDA){
    preFFTRep *poly;
    FILE *file = fopen ("./TestMC2.Result",  "w");
    //fprintf(file, "N=%d, sz=%d, ", N, sz);
    //fprintVec(file, N-1, dgs);
    //fprintVec(file, sz-1, MDA);
    poly = createOneWrapperPoly(N, dgs, MDA);
    //fprintVec(file, sz-1, DAT(poly));
    fprintPoly(file, poly);
    freeOneWrapperPoly(poly);
    fclose(file);
}

void D2SConvert(sfixn *SArrSizeAddr, sfixn *SArrUsedAddr, sfixn *noItemsAddr, 
                sfixn N, sfixn *DArr, sfixn Dbase, sfixn *SArr, sfixn *SbaseAddr,
                sfixn *varExpo, sfixn *accum, sfixn level, sfixn *bufs)
{
    int i, j;
    if (DEBUG24) printf("the leve now is %ld.\n", level);
    if (level==1) {
        if (DEBUG24) printf("bufs[level] is %ld.\n", bufs[level]);
        for (i=0; i<=bufs[level]; i++) {
            if (DArr[Dbase+i] != 0) {
                SArr[(*SbaseAddr)++] = DArr[Dbase+i];
                for (j=1; j<=N; j++) {
                    SArr[(*SbaseAddr)++] = varExpo[j];
                }
                (*noItemsAddr)++;
                (*SArrUsedAddr)=(*SArrUsedAddr)+N+1;
            }
            varExpo[level]++;
        }
        return;
    }

    for (i=0; i<=bufs[level]; i++) {
        for(j=1; j<level; j++) varExpo[j] = 0; 
        D2SConvert(SArrSizeAddr, SArrUsedAddr, noItemsAddr, N, DArr, Dbase, 
                   SArr, SbaseAddr, varExpo, accum, level-1, bufs);
        Dbase += accum[level];
        varExpo[level]++;
    }
}

// return the no of item of SArr which consists qua-truple.
sfixn *DensePoly2SparseMonos(sfixn *noItemsAddr, preFFTRep *poly){
    sfixn SArrSize = SIZ(poly)*4, *SArrSizeAddr = &SArrSize;
    sfixn SArrUsed=0, *SArrUsedAddr = &SArrUsed;
    sfixn Dbase=0;
    sfixn Sbase=0, *SbaseAddr = &Sbase;
    sfixn *SArr;
    sfixn N = N(poly);
    sfixn *varExpo = (sfixn *)my_calloc(N+1, sizeof(sfixn));
    SArr = (sfixn *)my_calloc(SArrSize, sizeof(sfixn));
    
    D2SConvert(SArrSizeAddr, SArrUsedAddr, noItemsAddr, N, DAT(poly), Dbase,
               SArr, SbaseAddr, varExpo, CUM(poly), N, BUSZS(poly));
    my_free(varExpo);
    return SArr;
}

// converting Dense poly to Sparse Poly.
void TestMC3() {
    int i;
    sfixn N=3, p=1009;
    sfixn noItems=0, *noItemsAddr=&noItems;
    sfixn *dgs;
    sfixn *SArr;
    preFFTRep * poly;
    
    poly = (preFFTRep *)my_malloc(sizeof(preFFTRep));
    dgs = (sfixn *)my_calloc(N+1, sizeof(sfixn));
    for (i=1; i<=N; i++) dgs[i] = i*10;
    InitOnePoly(poly, N, dgs);
    randomPoly(poly, p);
    SArr = DensePoly2SparseMonos(noItemsAddr, poly);
 
    /* printf("the BUFS is:"); */
    /* printVec(N, BUSZS(poly)); */
    /* printf("the randPoly is:\n"); */
    /* printPoly(poly); */
    /* printf("the Sparse list is:\n"); */
    /* printVec(noItems*(N+1), SArr); */
    my_free(SArr);
    printf("no of slots needed is %ld.\n", noItems*(N+1) );
    freePoly(poly);
    my_free(dgs);
    my_free(poly);
}

void getSMPfromC(sfixn sz, sfixn *buffer) {
    TestMC3 ();
}

// Generate a random poly.
// filling the paritialDegVec, and CoeffsVec
// which are provided empty buffer. 

// the size of the PartialDegVec should be 
// E.g. f in [x,y,z], x>y>z, deg(f,x)=3, deg(f,y)=2, deg(f,z)=4.
//                              => size of the vec is 1+3*2.      
// the partial degree vector's maximum possible size.
// Only encoding the degrees. plus 2 is for the top level var, 
// and for keeping the size info of the vector.

// Just a testor, supporting the given buffer is sufficiently large
// to keep the result extracted from the random generated polynomial
// in this funciton.
#ifdef WINDOWS
__declspec(dllexport) void __stdcall 
#else
void
#endif
TestC2Recden( sfixn pdVdeg, sfixn *pdegVec, sfixn cVdeg, sfixn *coefVec){
    sfixn N = 3;
    sfixn dgs[4] = {0, 3, 2, 4};
    sfixn p = 962592769;
    preFFTRep *poly;
    sfixn loc=1, *locAddr=&loc;
    sfixn coefVecSiz=0, *coefVecSizAddr=&coefVecSiz;
    sfixn coefSizeFinal=1, *coefSizeFinalAddr=&coefSizeFinal;

    poly = (preFFTRep *)my_malloc(sizeof(preFFTRep));
    InitOnePoly(poly, N, dgs);
    srand(getSeed());
    randomSparsePoly(poly, p);
    if (N(poly)==0)  { 
        printf("in EX_getPartiaProject00-all-in-1-forMAPLElDegVec N(poly)==0, which is invalid input."); 
        fflush(stdout);
        Interrupted = 1;
        freePoly(poly);
        my_free(poly);
        return;
    }
    EX_getPartialDegVec_inner(coefVecSizAddr, N(poly), BUSZS(poly),
                              CUM(poly), DAT(poly), locAddr, pdegVec);
    loc--;
    // keeping the size of the pdegVec in pdegVec[0].
    pdegVec[0] = loc;
    // coefVec=(sfixn *)my_calloc(coefVecSiz, sizeof(sfixn));
    // keeping the size of coefVec in coefVec[0];
    // coefSizeFinal starts from 1.
    RemoveLCMinusOneAndFillCoefs(SIZ(poly), coefSizeFinalAddr, poly, coefVec,
                                 locAddr, N(poly),  pdegVec);
    RemoveMinusTwo(pdegVec);
    coefVec[0] = coefSizeFinal;

    freePoly(poly);
    my_free(poly);
}

//====================================================================
// ABOVE IS TESTING, NOW THE REAL helper functions 
//      + computational functions supporting Maple level operation
//====================================================================


// Helper function#0
// see the function  "SLG *Maple2C_DAG2DAG(int GN, sfixn *MDA)" before.
// This function decode the MDA GN x 3 array passed from Maple.
// INPUT: MDA is a GN x 3 two-dimensional array.
// OUTPUT: The C-level "*slg", straight line programming graph .
// Funciton is in this file.

// Helper function#1:
// INPUT: N, no of vars; dgs, partial degrees of Poly, data, coefficients of Poly. 
// OUTPUT: Poly.
preFFTRep *createOneWrapperPoly(sfixn N, sfixn *dgs, sfixn *data){
  
    //printf("N=%d\n", N);
    //printVec(N, dgs);
    //fflush(stdout);
    preFFTRep *poly = (preFFTRep *)my_malloc(sizeof(preFFTRep));

    int i;
    sfixn *dgs2;

    if (dgs[0]==-1) {
        dgs2 = (sfixn *)my_calloc(N, sizeof(sfixn));
        for (i=0; i<N; i++) dgs2[i] = dgs[i+1];
        InitOneMapleWrapperPoly(poly, N, dgs2, data);
        my_free(dgs2);
    } else {
        InitOneMapleWrapperPoly(poly, N, dgs, data);
    }
    return poly;
}

// pdegVec[0]+1 is size of the whole array.
// coefVec[0]+1 is size of the whole array.
// Helper function#2.
// szPdegVec, szCoefVec will be estimated by Maple function.
/**
 * create_pdeg_coef_Vec:
 * @pdegVec: (output) the partial degree vector in 2-Vector encoding.
 * @coefVec: (output) the coefficient vector in 2-Vector encoding.
 * @poly: a C-Cube polynomial.
 * 
 * To convert a C-Cube polynomial into 2-Vector encoding.
 * The 2-Vector encoding is saved in (pdegVe, coefVec).
 * Return value: 
 **/
void create_pdeg_coef_Vec(sfixn *pdegVec, sfixn *coefVec, preFFTRep *poly) {
    sfixn loc=1, *locAddr=&loc;
    sfixn coefVecSiz=0, *coefVecSizAddr=&coefVecSiz;
    sfixn coefSizeFinal=1, *coefSizeFinalAddr=&coefSizeFinal;

    EX_getPartialDegVec_inner(coefVecSizAddr, N(poly), BUSZS(poly), CUM(poly),
                              DAT(poly), locAddr, pdegVec);
    loc--;
    pdegVec[0] = loc;
    RemoveLCMinusOneAndFillCoefs(SIZ(poly), coefSizeFinalAddr, poly, coefVec,
                                 locAddr, N(poly),  pdegVec);
    coefVec[0]=coefSizeFinal-1;
    RemoveMinusTwo(pdegVec);
}

void FillCoefsfrom2VtoV(sfixn base, sfixn *coefSizeFinalAddr, preFFTRep *Ptr,
                        sfixn *coefVec, sfixn *locAddr, sfixn level, sfixn *vec)
{
    int d, d2, i, diff;

    if ((*locAddr) == 0) return;
    if (level==1) {
        d=vec[(*locAddr)--];
        if (d >= 0) {
            copyVec_0_to_d(d, DAT(Ptr)+base-CUMI(Ptr, level+1), coefVec+(*coefSizeFinalAddr));
            (*coefSizeFinalAddr) += d+1;
        }
        return;
    } else {
        d = vec[(*locAddr)--];
        if (level == N(Ptr) && d == -1 ) return;     
        if (d==-1) {
            // do nothing.
        } else {
            d2 = BUSZSI(Ptr, level) - d;
            diff = BUSZSI(Ptr, level) - d;
            base -= diff*CUMI(Ptr, level);    
            for (i=0; i<=d; i++) {
                FillCoefsfrom2VtoV(base, coefSizeFinalAddr, Ptr, coefVec, locAddr, level-1, vec);
	            base -= CUMI(Ptr, level);
            }
        }
    }
}

/**
 * estimatePartialDegVecSize:
 * @dgs: the parital degrees of some C-Cube polynomial.
 * @n: the number of variables.
 *
 * Compute the estimated size of the partial degree vector according to
 * the input dgrees 'dgs' 
 * 
 * Return value: the estimated size of the parital degree vector.
 **/
int estimatePartialDegVecSize(sfixn *dgs, sfixn n){
    int siz = 1, ac = 1, i;
    for (i=n; i>=2; i--) {
        ac *= (dgs[i]+1);
        siz += ac;
    }
    return siz + 1;
}

// The inverse function of create_pdeg_coef_Vec()
// Take the PdegVec, and ceofVec, returns the a *poly.
// BUFSZ(poly)=dgs. which need to provided (from Maple).
// N(poly)=N. 
/**
 * inverse_create_pdeg_coef_Vec:
 * @N: the number of variables.
 * @dgs: the partial degrees for the output C-Cube polynomial (pre-computed).
 * @pdegVec: the partial degree vector in the 2-Vector encoding.
 * @coefVec: the coefficient vector in the 2-vector encoding.
 * 
 * Return value: a C-Cube polynomial according to the input 2-Vector encoding 
 *              (pdegVec, coefVec).
 **/
preFFTRep *inverse_create_pdeg_coef_Vec(sfixn N, sfixn *dgs, sfixn *pdegVec, sfixn*coefVec) {
    preFFTRep *poly;
    sfixn loc=pdegVec[0], *locAddr=&loc;
    sfixn coefSizeFinal=1, *coefSizeFinalAddr=&coefSizeFinal;
    poly = (preFFTRep *)my_malloc(sizeof(preFFTRep));
    InitOnePoly(poly, N, dgs);  
    FillCoefsfrom2VtoV(SIZ(poly), coefSizeFinalAddr, poly, coefVec, locAddr, N(poly), pdegVec);
    assert(coefVec[0]==coefSizeFinal);
    return poly;
}

// Maple connector function.
// N -- #var, 
// rdgs -- result degree vector.
// rBsz -- size of resBuffer, resBuffer -- Buffer to keep result.
// p1dgs -- p1 degree vector.
// p1sz -- size of p1Buffer, p1Buffer -- Buffer of p1, 
// p2dgs -- p2 degree vector.
// p2sz -- size of p2Buffer, p2Buffer -- Buffer of p2.
// p -- prime number.
#ifdef WINDOWS
__declspec(dllexport) void __stdcall 
#else
void
#endif
MulPolyTFTFFTCN(sfixn N, sfixn *rdgs, sfixn rBsz, sfixn *resBuffer, sfixn *p1dgs,
                sfixn p1sz, sfixn *p1Buffer, sfixn *p2dgs, sfixn p2sz, sfixn *p2Buffer,
                sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn p)
{
    preFFTRep *poly1, *poly2, *prod;
    MONTP_OPT2_AS_GENE prime;

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly1 = createOneWrapperPoly(N, p1dgs, p1Buffer);
    poly2 = createOneWrapperPoly(N, p2dgs, p2Buffer);
    prod = createOneWrapperPoly(N, rdgs, resBuffer);
    
    EX_mulPoly_TFTFFT(N, prod, poly1, poly2, &prime);
    create_pdeg_coef_Vec(pdegVec, coefVec, prod);

    freeOneWrapperPoly(poly1);
    freeOneWrapperPoly(poly2);
    freeOneWrapperPoly(prod);
}

// Maple connector function for multivariate multiplication based FFT/TFT.
// N -- #var, 
// rdgs -- result degree vector.
// rBsz -- size of resBuffer, resBuffer -- Buffer to keep result.
// p1dgs -- p1 degree vector.
// p1sz -- size of p1Buffer, p1Buffer -- Buffer of p1, 
// p2dgs -- p2 degree vector.
// p2sz -- size of p2Buffer, p2Buffer -- Buffer of p2.
// p -- prime number.
#ifdef WINDOWS
__declspec(dllexport) void __stdcall 
#else
void
#endif
MulPolyTFTFFTCNC(sfixn N, sfixn *dgs1, sfixn p1dgssz, sfixn *p1dgs, sfixn p1sz,
                 sfixn *p1Buffer, sfixn *dgs2, sfixn p2dgssz, sfixn *p2dgs, 
                 sfixn p2sz, sfixn *p2Buffer, sfixn *rdgs, sfixn dVsz, 
                 sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn p)
{
    preFFTRep *poly1, *poly2, *prod;
    MONTP_OPT2_AS_GENE prime;
    EX_MontP_Init_OPT2_AS_GENE(&prime,  p);

    poly1 = inverse_create_pdeg_coef_Vec(N, dgs1, p1dgs, p1Buffer);
    poly2 = inverse_create_pdeg_coef_Vec(N, dgs2, p2dgs, p2Buffer);
    prod = (preFFTRep *)my_malloc(sizeof(preFFTRep));
    InitOnePoly(prod, N, rdgs);

    EX_mulPoly_TFTFFT(N, prod, poly1, poly2,  &prime);
    create_pdeg_coef_Vec (pdegVec, coefVec, prod);

    freePoly(poly1);
    freePoly(poly2);
    freePoly(prod);
    my_free(poly1);
    my_free(poly2);
    my_free(prod);
}

int useTFT(sfixn d1, sfixn d2){
    sfixn n1, n2, diff=20;
    n1 = 1<<(logceiling(d1));
    n2 = 1<<(logceiling(d2));
    if (((n1-d1)>diff) && ((n2-d2)>diff)) {
        return 1;
    }
    return 0;
}

// Maple connector function for univariate multiplication based on FFT/TFT.
#ifdef WINDOWS
__declspec(dllexport) void __stdcall 
#else
void
#endif
TFTFFTUNIC(sfixn dr, sfixn *resPtr, sfixn d1, sfixn *v1Ptr, sfixn d2, sfixn *v2Ptr, sfixn p)
{ 
    MONTP_OPT2_AS_GENE prime;
    EX_MontP_Init_OPT2_AS_GENE(&prime,  p);

    if (useTFT(d1, d2)) {
        EX_Mont_TFTMul_OPT2_AS_GENE_WHOLE(resPtr, d1,v1Ptr, d2, v2Ptr, &prime);
    } else {
        EX_Mont_FFTMul(dr, resPtr, d1, v1Ptr, d2, v2Ptr, &prime);
    }
}

// Maple connector function for fast univariate division.
#ifdef WINDOWS
__declspec(dllexport) void __stdcall 
#else
void
#endif
FASTDIVC(sfixn degR, sfixn *RPtr, sfixn degQ, sfixn *QPtr, sfixn degA, sfixn *APtr,
         sfixn degB, sfixn *BPtr, sfixn p)
{
    MONTP_OPT2_AS_GENE prime;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    fastDiv(RPtr, degQ, QPtr, degA, APtr, degB, BPtr, &prime);
}

// Maple connector function for classical univariate division.
#ifdef WINDOWS
__declspec(dllexport) void __stdcall 
#else
void
#endif
PLAINDIVC(sfixn degR, sfixn *RPtr,sfixn degQ, sfixn *QPtr, sfixn degA, sfixn *APtr,
          sfixn degB, sfixn *BPtr, sfixn p)
{
    MONTP_OPT2_AS_GENE prime;
    EX_MontP_Init_OPT2_AS_GENE(&prime,  p);
    plainDivNew(RPtr, degQ, QPtr, degA, APtr, degB, BPtr, &prime); 
}

// Maple connector function for classical univariate GCD.
#ifdef WINDOWS
__declspec(dllexport) void __stdcall 
#else
void
#endif
PLAINGCDUNIC(sfixn ud, sfixn *uPtr, sfixn vd, sfixn *vPtr, sfixn gd, sfixn *gcdPtr,
             sfixn dA, sfixn *APtr, sfixn dB, sfixn *BPtr, sfixn p)
{
    sfixn dC = dA+dB, dD = dA+dB, dG = dA, *dCAddr = &dC, *dDAddr = &dD, *dGAddr = &dG;
    MONTP_OPT2_AS_GENE prime;
    EX_MontP_Init_OPT2_AS_GENE(&prime,  p);
    ExGcd_Uni(uPtr, dCAddr, vPtr, dDAddr, gcdPtr, dGAddr, APtr, dA, BPtr, dB, &prime);
}

// Maple connector function for classical univariate rational function reconstruction.
#ifdef WINDOWS
__declspec(dllexport) void __stdcall 
#else
void
#endif
PLAINRFRUNIC(sfixn d, sfixn vd, sfixn *vPtr, sfixn gd, sfixn *gcdPtr, sfixn dA,
             sfixn *APtr, sfixn dB, sfixn *BPtr, sfixn p)
{
    sfixn dD=dA+dB, dG=dA, *dDAddr=&dD, *dGAddr=&dG;
    MONTP_OPT2_AS_GENE prime;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    ExGcd_Uni_RFR(d, vPtr, dDAddr, gcdPtr, dGAddr, APtr, dA, BPtr, dB, &prime);
}

// Maple connector function of fast univariate GCD.
#ifdef WINDOWS
__declspec(dllexport) void __stdcall 
#else
void
#endif
FASTGCDUNIC(sfixn ud, sfixn *uPtr, sfixn vd, sfixn *vPtr, sfixn gd, sfixn *gcdPtr,
            sfixn dA, sfixn *APtr, sfixn dB, sfixn *BPtr, sfixn p)
{
    int i;
    sfixn dC=dB, dD=dA, dG=0, *dCAddr=&dC, *dDAddr=&dD, *dGAddr=&dG;
    sfixn invG;
    MONTP_OPT2_AS_GENE prime;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);

    XGCD(uPtr, dCAddr, vPtr, dDAddr, gcdPtr, dGAddr, APtr, dA, BPtr, dB, &prime);
    invG = inverseMod(gcdPtr[dG], p);

    for(i=0; i<=dD; i++) vPtr[i] = MulMod(vPtr[i], invG, p);
    for(i=0; i<=dC; i++) uPtr[i] = MulMod(uPtr[i], invG, p);
    for(i=0; i<=dG; i++) gcdPtr[i] = MulMod(gcdPtr[i], invG, p);
}

//  Maple connector function of sub-product tree trick.
#ifdef WINDOWS
__declspec(dllexport) void __stdcall 
#else
void
#endif
subProdTreeCreWrapC(sfixn h, sfixn levels, sfixn *W, sfixn *NoNodes, 
                   sfixn *Bases, sfixn totSZ, sfixn *data, 
                   sfixn itemNo, sfixn itemSz, sfixn p)
{
    subProdTree *tree = (subProdTree *)my_malloc(sizeof(subProdTree));
    int i, j, k;
    int q, r, base1, base2, Wdouble, tmp;
    MONTP_OPT2_AS_GENE prime;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);

    tree->h = h;
    tree->W = W;
    tree->NoNodes = NoNodes;
    tree->Bases = Bases;
    tree->data = data;
    tmp = itemNo * itemSz;
    base2 = totSZ - tmp;

    for (i=1; i<levels; i++) {
        tree->Bases[i] = base2;
        q = (tree->NoNodes)[i]/2;
        r = (tree->NoNodes)[i] - q*2;
        base1 = base2-(tree->NoNodes)[i+1]*(tree->W)[i+1];
        Wdouble = (tree->W)[i]*2;
        for (j=0, k=0; j<q*Wdouble; j+=Wdouble) {
            mulNodes((tree->W)[i+1]-1, (tree->data)+base1+k, (tree->W)[i]-1, 
                                       (tree->data)+base2+j, (tree->data)+base2+j+(tree->W)[i], 
                                       (tree->W)[i], &prime);
            k += (tree->W)[i+1];
        }
        if (r) { copyVec_0_to_d((tree->W)[i]-1, (tree->data)+base1+k, (tree->data)+base2+j); }
        base2 = base1;
    } 
    subProdTreeFreeWrapC(tree);
}

subProdTree *subProdTreeGatherWrapC(sfixn h, sfixn *W, sfixn *NoNodes, sfixn *Bases, sfixn *data)
{
    subProdTree *tree =(subProdTree *)my_malloc(sizeof(subProdTree));
    tree->h=h;
    tree->W=W;
    tree->NoNodes=NoNodes;
    tree->Bases=Bases;
    tree->data=data;
    return tree;
}

void subProdTreeFreeWrapC(subProdTree *tree){
    if (tree){
        if(tree->W) { (tree->W) = NULL;}
        if(tree->NoNodes) {(tree->NoNodes) = NULL;}
        if(tree->data){ (tree->data) = NULL;}
        my_free(tree);
        tree = NULL;
    }
}

// Maple connector function of univariate fast evaluation based on sub-product tree trick.
#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
FastEvalWrapC(sfixn n, sfixn *EvalPts, sfixn degf, sfixn *fPtr, sfixn h, sfixn *W,
              sfixn *NoNodes, sfixn *Bases, sfixn *data, sfixn p)
{
    int i;
    sfixn *EvalVec;
    subProdTree *tree;
    tree = subProdTreeGatherWrapC(h, W, NoNodes, Bases, data);
    EvalVec = FastEvaluation(n,degf, fPtr, tree, p);
    for (i=0; i<n; i++) EvalPts[i] = EvalVec[i];
    my_free(EvalVec);
    subProdTreeFreeWrapC(tree);
}

// Maple connector function of univariate fast interpolation based on sub-product tree trick.
#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
FastInterpWrapC(sfixn n, sfixn *InterpedPts, sfixn *EvaluatingPts, sfixn *EvaluatedPts,
                sfixn h, sfixn *W, sfixn *NoNodes, sfixn *Bases, sfixn *data, sfixn p)
{
    int i;
    sfixn *coeffVecTmp;
    subProdTree *tree;
    sfixn polyDg, *polyDgAddr=&polyDg;
    tree = subProdTreeGatherWrapC(h, W, NoNodes, Bases, data);
    coeffVecTmp = fastInterp(polyDgAddr, n, EvaluatingPts, tree, EvaluatedPts, p);
    for (i=0; i<=polyDg; i++) InterpedPts[i] = coeffVecTmp[i];
    my_free(coeffVecTmp);
    subProdTreeFreeWrapC(tree);
}

// Maple connector function of fast interplation + rational function reconstruction.
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
FastInterpRFRWrapC(sfixn *np, sfixn *InterpedPtsNum, sfixn *InterpedPtsDen,
                   sfixn *EvaluatingPts, sfixn *EvaluatedPts, sfixn h,
                   sfixn *W, sfixn *NoNodes, sfixn *Bases, sfixn *data)
{
    sfixn n, p;
    int res;
    sfixn *coeffVecTmp, MDg;
    subProdTree *tree;
    sfixn polyDg, *polyDgAddr = &polyDg, DenDg, *DenDgAddr = &DenDg, NumDg, *NumDgAddr=&NumDg;
    MONTP_OPT2_AS_GENE prime;
    sfixn mymsg=0;

    n = np[0]; 
    p = np[1];
    if (testPrimeNumber(p)==0) return -200;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);

    tree = subProdTreeGatherWrapC(h, W, NoNodes, Bases, data);
    coeffVecTmp = fastInterp(polyDgAddr, n, EvaluatingPts, tree, EvaluatedPts, p);
    MDg = (tree->W)[(tree->h)+1];
    MDg = shrinkDegUni(MDg-1, (tree->data));
    DenDg = n-1;

    res = ExGcd_Uni_RFR(n, InterpedPtsDen, DenDgAddr, InterpedPtsNum, NumDgAddr,
                        tree->data, MDg, coeffVecTmp, polyDg, &prime);
    if (res == -1) {
        printf("rational function reconstruction failed!\n");
        fflush(stdout);
        Interrupted=1;
    }
    my_free(coeffVecTmp);
    subProdTreeFreeWrapC(tree);
    if (Interrupted==1) mymsg+=-10;
    if (PrimeError==1) mymsg+=-100;
    return mymsg;
}

//<><><><><><><><><><><><><><><><><><><><><><><><><><><><>

// Maple-connecting function, 
// creating good points + corresponding subProductTrees at C level.
// Compacting the results into Maple level buffers for later usage.
// from C -> Maple, the data will be copied into maple buffer.
// principle is maple buffer can read anywhere in the C code.
// But they only be filled in the connecting functions.
// this should be the safest way with maximal space reuse.
// the size of W_s, NoNodes_s, and Bases_s are m*(N+1).
// the rest buffer keep data in completely compact way without any useless slot.

// This function is doubtful. Only dims1 is used.
// The second dims1 should be dims2. WP

void splitdims12(sfixn N, sfixn *dims1, sfixn *dims2, sfixn *dims12){
    sfixn i;
    for (i=0; i<N; i++) dims1[i+1] = dims12[i];
    for (i=N; i<2*N; i++) dims1[i+1] = dims12[i];
}

// Nmp[0]=N; Nmp[1]=m; Nmp[2]=p;
// ptPHWDSZ[0]=pts_sSz;
// ptPHWDSZ[1]=h_sSz;
// ptPHWDSZ[2]=WNB_sSz;
// ptPHWDSZ[3]=data_sSz;
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
PTRTREESCRECN(sfixn *Nmp, sfixn *ptPHWDSZ, sfixn *bounds, sfixn *dims1, sfixn *dims2,
              sfixn *pts_s, sfixn *h_s, sfixn *W_s, sfixn *NoNodes_s, sfixn *Bases_s,
              sfixn *data_s, sfixn *p1dgs, sfixn p1sz, sfixn *p1Buffer, sfixn *p2dgs,
              sfixn p2sz, sfixn *p2Buffer)
{
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    int i, offsetPts = 0, offsetTreeWNB = 0, offsetTreeData = 0, dataSz = 0;
    PTS_TREE *pts_tree;
    subProdTree *tree;
    preFFTRep *f1, *f2;
    MONTP_OPT2_AS_GENE prime;
    sfixn mymsg = 0;
    if (testPrimeNumber(p)==0) return -200;

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    f1 = createOneWrapperPoly(N, p1dgs, p1Buffer);
    f2 = createOneWrapperPoly(N, p2dgs, p2Buffer);
    // f1, f2 in x_n has no leading zeros.

    pts_tree = createGoodPtsForf1f2(N, BUSZSI(f1, N), f1, BUSZSI(f2, N), 
                                    f2, m, bounds, dims1, dims2, &prime);

    if (pts_tree==NULL) { 
        freeOneWrapperPoly(f1);
        freeOneWrapperPoly(f2);
        if (Interrupted==1) mymsg+=-10;
        if (PrimeError==1) mymsg+=-100;
        return mymsg;
    }

    if (Interrupted==1) mymsg+=-10;
    if (PrimeError==1) mymsg+=-100;
    if (mymsg<0) {
        freeOneWrapperPoly(f1);
        freeOneWrapperPoly(f2);
        return mymsg;
    }
    // copy created pts_tree into Maple's buffer
    // pts_tree-> no actually is m in the main algorithm.
    for (i=1; i<=(pts_tree->no); i++) {
        // the real data for pts starts from slot-0.
        copyVec_0_to_d(bounds[i]-1, pts_s+offsetPts, (pts_tree->ptsPtr)[i]);
        offsetPts += bounds[i];
        tree = (pts_tree->trees)[i];
        // storing the tree data without first 0.
        // h, W, NoNodes, Bases, data.
        // the real data for h_s starts from slot-0.
        h_s[i-1] = tree->h;
        // the real data for W_s, NoNodes_s, Bases_s starts from slot-1.
        // so needs to jump N+1, each time
        copyVec_1_to_n((tree->h)+1, W_s+offsetTreeWNB, (tree->W));
        copyVec_1_to_n((tree->h)+1, NoNodes_s+offsetTreeWNB, (tree->NoNodes));
        copyVec_1_to_n((tree->h)+1, Bases_s+offsetTreeWNB, (tree->Bases));
        offsetTreeWNB += (tree->h)+1+1;
        dataSz = (tree->Bases)[1]+(tree->NoNodes[1])*(tree->W[1]);
        // real data for data starts from slot-0.
        copyVec_0_to_d(dataSz-1, data_s+offsetTreeData, tree->data);
        offsetTreeData += dataSz;
    }
 
    freeOneWrapperPoly(f1);
    freeOneWrapperPoly(f2);
    freePTS_TREE(pts_tree);

    return mymsg;
}

// Maple connector function of pass a evaluation data structure from Maple to C.
//  bounds=my_calloc(m+1, sizeof(sfixn));
//  dims1=my_calloc(N+1, sizeof(sfixn));
//  dims2=my_calloc(N+1, sizeof(sfixn));
//  slicedims=my_calloc(N+1, sizeof(sfixn));
//  subslicedims=my_calloc(N+1, sizeof(sfixn));
//  dgs1=my_calloc(N+1, sizeof(sfixn));
//  dgs2=my_calloc(N+1, sizeof(sfixn));
// for second case.
void PTRTREESCRECNC(sfixn *Nmp, sfixn *ptPHWDSZ, sfixn *bounds, sfixn *dims1,
                    sfixn *dims2, sfixn *pts_s, sfixn *h_s, sfixn *W_s,
                    sfixn *NoNodes_s, sfixn *Bases_s, sfixn *data_s,
                    sfixn *dgs1, sfixn p1dgssz, sfixn *p1dgs, sfixn p1sz,
                    sfixn *p1Buffer, sfixn *dgs2, sfixn p2dgssz, sfixn *p2dgs,
                    sfixn p2sz, sfixn *p2Buffer)
{ 
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    int i, offsetPts = 0, offsetTreeWNB = 0, offsetTreeData = 0, dataSz = 0;
    PTS_TREE *pts_tree;
    subProdTree *tree;
    preFFTRep *f1, *f2;
    MONTP_OPT2_AS_GENE prime;

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    f1 = inverse_create_pdeg_coef_Vec(N, dgs1, p1dgs, p1Buffer);
    f2 = inverse_create_pdeg_coef_Vec(N, dgs2, p2dgs, p2Buffer);
    // f1, f2 in x_n has no leading zeros.
    pts_tree = createGoodPtsForf1f2(N, BUSZSI(f1, N), f1, BUSZSI(f2, N), f2, m,
                                    bounds, dims1, dims2, &prime);
  
    // copy created pts_tree into Maple's buffer
    // pts_tree-> no actually is m in the main algorithm.
    for (i=1; i<=(pts_tree->no); i++) {
        // the real data for pts starts from slot-0.
        copyVec_0_to_d(bounds[i]-1, pts_s+offsetPts, (pts_tree->ptsPtr)[i]);
        offsetPts += bounds[i];
        tree = (pts_tree->trees)[i];
        // storing the tree data without first 0.
        // h, W, NoNodes, Bases, data.
        // the real data for h_s starts from slot-0.
        h_s[i-1] = tree->h;
        // the real data for W_s, NoNodes_s, Bases_s starts from slot-1.
        // so needs to jump N+1, each time

        copyVec_1_to_n((tree->h)+1, W_s+offsetTreeWNB, (tree->W));
        copyVec_1_to_n((tree->h)+1, NoNodes_s+offsetTreeWNB, (tree->NoNodes));
        copyVec_1_to_n((tree->h)+1, Bases_s+offsetTreeWNB, (tree->Bases));
        offsetTreeWNB += (tree->h)+1+1;
        dataSz = (tree->Bases)[1]+(tree->NoNodes[1])*(tree->W[1]);
        // real data for data starts from slot-0.
        copyVec_0_to_d(dataSz-1, data_s+offsetTreeData, tree->data);
        offsetTreeData += dataSz;
    }
    freePoly(f1); my_free(f1);
    freePoly(f2); my_free(f2);
    freePTS_TREE(pts_tree);
}

// creating a wrapper tree and hooking up the maple passed in data 
// onto the tree structure.
subProdTree *createWrapperTree(sfixn h, sfixn *W, sfixn *NoNodes, sfixn *Bases, sfixn *data)
{
    subProdTree *tree;
    tree = (subProdTree *)my_malloc(sizeof(subProdTree));
    tree->h = h;
    tree->W = W;
    tree->NoNodes = NoNodes;
    tree->data = data;
    tree->Bases = Bases;
    return tree;
}

/* struct pts_trees_struct{ */
/*   sfixn no; // no of trees / <set of points>. */
/*   sfixn **ptsPtr; // has no+1 slots, first one is not used. */
/*   subProdTree **trees; // has no+1 slots, first one is not used. */
/*   sfixn times; */
/* }; */

// creating the PTS_TREE from data passed from Maple.
// the returned PTS_TREE *trees will be used in C level functions.
// from maple to C, the Maple buffer will be hooked into the wrapper structs.
PTS_TREE *createWrapperPTS_TREE(sfixn N, sfixn m, sfixn *bounds, sfixn pts_sSz,
                                sfixn *pts_s, sfixn h_sSz, sfixn *h_s, sfixn WNB_sSz,
                                sfixn *W_s, sfixn *NoNodes_s, sfixn *Bases_s, 
                                sfixn data_sSz, sfixn *data_s, sfixn p)
{
    int i, offsetPts = 0, offsetTreeWNB = 0, offsetTreeData = 0, dataSz = 0;
    PTS_TREE *pts_tree;
    MONTP_OPT2_AS_GENE prime;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    pts_tree = (PTS_TREE *)my_malloc(sizeof(PTS_TREE));
    pts_tree->no = m;
    // this statistic info is not useful here.
    pts_tree->times = 0;
    pts_tree->ptsPtr = (sfixn **)my_calloc((pts_tree->no)+1, sizeof(sfixn *));
    (pts_tree->ptsPtr)[0] = NULL;
    pts_tree->trees = (subProdTree **)my_calloc((pts_tree->no)+1, sizeof(subProdTree *));
    (pts_tree->trees)[0] = NULL;

    for (i=1; i<=m; i++) {
        (pts_tree->ptsPtr)[i] = (sfixn *)my_calloc(bounds[i], sizeof(sfixn));
        copyVec_0_to_d(bounds[i]-1, (pts_tree->ptsPtr)[i], pts_s+offsetPts);
        offsetPts += bounds[i];
        (pts_tree->trees)[i] = createWrapperTree(h_s[i-1], W_s+offsetTreeWNB,
                               NoNodes_s+offsetTreeWNB, Bases_s+offsetTreeWNB,
                               data_s+offsetTreeData);
        dataSz = Bases_s[offsetTreeWNB+1]+NoNodes_s[offsetTreeWNB+1]*W_s[offsetTreeWNB+1];
        offsetTreeWNB += h_s[i-1]+1+1; 
        offsetTreeData += dataSz;
    }
    return (pts_tree);
}

// simply free the tree shell.
void freeWrapperTree(subProdTree *tree){
    my_free(tree);
}

void freeWrapperPTS_TREE(PTS_TREE *pts_tree){
    int i;
    if (pts_tree) {
        if (pts_tree->ptsPtr) my_free(pts_tree->ptsPtr);
        if (pts_tree->trees) {
            for(i=1; i<=(pts_tree->no); i++) 
                freeWrapperTree((pts_tree->trees)[i]);
            my_free(pts_tree->trees);
        }
        my_free(pts_tree);
    }
}

// Maple connector function of multivariate fast evaluaton.
// N is the no. of variable of the input polynomial f.
// m is the no. of variables x_1,...,x_m of f that we want to evaluate. 
// following is the (size, array) pairs
// (m+1, bounds),
// (pts_sSz, pts_s), (h_sSz, h_s), (WNB_sSz, W_s),  (WNB_sSz, NoNodes_s),  (WNB_sSz, Bases_s),
// (data_sSz, data_s), 
// (N+1, dims), dims the is the sizes of E. E should has the same dimension as the input
// polynomial no matter how many variables we will evaluate.
// (Esz, E),  (N+1, p1dgs), (p1sz, p1buffer).
// (dVsz, pdegVec), (cVsz, coefVec),
// CN means this is a connector function use case1 to encode poly.
// OUTPUT: E.
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
FastEvalMultiWrapCN(sfixn *Nmp, sfixn *ptPHWDSZ, sfixn *bounds, sfixn *pts_s,
                    sfixn *h_s, sfixn *W_s, sfixn *NoNodes_s, sfixn *Bases_s,
                    sfixn *data_s, sfixn *dims, sfixn Esz, sfixn *E,
                    sfixn *fdgs, sfixn fsz, sfixn *fBuffer)
{
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    sfixn pts_sSz = ptPHWDSZ[0], h_sSz = ptPHWDSZ[1];
    sfixn WNB_sSz=ptPHWDSZ[2], data_sSz = ptPHWDSZ[3]; 
    preFFTRep *poly;
    MONTP_OPT2_AS_GENE prime;
    PTS_TREE *pts_tree;
    sfixn mymsg = 0;
    if (testPrimeNumber(p)==0) return -200;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    pts_tree = createWrapperPTS_TREE(N, m, bounds, pts_sSz, pts_s, h_sSz, h_s,
                       WNB_sSz, W_s, NoNodes_s, Bases_s, data_sSz, data_s, p);
    poly = createOneWrapperPoly(N, fdgs, fBuffer);
    fastEvalMulti_test(m, dims, E, poly, pts_tree->trees, &prime);
    freeOneWrapperPoly(poly);
    freeWrapperPTS_TREE(pts_tree);
    if (Interrupted==1) mymsg += -10;
    if (PrimeError==1) mymsg += -100;
    return mymsg;
}

// CNC means this is a connector function using case2 to encode poly.

// Maple connector function of multivariate fast interpolation.
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
FastInterpMultiWrapCN(sfixn *Nmp, sfixn *ptPHWDSZ, sfixn *bounds, 
	                  sfixn *pts_s, sfixn *h_s, sfixn *W_s, sfixn *NoNodes_s,
                      sfixn *Bases_s, sfixn *data_s, sfixn *dims, sfixn Esz,
                      sfixn *E, sfixn dVsz, sfixn *pdegVec,
                      sfixn cVsz, sfixn *coefVec)
{
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    sfixn pts_sSz = ptPHWDSZ[0], h_sSz = ptPHWDSZ[1];
    sfixn WNB_sSz = ptPHWDSZ[2], data_sSz = ptPHWDSZ[3]; 
    MONTP_OPT2_AS_GENE prime;
    PTS_TREE *pts_tree;
    preFFTRep *poly;
    sfixn mymsg=0;

    if (testPrimeNumber(p)==0) return -200;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    pts_tree = createWrapperPTS_TREE(N, m, bounds, pts_sSz, pts_s, h_sSz, h_s,
                       WNB_sSz, W_s, NoNodes_s, Bases_s, data_sSz, data_s, p);

    poly = fastInterpMulti_test(N, m, dims,  E, pts_tree->ptsPtr, pts_tree->trees, &prime);

    if (poly==NULL) {
        freeWrapperPTS_TREE(pts_tree);
        if (Interrupted==1) mymsg+=-10;
        if (PrimeError==1) mymsg+=-100;
        return mymsg;
    }
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);
    freeWrapperPTS_TREE(pts_tree);
    if (Interrupted==1) mymsg += -10;
    if (PrimeError==1) mymsg += -100;
    return mymsg;
}

// N is the no. of variable of the input polynomial f.
// (N+1, Edims1/Edims2), dims is the sizes of E.
// E should has the same dimension as the input.
// OUTPUT: S.
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
SubResultantChains(sfixn N, sfixn w, sfixn Ssz, sfixn *S, sfixn* Edims1, sfixn E1sz,
                   sfixn *E1, sfixn *Edims2, sfixn E2sz, sfixn*E2, sfixn p)
{
    MONTP_OPT2_AS_GENE prime;
    sfixn mymsg = 0;
    if (testPrimeNumber(p)==0) return -200;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);

    getSubResultantChains(N, w, Ssz, S, Edims1, E1, Edims2, E2, &prime);
    if (Interrupted==1) mymsg += -10;
    if (PrimeError==1) mymsg += -100;
    return mymsg;
}

// Maple connector function of compute the multivariate quotient. 
#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
GetQuotientCN(sfixn N, sfixn dd, sfixn Ssz, sfixn *S, sfixn* Edims1,
              sfixn E1sz, sfixn *E1, sfixn *Edims2, sfixn E2sz, sfixn*E2,
              sfixn p, sfixn opt)
{
    MONTP_OPT2_AS_GENE prime;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    getQuotients(N, dd, Ssz, S, Edims1, E1, Edims2, E2, &prime, (int)opt);
}

// Maple connector function of interpolate a coefficient from the sub-resultant chain.
#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
InterpIthDthMultiWrapCN(sfixn *Nmp, sfixn *ptPHWDSZ, sfixn ith, sfixn dth, sfixn *bounds, 
	                    sfixn *pts_s, sfixn *h_s, sfixn *W_s, sfixn *NoNodes_s,
                        sfixn *Bases_s, sfixn *data_s, sfixn w, sfixn subslicesz,
                        sfixn *subslicedims, sfixn Ssz, sfixn *S, sfixn dVsz,
                        sfixn *pdegVec, sfixn cVsz, sfixn *coefVec)
{
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    sfixn pts_sSz = ptPHWDSZ[0], h_sSz = ptPHWDSZ[1]; 
    sfixn WNB_sSz = ptPHWDSZ[2], data_sSz = ptPHWDSZ[3]; 
    MONTP_OPT2_AS_GENE prime;
    PTS_TREE *pts_tree;
    preFFTRep *poly;

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    pts_tree = createWrapperPTS_TREE(N, m, bounds, pts_sSz, pts_s, h_sSz, h_s, WNB_sSz,
                                     W_s, NoNodes_s, Bases_s, data_sSz, data_s, p);
    poly = interpIthDthSlice(ith, dth, N, m, w, subslicesz, subslicedims, Ssz, S,  pts_tree,  &prime);
    create_pdeg_coef_Vec (pdegVec, coefVec, poly);

    freePoly(poly);
    my_free(poly);
    freeWrapperPTS_TREE(pts_tree);
}

// Maple connector function of interpolate polynomial from the sub-resultant chain.
#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
InterpIthMultiWrapCN(sfixn *Nmp, sfixn *ptPHWDSZ, sfixn ith, sfixn *bounds, 
	                 sfixn *pts_s, sfixn *h_s, sfixn *W_s, sfixn *NoNodes_s,
                     sfixn *Bases_s, sfixn *data_s, sfixn w, sfixn slicesz,
                     sfixn *slicedims, sfixn Ssz, sfixn *S, sfixn dVsz,
                     sfixn *pdegVec, sfixn cVsz, sfixn *coefVec)
{
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    sfixn pts_sSz = ptPHWDSZ[0], h_sSz = ptPHWDSZ[1];
    sfixn WNB_sSz = ptPHWDSZ[2], data_sSz = ptPHWDSZ[3]; 
    MONTP_OPT2_AS_GENE prime;
    PTS_TREE *pts_tree;
    preFFTRep *poly;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    pts_tree=createWrapperPTS_TREE(N, m, bounds, pts_sSz, pts_s, h_sSz, h_s,
                        WNB_sSz, W_s, NoNodes_s, Bases_s, data_sSz, data_s, p);
    poly = interpIthSlice(ith, N, m, w, slicesz, slicedims, Ssz, S,  pts_tree, &prime);

    create_pdeg_coef_Vec (pdegVec, coefVec, poly);

    freePoly(poly);
    my_free(poly);
    freeWrapperPTS_TREE(pts_tree);
}

// Maple connector function of interpolating the next leading coefficient in
// the sub-resultant chain.

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
                sfixn cVsz, sfixn *coefVec)
{
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    sfixn pts_sSz=ptPHWDSZ[0], h_sSz=ptPHWDSZ[1];
    sfixn WNB_sSz=ptPHWDSZ[2], data_sSz=ptPHWDSZ[3];
    MONTP_OPT2_AS_GENE prime;
    PTS_TREE *pts_tree;
    preFFTRep *poly;
    int nexti, *nextiAddr=&nexti;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);  
    pts_tree = createWrapperPTS_TREE(N, m, bounds, pts_sSz, pts_s, h_sSz, h_s,
                       WNB_sSz, W_s, NoNodes_s, Bases_s, data_sSz, data_s, p);

    poly = interpNextCandidateSliceLCDefective(nextiAddr, start, N, m, w, subslicesz,
                                             subslicedims, Ssz, S, pts_tree, &prime);
    if (poly==NULL) return -1;
    create_pdeg_coef_Vec (pdegVec, coefVec, poly);

    freePoly(poly);
    my_free(poly);
    freeWrapperPTS_TREE(pts_tree);

    return nexti;
}

// Maple connector function of interpolating the next leading coefficient in
// the sub-resultant chain.

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
InterpNextLCMultiWrapCN(sfixn *Nmp, sfixn *ptPHWDSZ,sfixn start, sfixn *bounds, 
	           sfixn *pts_s, sfixn *h_s, sfixn *W_s, sfixn *NoNodes_s, sfixn *Bases_s,
               sfixn *data_s, sfixn w, sfixn subslicesz, sfixn *subslicedims, 
               sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
               sfixn cVsz, sfixn *coefVec)
{
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    sfixn pts_sSz = ptPHWDSZ[0], h_sSz = ptPHWDSZ[1];
    sfixn WNB_sSz = ptPHWDSZ[2], data_sSz = ptPHWDSZ[3];
    MONTP_OPT2_AS_GENE prime;
    PTS_TREE *pts_tree;
    preFFTRep *poly;
    int nexti, *nextiAddr=&nexti;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);  

    pts_tree = createWrapperPTS_TREE(N, m, bounds, pts_sSz, pts_s, h_sSz, h_s,
                       WNB_sSz, W_s, NoNodes_s, Bases_s, data_sSz, data_s, p);

    poly = interpNextCandidateSliceLC(nextiAddr, start, N, m, w, subslicesz,
                                      subslicedims, Ssz, S, pts_tree, &prime);
    if (poly==NULL) return -1;
    create_pdeg_coef_Vec (pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);
    freeWrapperPTS_TREE(pts_tree);
    return nexti;
}

// Maple connector function of multivariate DFT.
#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
DftMultiWrapCN(sfixn *Nmp, sfixn *es, sfixn *dims, sfixn Esz, sfixn *E,
	           sfixn *fdgs, sfixn fsz, sfixn *fBuffer, sfixn *rootsPtr)
{
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    preFFTRep *poly;
    MONTP_OPT2_AS_GENE prime;
    
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = createOneWrapperPoly(N, fdgs, fBuffer);
    fastDftMulti_test(m, es, dims, E, poly, rootsPtr, &prime);
    freeOneWrapperPoly(poly);
}

// ...CNC means this is a connector function using case2 to encode poly.

// Maple connector function of multivariate inverse DFT.
#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
InvDftMultiWrapCN(sfixn *Nmp, sfixn *es, sfixn *dims, sfixn Esz, sfixn *E,
                  sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec,
                  sfixn *rootsPtr)
{
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly;

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = fastInvDftMulti_test(N, m, es, dims, E, rootsPtr, &prime);
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);
}

// Maple connector function of interpolate a coefficient from the sub-resultant 
// via inverse DFT
#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
InvDftIthDthMultiWrapCN(sfixn *Nmp, sfixn ith, sfixn dth, sfixn w,
                        sfixn subslicesz, sfixn *subslicees,
                        sfixn *subslicedims, sfixn Ssz, sfixn *S,
                        sfixn dVsz, sfixn *pdegVec, sfixn cVsz,
                        sfixn *coefVec, sfixn *rootsPtr)
{
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = interpIthDthSliceDFT(ith, dth, N, m, w, subslicesz, subslicees,
                                subslicedims, Ssz, S, rootsPtr, &prime);
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);
}

// Maple connector function of interpolate a polynomial from the sub-resultant 
// via inverse DFT
#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
InvDftIthMultiWrapCN(sfixn *Nmp, sfixn ith, sfixn w, sfixn slicesz,
                     sfixn *slicees, sfixn *slicedims, sfixn Ssz,
                     sfixn *S, sfixn dVsz, sfixn *pdegVec, sfixn cVsz,
                     sfixn *coefVec, sfixn *rootsPtr)
{
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly;

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = interpIthSliceDFT(ith, N, m, w, slicesz, slicees, slicedims,
                             Ssz, S, rootsPtr, &prime);
    create_pdeg_coef_Vec (pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);
}

// Maple connector function of interpolate next leading coefficient from 
// the sub-resultant via inverse DFT

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
InvDftNextDefectiveLCMultiWrapCN(sfixn *Nmp, sfixn start, sfixn w,
                   sfixn subslicesz, sfixn *subslicees,
                   sfixn *subslicedims, sfixn Ssz, sfixn *S,
                   sfixn dVsz, sfixn *pdegVec, sfixn cVsz,
                   sfixn *coefVec, sfixn *rootsPtr)
{
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly;
    int nexti, *nextiAddr=&nexti;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);

    poly = interpNextCandidateSliceLCDFTDefective(nextiAddr, start, N, m, w,
              subslicesz, subslicees,subslicedims, Ssz, S, rootsPtr, &prime);

    if (poly==NULL) return -1;
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);
    return nexti;
}

// Maple connector function of interpolate next leading coefficient from 
// the sub-resultant via inverse DFT

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
InvDftNextLCMultiWrapCN(sfixn *Nmp, sfixn start, sfixn w, sfixn subslicesz,
                   sfixn *subslicees, sfixn *subslicedims, sfixn Ssz,
                   sfixn *S, sfixn dVsz, sfixn *pdegVec, sfixn cVsz,
                   sfixn *coefVec, sfixn *rootsPtr)
{
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly;
    int nexti, *nextiAddr=&nexti;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = interpNextCandidateSliceLCDFT(nextiAddr, start, N, m, w, subslicesz,
                             subslicees,subslicedims, Ssz, S, rootsPtr, &prime);

    if (poly==NULL) return -1;
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);
    return nexti;
}

// Maple connector function of generating good roots for FFT
// return how many time used to choose roots
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
createGoodRootsCN(sfixn N, sfixn M, sfixn *f1dgs, sfixn *f1Buffer, 
                  sfixn *f2dgs, sfixn *f2Buffer, sfixn *es,
                  sfixn *dims, sfixn *rootsPtr, sfixn p)
{
    MONTP_OPT2_AS_GENE prime;
    sfixn times;
    sfixn mymsg=0;
    preFFTRep *f1, *f2;

    if(testPrimeNumber(p)==0) return -200;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    f1 = createOneWrapperPoly(N, f1dgs, f1Buffer);
    f2 = createOneWrapperPoly(N, f2dgs, f2Buffer);
    times = createGoodRootsForf1f2(N, M, BUSZSI(f1, N), f1, BUSZSI(f2, N), f2,
                                   es, dims, rootsPtr, &prime);
    freeOneWrapperPoly(f1);
    freeOneWrapperPoly(f2);

    if (Interrupted==1) mymsg+=-10;
    if (PrimeError==1)  mymsg+=-100;
    if (mymsg<0) return mymsg;
    return times;
}

// Maple connector function of converting the input system (lifting) into C data encoding.
// GNS[0] keeps the first info, and so on ...
// "no" is the number of polynomials in the input systems.
POLYVECTOR_SLG *InputSysConvertIn(sfixn no, sfixn *GNS, sfixn *MDAS)
{
    int i, offset=0;
    POLYVECTOR_SLG *vec=(POLYVECTOR_SLG *)my_malloc(sizeof(POLYVECTOR_SLG));
    SLG *slg, *newslg;
    vec->M = no;
    vec->entries = (SLG **)my_malloc((vec->M)*sizeof(SLG *));
    // GN is # of nodes, and each node is a triplet.
    for(i=0; i<(vec->M); offset+=GNS[i]*3, i++){
      //printf("GNS[%d]=%ld\n", i, GNS[i]);
       slg = Maple2C_DAG2DAG(GNS[i], MDAS+offset);
       //printSLG_Line(slg);
       //fflush(stdout);
       newslg = removRedun(slg);
       //printf("i=%ld\n", i);
       //printSLG_Line(newslg);
       freeSLG(slg);
       ENTRYI_V(vec, i) = newslg;
    }
    return vec;
}

TriSet *createWrapperDenseTriSet(sfixn N, sfixn *dgs) {
    int i;
    TriSet *tPtr = (TriSet *)my_malloc(sizeof(TriSet)); 
    N(tPtr) = N;
    // Initially, property if normailzed is unknown
    // This is suspected by WP to be the source of Windows32 bug.
    NMLZ(tPtr) = 0;
    ELEM(tPtr) = (preFFTRep **)my_calloc((N+1), sizeof(preFFTRep *) );
    BDS(tPtr) = (sfixn *) my_calloc((N+1), sizeof(sfixn));
    for(i=1; i<=N; i++)  BDSI(tPtr, i) = dgs[i-1] - 1;
    for(i=1; i<=N; i++){
        ELEMI(tPtr, i) = (preFFTRep *)my_calloc(1, sizeof(preFFTRep));
    }
    return tPtr;
}

void freeWrapperDenseTriSetPlusWrapperPolys(TriSet *tPtr){
    int i;
    if (tPtr) {
        if (BDS(tPtr)) my_free(BDS(tPtr));
        if (ELEM(tPtr)) {
            for(i=1; i<=N(tPtr); i++) {
                if(ELEMI(tPtr, i)) freeOneWrapperPoly(ELEMI(tPtr, i));
            }
            my_free(ELEM(tPtr));
        }
        my_free(tPtr);
    }
}

sfixn *getMaxDgs_1_N(sfixn N, sfixn *dgs1, sfixn *dgs2){
    int i;
    sfixn *dgs = (sfixn *)my_calloc(N+1, sizeof(sfixn));
    for (i=1; i<=N; i++) {
        if (dgs1[i]>dgs2[i]) {
            dgs[i] = dgs1[i];
        } else {
            dgs[i]=dgs2[i];         
        }
    }
    return dgs;
}

// N polynomials in the TS, ELEMI(ts, 1), ..., ELEMI(ts, N).
// TS_DGS start from 0.
// inDGS start from 0, N, 2N, ...
// inSIZS start from 0.
// inCOEFS start form 0.
TriSet *InputTriSetConvertIn(sfixn N, sfixn *TS_DGS, sfixn *inDGS, sfixn *inSIZS, sfixn *inCOEFS){
    TriSet* ts;
    int i, offset1=0, offset2=0;
    preFFTRep *poly;
    sfixn *dgs;

    ts = createWrapperDenseTriSet(N, TS_DGS);

    for(i=1; i<=N; offset1+=N, offset2+=inSIZS[i-1], i++){
        InitOneMapleWrapperPoly(ELEMI(ts, i), i, inDGS+offset1, inCOEFS+offset2);

        BDSI(ts, i)++;
        dgs=getMaxDgs_1_N(i, BUSZS(ELEMI(ts, i)),  BDS(ts));
        poly = EX_InitOnePoly(i, dgs);
        BDSI(ts, i) --;
        my_free(dgs);
        fromtofftRep(i, CUM(poly), DAT(poly), CUM(ELEMI(ts, i)), 
                     BUSZS(ELEMI(ts, i)), DAT(ELEMI(ts, i)));
        FreeOneMapleWrapperPoly(ELEMI(ts, i));     
        my_free(ELEMI(ts, i));

        ELEMI(ts, i) = poly;
    }
    return ts;
}



//lifted_ts -> (*outPDGVECS, *outCOEFVECS).
//discard first one y^{2^iter}.
void EncodeTriSet_Lift(sfixn *outPDGVECS, sfixn *outCOEFVECS, TriSet *lifted_ts){
    int i;
    sfixn N = N(lifted_ts);
    sfixn *PDGVEC = outPDGVECS, *COEFVEC = outCOEFVECS;
    int offset1=0, offset2=0;
    for (i=2; i<=N; offset1=PDGVEC[0]+1, offset2=COEFVEC[0]+1, i++) {
        PDGVEC+=offset1;
        COEFVEC+=offset2;
        // suppose pdegVec[0] keeps the size of the real data exclusive 
        // this first slot itself.         
        create_pdeg_coef_Vec (PDGVEC, COEFVEC, ELEMI(lifted_ts, i));
    }
}

// Maple connector function of Hensel lifting.
// *outPDGVECS, the gross degree vecs of polys  the returning TriSet.
// *outCOEFVECS the gross coefficient vecs of polys in the returning TriSet.
// X_Y is the variable has been specialized.
// y0, subs(Y=Y+y0, input systems and input triangular set).
// N is the number of variables. #Fs and #TriSet are both N-1.
// *GNS the gross number of nodefor DAG-Graphs of the input system.
// *MDAS the gross data of the input DAG-Graphs vectors of the input system.
// *TS_DGS is degrees of the input TriSet.
// *inDGS the gross degree vecs of polys in the input TriSet.
// *inSIZS the sizes of coefficient vecs of polys in the input Triset.
// *inCOEFS the gross coefficient vecs of polys in the returning TriSet.
// p the prime number.
#ifdef WINDOWS
__declspec(dllexport) int __stdcall
#else
int
#endif
NewtonLiftUniCN(sfixn *outPDGVECS, sfixn *outCOEFVECS, sfixn Y, sfixn y0,
                sfixn N, sfixn *GNS, sfixn *MDAS, sfixn *TS_DGS, sfixn *inDGS,
                sfixn *inSIZS, sfixn *inCOEFS, sfixn p)
{
    POLYVECTOR_SLG *inputSystem;
    TriSet *input_ts, *lifted_ts;
    MONTP_OPT2_AS_GENE prime;
    int iter, *iterAddr=&iter;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);

    //decode inputSystem. i.e. (N-1, *GNS, *MDAS )->inputSystem.
    inputSystem = InputSysConvertIn(N-1, GNS, MDAS);
    //decode input_ts. i.e. (N-1, *TS_DGS, *inDGS, *inSIZS, *inCOEFS)-> input_ts.
    input_ts = InputTriSetConvertIn(N, TS_DGS, inDGS, inSIZS, inCOEFS);
    // Lifting ...
    lifted_ts=EX_UniNewtonLift(iterAddr, Y, y0, inputSystem, input_ts, N, &prime);

    if(lifted_ts == NULL) {
        EX_freeTriSet(input_ts);
        freeVec_SLG(inputSystem);
        return -1;
    }
    // printf("===============================>>>\n");
    // printf("Lifting succeed at iteration #%d.\n", iter);
    // printf("===============================>>>\n");
    // freeing ...
    EX_freeTriSet(input_ts);
    freeVec_SLG(inputSystem);
    //encode output_ts, i.e. lifted_ts -> (*outPDGVECS, *outCOEFVECS).
    EncodeTriSet_Lift(outPDGVECS, outCOEFVECS, lifted_ts);
    EX_freeTriSet(lifted_ts);

    return iter;
}

// Maple connector function of invertibility test (without splitting!)
// x_N is the mainvar of poly.
// suppose the polynomial is reduced w.r.t TriSet.
// return 1, means yes.
// return 0, means no. it's NOT invertible.
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif 
isInvertableCN(sfixn N, sfixn *fdgs, sfixn *fBuffer, sfixn *TS_DGS, 
               sfixn *inDGS, sfixn *inSIZS, sfixn *inCOEFS, sfixn p)
{
    int i;
    TriSet *input_ts;
    TriRevInvSet *tris;
    preFFTRep *poly, *poly2, *newpoly, tmp, *tmpPtr=&tmp;
    sfixn reci,  *dgs, invertibility;
    MONTP_OPT2_AS_GENE prime;
    sfixn mymsg=0;
    if (testPrimeNumber(p)==0) return -200;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);

    poly = createOneWrapperPoly(N, fdgs, fBuffer);
    input_ts = InputTriSetConvertIn(N, TS_DGS, inDGS, inSIZS, inCOEFS);
    invertibility=MonicizeTriSet_1(N, input_ts, &prime);

    if (Interrupted==1) mymsg += -10;
    if (PrimeError==1) mymsg += -100;
    if (mymsg<0) {
        freeOneWrapperPoly(poly);
        //freeWrapperDenseTriSetPlusWrapperPolys(input_ts);
        EX_freeTriSet(input_ts);
        return mymsg;
    }

    if (invertibility==-1){
        printf("Input triangular set cannot be normalized!\n");
        fflush(stdout);
        freeOneWrapperPoly(poly);
        //freeWrapperDenseTriSetPlusWrapperPolys(input_ts);
        EX_freeTriSet(input_ts);
        return -10;
    }
    newpoly = EX_InitOnePoly(N, BDS(input_ts));
    dgs = (sfixn *)my_calloc(N+1, sizeof(sfixn));
    for (i=1; i<=N; i++) {
        dgs[i] = BDSI(input_ts,i)+1;
        if (dgs[i]<BUSZSI(poly, i)) dgs[i] = BUSZSI(poly, i);
        dgs[i] <<= 1;
    }
    poly2 = EX_InitOnePoly(N, dgs);
    fromtofftRep(N, CUM(poly2), DAT(poly2), CUM(poly), BUSZS(poly), DAT(poly));

    tris = EX_initTriRevInvSet(dgs, N, input_ts);
    getRevInvTiSet(dgs, N, tris, input_ts, &prime);

    my_free(dgs);
    freeOneWrapperPoly(poly);

    MultiMod(N, newpoly, poly2, input_ts, tris,  &prime);
    EX_FreeOnePoly(poly2);

    BDSI(input_ts, N) += 1;
    InitOneReducedPoly(tmpPtr, N, BDS(input_ts));
    BDSI(input_ts, N) -= 1;

    reci = MultiRecip(N, tmpPtr, newpoly, input_ts, tris, &prime);

    EX_freeTriRevInvSet(tris);
    freePoly(tmpPtr);

    //  freeWrapperDenseTriSetPlusWrapperPolys(input_ts);

    EX_freeTriSet(input_ts);
    freePoly(newpoly);
    my_free(newpoly);

    if (Interrupted==1) mymsg+=-10;
    if (PrimeError==1) mymsg+=-100;
    if (mymsg<0) return mymsg;
    if (reci==-1) { return 0; }
    return 1;
}

// Maple connector function of 0-dimensional iterated-resultant.
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
IterResZeroDimCN(sfixn M, sfixn *fdgs, sfixn *fBuffer, sfixn N, sfixn *TS_DGS,
                 sfixn *inDGS, sfixn *inSIZS, sfixn *inCOEFS, sfixn p)
{
    preFFTRep *FPtr;
    TriSet *ts;
    MONTP_OPT2_AS_GENE prime;
    sfixn res;

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    FPtr = createOneWrapperPoly(M, fdgs, fBuffer);
    ts = InputTriSetConvertIn(N, TS_DGS, inDGS, inSIZS, inCOEFS);
    res = iteratedResultant_zerodim(FPtr, ts, &prime);

    freeOneWrapperPoly(FPtr);
    EX_freeTriSet(ts);
    return res;
}

// Maple connector function of normal form.
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
MultiModCN(sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn N, sfixn *fdgs,
           sfixn *fBuffer, sfixn *TS_DGS, sfixn *inDGS, sfixn *inSIZS, sfixn *inCOEFS,
           sfixn p, sfixn opt)
{
    TriSet *input_ts;
    TriRevInvSet *tris;
    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly, *poly2, *newpoly;
    sfixn *dgs, invertibility;
    int i;
    sfixn mymsg=0;

    if (testPrimeNumber(p)==0) return -200;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = createOneWrapperPoly(N, fdgs, fBuffer); 
    input_ts = InputTriSetConvertIn(N, TS_DGS, inDGS, inSIZS, inCOEFS);

    if (Interrupted==1) mymsg += -10;
    if (PrimeError==1) mymsg += -100;
    if (mymsg<0){
        freeOneWrapperPoly(poly);
        EX_freeTriSet(input_ts); 
        return mymsg;
    }

    if (invertibility==-1) {
        printf("The input triangular set cannot be normalized!\n");
        fflush(stdout);
        freeOneWrapperPoly(poly);
	    EX_freeTriSet(input_ts);
        Interrupted=1;
        return -10;
    }
  
    newpoly = EX_InitOnePoly(N, BDS(input_ts));
    dgs = (sfixn *)my_calloc(N+1, sizeof(sfixn));
    for (i=1; i<=N; i++) {
        dgs[i] = BDSI(input_ts,i)+1;
        if (dgs[i]<BUSZSI(poly, i)) dgs[i] = BUSZSI(poly, i);
        dgs[i] <<= 1;
    }

    poly2 = EX_InitOnePoly(N, dgs);
    fromtofftRep(N, CUM(poly2), DAT(poly2), CUM(poly),  BUSZS(poly), DAT(poly));
    tris = EX_initTriRevInvSet(dgs, N, input_ts);
    getRevInvTiSet(dgs, N, tris, input_ts, &prime);

    my_free(dgs);
    freeOneWrapperPoly(poly);

    MultiMod_OPT(N, newpoly, poly2, input_ts, tris,  &prime, opt);
  
    EX_FreeOnePoly(poly2);
    EX_freeTriRevInvSet(tris);
    EX_freeTriSet(input_ts);

    create_pdeg_coef_Vec(pdegVec, coefVec, newpoly);
    EX_FreeOnePoly(newpoly);

    if (Interrupted==1) mymsg += -10;
    if (PrimeError==1) mymsg += -100;
    return mymsg;
}

// Maple connector function of reduce coefficients.
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
ReduceCoeffCN(sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn N,
              sfixn *fdgs, sfixn *fBuffer, sfixn *TS_DGS, sfixn *inDGS, 
              sfixn *inSIZS, sfixn *inCOEFS, sfixn p)
{
    TriSet *input_ts;
    TriRevInvSet *tris;
    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly, *poly2, *newpoly;
    sfixn *dgs, invertibility;
    int i;
    sfixn mymsg=0;

    if (testPrimeNumber(p)==0) return -200;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = createOneWrapperPoly(N, fdgs, fBuffer);  
    input_ts = InputTriSetConvertIn(N-1, TS_DGS, inDGS, inSIZS, inCOEFS);
  
    invertibility = MonicizeTriSet_1(N-1, input_ts, &prime);
    if (Interrupted==1) mymsg += -10;
    if (PrimeError==1) mymsg += -100;
    if (mymsg<0) {
        freeOneWrapperPoly(poly);
        EX_freeTriSet(input_ts);
        return mymsg;
    }

    if (invertibility==-1) {
        printf("The input Triangular Set can't be normalized!\n");
        fflush(stdout);
        freeOneWrapperPoly(poly);
	    EX_freeTriSet(input_ts);
        Interrupted = 1;
        return -10;
    }

    dgs = (sfixn *)my_calloc(N+1, sizeof(sfixn));
    for (i=1; i<N; i++) {
        dgs[i] = BDSI(input_ts,i)+1;
        if (dgs[i]<BUSZSI(poly, i)) dgs[i]=BUSZSI(poly, i);
        dgs[1] <<= 1;
    }
    dgs[N] = BUSZSI(poly, N);
    poly2 = EX_InitOnePoly(N, dgs);
    fromtofftRep(N, CUM(poly2), DAT(poly2), CUM(poly),  BUSZS(poly), DAT(poly));
    tris = EX_initTriRevInvSet(dgs, N-1, input_ts);
    getRevInvTiSet(dgs, N-1, tris, input_ts, &prime);
    my_free(dgs);
    freeOneWrapperPoly(poly);
    newpoly = Ex_ReduceCoeffs(N, poly2, input_ts, tris, &prime);

    EX_FreeOnePoly(poly2);
    EX_freeTriRevInvSet(tris);
    EX_freeTriSet(input_ts);

    create_pdeg_coef_Vec (pdegVec, coefVec, newpoly);

    EX_FreeOnePoly(newpoly);
    if (Interrupted==1) mymsg+=-10;
    if (PrimeError==1) mymsg+=-100;
    return mymsg;
}



// Maple connector function of multivariate quotient computation.
#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
QuotientModTriSetCN(sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec,
                    sfixn N, sfixn *fdgs1, sfixn *fBuffer1, sfixn *fdgs2, 
                    sfixn *fBuffer2, sfixn *TS_DGS, sfixn *inDGS, 
                    sfixn *inSIZS, sfixn *inCOEFS, sfixn p)
{
    TriSet *input_ts;
    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly1, *poly1En, *poly2, *poly2En, *newpoly;
    sfixn *dgs, invertibility;
    int i;

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly1 = createOneWrapperPoly(N, fdgs1, fBuffer1);
    poly2 = createOneWrapperPoly(N, fdgs2, fBuffer2);
    input_ts = InputTriSetConvertIn(N-1, TS_DGS, inDGS, inSIZS, inCOEFS);
    invertibility = MonicizeTriSet_1(N-1, input_ts, &prime);

    if(invertibility==-1){
        printf("The input Triangular Set can't be normalized!\n");
        fflush(stdout);
        freeOneWrapperPoly(poly1);
        freeOneWrapperPoly(poly2);
        EX_freeTriSet(input_ts);
        Interrupted=1;
        return;
    }

    dgs = (sfixn *)my_calloc(N+1, sizeof(sfixn));
    for (i=1; i<N; i++) {
        dgs[i] = BDSI(input_ts, i);
        if (dgs[i]<BUSZSI(poly1, i)) dgs[i] = BUSZSI(poly1, i);
    }
    dgs[N] = BUSZSI(poly1, N);
    poly1En = EX_InitOnePoly(N, dgs);
    dgs[N] = BUSZSI(poly2, N);
    poly2En = EX_InitOnePoly(N, dgs);
    my_free(dgs);

    fromtofftRep(N, CUM(poly1En), DAT(poly1En), CUM(poly1),  BUSZS(poly1), DAT(poly1));
    freeOneWrapperPoly(poly1);
    fromtofftRep(N, CUM(poly2En), DAT(poly2En), CUM(poly2),  BUSZS(poly2), DAT(poly2));
    freeOneWrapperPoly(poly2);

    newpoly = EX_MonicMultiPlainDivide(N, poly1En, poly2En, input_ts, &prime);
    EX_FreeOnePoly(poly1En);
    EX_FreeOnePoly(poly2En);
    EX_freeTriSet(input_ts);
    create_pdeg_coef_Vec (pdegVec, coefVec, newpoly);
    EX_FreeOnePoly(newpoly);
}

void EncodeTriSet(sfixn *outPDGVECS, sfixn *outCOEFVECS, TriSet *lifted_ts) {
    int i;
    sfixn N=N(lifted_ts);
    sfixn *PDGVEC=outPDGVECS, *COEFVEC=outCOEFVECS;
    int offset1=0, offset2=0;
    for (i=1; i<=N; i++){
  	    create_pdeg_coef_Vec (PDGVEC, COEFVEC, ELEMI(lifted_ts, i));
        offset1 += PDGVEC[0]+1;
        offset2 += COEFVEC[0]+1;
        PDGVEC = outPDGVECS+offset1;
        COEFVEC = outCOEFVECS+offset2;
    }
}

// Maple connector function of normalizing a polynomial w.r.t a regular chain.
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
NormalizeCN(sfixn *outPDGVECS, sfixn *outCOEFVECS, sfixn N, sfixn *TS_DGS, sfixn *inDGS,
            sfixn *inSIZS, sfixn *inCOEFS, sfixn p)
{
    TriSet *input_ts;
    MONTP_OPT2_AS_GENE prime;
    int bool1;
    sfixn mymsg=0;

    if (testPrimeNumber(p)==0) return -200;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);

    input_ts = InputTriSetConvertIn(N, TS_DGS, inDGS, inSIZS, inCOEFS);
    bool1 = MonicizeTriSet_1(N, input_ts, &prime);
    if (bool1==1) EncodeTriSet(outPDGVECS, outCOEFVECS, input_ts);

    EX_freeTriSet(input_ts);
    if (Interrupted==1) mymsg+=-10;
    if (PrimeError==1) mymsg+=-100;
    if (mymsg<0) return mymsg;
    return bool1;
}

// Maple connector function of invertibility test (splitting case).
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
RegularGcdChainCN(sfixn *Ns, sfixn *outPolyPDGVECS, sfixn *outPolyCOEFVECS, sfixn *outTsPDGVECS,
                  sfixn *outTsCOEFVECS, sfixn N, sfixn M, sfixn *fdgs1, sfixn *fBuffer1, 
                  sfixn *fdgs2, sfixn *fBuffer2, sfixn *TS_DGS, sfixn *inDGS, sfixn *inSIZS, 
                  sfixn *inCOEFS, sfixn p)
{
    sfixn NoOfPairs, diff;
    preFFTRep *poly1, *poly2;
    LinkedQueue *resQueue;
    TriSet *input_ts;
    MONTP_OPT2_AS_GENE prime;
    sfixn mymsg=0;
    RegularPair *resPair;
    int i=0, j;
    if(testPrimeNumber(p)==0) return -200;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    
    input_ts = InputTriSetConvertIn(M, TS_DGS, inDGS, inSIZS, inCOEFS);
    poly1 = createOneWrapperPoly(N, fdgs1, fBuffer1); 
    poly2 = createOneWrapperPoly(N, fdgs2, fBuffer2); 

    /* printf("M=%d\n", M);   */
    /* printf("poly1=\n");    */
    /* printPoly(poly1);      */
    /* printf("poly2=\n");    */
    /* printPoly(poly1);      */
    /* fflush(stdout);        */
    /* printf("TriSet=\n");   */
    /* printTriSet(input_ts); */
    /* fflush(stdout);        */

    resQueue = EX_RegularGcd_Wrapped(poly1, poly2, input_ts, N, &prime);

    /* printf("after RegularGcd"); */
    /* fflush(stdout); */

    if (Interrupted==1) mymsg+=-10;
    if (PrimeError==1) mymsg+=-100;
    if (mymsg<0) {
        if (resQueue!=NULL) EX_LinkedQueue_Free(resQueue, EX_RegularPair_Free);
        EX_freeTriSet(input_ts);
        freeOneWrapperPoly(poly1);
        freeOneWrapperPoly(poly2);
        return mymsg;
    }

    NoOfPairs = resQueue->count;

    if (NoOfPairs>0) {
        while (! EX_LinkedQueue_IsEmpty(resQueue)) {
            resPair=(RegularPair *)EX_LinkedQueue_Deqeue(resQueue);
            Ns[i++]=N(resPair->poly);
            create_pdeg_coef_Vec(outPolyPDGVECS, outPolyCOEFVECS, resPair->poly);

            EncodeTriSet(outTsPDGVECS, outTsCOEFVECS, resPair->ts);

            outPolyPDGVECS+=outPolyPDGVECS[0]+1;
            outPolyCOEFVECS+=outPolyCOEFVECS[0]+1;
            for (j=0; j<N; j++) {
	            outTsPDGVECS += outTsPDGVECS[0]+1;
	            outTsCOEFVECS += outTsCOEFVECS[0]+1;
	        }
            EX_RegularPair_Free((void *)resPair);
        }
        assert(i==NoOfPairs);
    } else {
        assert(NoOfPairs==0);
        resQueue->count=1;
        resPair=(RegularPair *)EX_LinkedQueue_Deqeue(resQueue);
        Ns[i++]=N(resPair->poly);
        create_pdeg_coef_Vec(outPolyPDGVECS, outPolyCOEFVECS, resPair->poly);
        printf("before free resPair\n");
        fflush(stdout);
        EX_RegularPair_Free((void *)resPair);
        printf("after free resPair\n");
        fflush(stdout);
    }

    freeOneWrapperPoly(poly1);
    freeOneWrapperPoly(poly2);
    EX_freeTriSet(input_ts);
    EX_LinkedQueue_Free(resQueue, EX_RegularPair_Free);
    if (Interrupted==1) mymsg+=-10;
    if (PrimeError==1) mymsg+=-100;
    if (mymsg<0) return mymsg;
    return NoOfPairs;
}

// Maple connector function of invertibility test (splitting case).
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
IsInvertibleChainCN(sfixn *Ns, sfixn *outPolyPDGVECS, sfixn *outPolyCOEFVECS,
                    sfixn *outTsPDGVECS, sfixn *outTsCOEFVECS, sfixn N, 
                    sfixn *fdgs, sfixn *fBuffer, sfixn *TS_DGS, sfixn *inDGS, 
                    sfixn *inSIZS, sfixn *inCOEFS, sfixn p)
{
    sfixn NoOfPairs, diff;
    preFFTRep *poly;
    LinkedQueue *resQueue;
    TriSet *input_ts;
    MONTP_OPT2_AS_GENE prime;
    sfixn mymsg=0;
    RegularPair *resPair;
    int i=0, j;
    if (testPrimeNumber(p)==0) return -200;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);

    input_ts = InputTriSetConvertIn(N, TS_DGS, inDGS, inSIZS, inCOEFS);
    poly = createOneWrapperPoly(N, fdgs, fBuffer); 

    resQueue = isInvertible_zeroDim(poly, input_ts, &prime);

    if (Interrupted==1) mymsg+=-10;
    if (PrimeError==1) mymsg+=-100;
    if (mymsg<0){
        if (resQueue!=NULL) EX_LinkedQueue_Free(resQueue, EX_RegularPair_Free);
        EX_freeTriSet(input_ts);
        freeOneWrapperPoly(poly);
        return mymsg;
    }

    NoOfPairs = resQueue->count;

    // printf("\n\n\n\n\n (trivial case) Final OUTPUT:");
    // printf("   resQueue = ");
    // EX_LinkedQueue_Print(resQueue, EX_RegularPair_Print);
    while (! EX_LinkedQueue_IsEmpty(resQueue)) {
        resPair=(RegularPair *)EX_LinkedQueue_Deqeue(resQueue);
        Ns[i++]=N(resPair->poly);
        create_pdeg_coef_Vec(outPolyPDGVECS, outPolyCOEFVECS, resPair->poly);

        EncodeTriSet(outTsPDGVECS, outTsCOEFVECS, resPair->ts);

        outPolyPDGVECS+=outPolyPDGVECS[0]+1;
        outPolyCOEFVECS+=outPolyCOEFVECS[0]+1;

        for(j=0; j<N; j++){
            outTsPDGVECS+=outTsPDGVECS[0]+1;
            outTsCOEFVECS+=outTsCOEFVECS[0]+1;
        }
        EX_RegularPair_Free((void *)resPair);
    }

    assert(i==NoOfPairs);
    freeOneWrapperPoly(poly);
    EX_freeTriSet(input_ts);

    EX_LinkedQueue_Free(resQueue, EX_RegularPair_Free);

    if (Interrupted==1) mymsg+=-10;
    if (PrimeError==1) mymsg+=-100;
    if (mymsg<0) return mymsg;

    return NoOfPairs;
}

TriSet *InputOneDimTriSetConvertIn(sfixn freeVarNo, sfixn N, sfixn *TS_DGS, sfixn *inDGS,
                                   sfixn *inSIZS, sfixn *inCOEFS)
{
    TriSet* ts;
    int i, offset1=0, offset2=0;
    preFFTRep *poly;
    sfixn *dgs;

    ts = createWrapperDenseTriSet(N, TS_DGS);
    for (i=1; i<=N; offset1+=N, offset2+=inSIZS[i-1], i++){
        if (i==freeVarNo) { 
            ELEMI(ts, i) = NULL;
            continue;
        }

        InitOneMapleWrapperPoly(ELEMI(ts, i), i, inDGS+offset1, inCOEFS+offset2);

        BDSI(ts, i) ++;
        dgs = getMaxDgs_1_N(i, BUSZS(ELEMI(ts, i)),  BDS(ts));
        poly = EX_InitOnePoly(i, dgs);
        BDSI(ts, i)--;
        my_free(dgs);
        fromtofftRep(i, CUM(poly), DAT(poly), CUM(ELEMI(ts, i)),
                     BUSZS(ELEMI(ts, i)), DAT(ELEMI(ts, i)));
        //printf("the original poly=");
        //printPoly(ELEMI(ts, i));
        FreeOneMapleWrapperPoly(ELEMI(ts, i));     
        my_free(ELEMI(ts, i));
        ELEMI(ts, i) = poly;
        //printf("the up2date poly=");
        //printPoly(ELEMI(ts, i));
    }
    return ts;
}

// Maple connector function of iterated resultant in zero dimension.
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
IterResOneDimCN(sfixn *outVec, sfixn M, sfixn *fdgs, sfixn *fBuffer, sfixn N,
                sfixn *TS_DGS, sfixn *inDGS, sfixn *inSIZS, sfixn *inCOEFS, 
                sfixn bound, sfixn freeVarNo, sfixn p)
{
    sfixn resDg;
    preFFTRep *FPtr;
    TriSet *ts;
    MONTP_OPT2_AS_GENE prime;
    sfixn mymsg=0;
    sfixn *result;
    int i;

    if (testPrimeNumber(p)==0) return -200;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    FPtr = createOneWrapperPoly(M, fdgs, fBuffer);
    // At freeVarNo-th, there should be a zero-polynomial in ts.
    ts = InputOneDimTriSetConvertIn(freeVarNo, N, TS_DGS, inDGS, inSIZS, inCOEFS);
    result = iteratedResultant_onedim(&resDg, FPtr, ts, bound, freeVarNo, &prime);
    if (result==NULL){
        EX_freeTriSet(ts);
        freeOneWrapperPoly(FPtr);
        if (Interrupted==1) mymsg+=-10;
        if (PrimeError==1) mymsg+=-100;
        return mymsg;
    }
    for(i=0; i <=resDg; i++){
        outVec[i] = result[i];
    }
    EX_freeTriSet(ts);
    freeOneWrapperPoly(FPtr);
    my_free(result);
    if (Interrupted==1) mymsg+=-10;
    if (PrimeError==1) mymsg+=-100;
    return mymsg;
}

#ifdef WINDOWS
__declspec(dllexport) sfixn  __stdcall 
#else
sfixn
#endif
ResultantMultivariateCN(sfixn *Nnew, sfixn *pdegVec, sfixn *coefVec, sfixn N,
                        sfixn *p1dgs, sfixn *p1Buffer, sfixn *p2dgs,
                        sfixn *p2Buffer, sfixn p)
{
    preFFTRep *poly1, *poly2, *res;
    sfixn mymsg=0;
    MONTP_OPT2_AS_GENE prime;
    EX_MontP_Init_OPT2_AS_GENE(&prime,  p);
    poly1=createOneWrapperPoly(N, p1dgs, p1Buffer);
    poly2=createOneWrapperPoly(N, p2dgs, p2Buffer);

    res = EX_Resultant_Multi(poly1, poly2, N, &prime);
    freeOneWrapperPoly(poly1);
    freeOneWrapperPoly(poly2);
    create_pdeg_coef_Vec (pdegVec, coefVec, res);
    Nnew[0] = N(res);
    EX_FreeOnePoly(res);

    if (Interrupted==1) mymsg+=-10;
    if (PrimeError==1) mymsg+=-100;
    return mymsg;
}

// WP09
// Maple connector function of evaluating a multivariate polynomial using TFT.
// This function does handle the case where M < N. 
#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
TFTMultiWrapCN(sfixn *Nmp, sfixn *es, sfixn *ls, sfixn Esz, sfixn *E,
	           sfixn *fdgs, sfixn fsz, sfixn *fBuffer, sfixn *rootsPtr)
{
    sfixn N = Nmp[0], M = Nmp[1], p = Nmp[2];
    preFFTRep *poly;
    MONTP_OPT2_AS_GENE prime;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = createOneWrapperPoly(N, fdgs, fBuffer);
    TFTEvalMultiD(M, es, ls, E, poly, rootsPtr, &prime);
    
    freeOneWrapperPoly(poly);
}

// Maple connector function of interpolating a multivariate polynomial using TFT
// This function does handle the case where M < N. 
// NOTE : sVsz and dVsz are not in use. 

#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
InvTFTMultiWrapCN(sfixn *Nmp, sfixn *es, sfixn *ls, sfixn Esz, sfixn *E, sfixn dVsz,
                  sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr)
{
    sfixn N = Nmp[0], M = Nmp[1], p = Nmp[2];
    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);

    poly = TFTInterpMultiD(N, M, es, ls, E, rootsPtr, &prime);

    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);
}

// N is the number of variable of the input polynomial 
// (N+1, Edims1/Edims2), dims is the sizes of E. E should has the same dimension as the input.
// OUTPUT: S.
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
SubResultantChainsTFT(sfixn N, sfixn M, sfixn w, sfixn Ssz, sfixn *S, sfixn *ls1,
                      sfixn E1sz, sfixn *E1, sfixn *ls2, sfixn E2sz, sfixn*E2, sfixn p)
{
    MONTP_OPT2_AS_GENE prime;
    sfixn mymsg = 0;
    if (testPrimeNumber(p)==0) return -200;
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    assert(N == M+1);
    getSubResultantChainsTFT(N, M, w, Ssz, S, ls1, E1, ls2, E2, &prime);
    if (Interrupted==1) mymsg += -10;
    if (PrimeError==1) mymsg += -100;
    return mymsg;
}

// Maple connector function of interpolate a coefficient from the sub-resultant via inverse TFT
#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
InvTFTIthDthMultiWrapCN(sfixn *Nmp, sfixn ith, sfixn dth, sfixn w, sfixn subslicesz,
                        sfixn *subslicees, sfixn *subslicedims, sfixn Ssz, sfixn *S, 
                        sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr)
{
    MONTP_OPT2_AS_GENE prime;
    preFFTRep* poly;
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = interpIthDthSliceTFT(ith, dth, N, m, w, subslicesz, subslicees,
                                subslicedims, Ssz, S, rootsPtr, &prime);
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);

    freePoly(poly);
    my_free(poly);
}

// Maple connector function of interpolate a polynomial from the sub-resultant via inverse FFT
#ifdef WINDOWS
__declspec(dllexport) void __stdcall
#else
void
#endif
InvTFTIthMultiWrapCN(sfixn *Nmp, sfixn ith, sfixn w, sfixn slicesz, sfixn *slicees,
                     sfixn *slicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
                     sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr)
{
    MONTP_OPT2_AS_GENE prime;
    preFFTRep* poly;
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = interpIthSliceTFT(ith, N, m, w, slicesz, slicees, slicedims,
                             Ssz, S, rootsPtr, &prime);
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);

    freePoly(poly);
    my_free(poly);
}


// Maple connector function of interpolate next leading coefficient from the sub-resultant via inverse TFT
#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
InvTFTNextDefectiveLCMultiWrapCN(sfixn *Nmp, sfixn start, sfixn w, sfixn subslicesz, sfixn *subslicees,
                                 sfixn *subslicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
                                 sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr)
{
    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly;
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    int nexti, *nextiAddr = &nexti;

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = interpNextCandidateSliceLCTFTDefective(nextiAddr, start, N, m, w, subslicesz, subslicees,
                                                  subslicedims, Ssz, S, rootsPtr, &prime);

    if (poly==NULL) return -1;
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);

    freePoly(poly);
    my_free(poly);
    return nexti;
}

// Maple connector function of interpolate next leading coefficient from the sub-resultant via inverse TFT

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
InvTFTNextLCMultiWrapCN(sfixn *Nmp, sfixn start, sfixn w, sfixn subslicesz, sfixn *subslicees,
                        sfixn *subslicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
                        sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr)
{
    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly;
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    int nexti, *nextiAddr = &nexti;

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = interpNextCandidateSliceLCTFT(nextiAddr, start, N, m, w,subslicesz, subslicees,
                                         subslicedims, Ssz, S, rootsPtr, &prime);

    if (poly==NULL) return -1;
    create_pdeg_coef_Vec (pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);
    return nexti;
}

#ifdef _mcompile_

/* MapleGcArray management */

/* pointer to kernel vector, defined in mplshlib.h */
static MKernelVector KV = NULL;

static void M_DECL MarkArray(ALGEB p) {
    sfixn *pp;
    pp = (sfixn *)MapleToPointer(KV, p);
    if ( pp ) MapleGcMark(KV, p);
}

// The function to be called by maple GC.
static void M_DECL DisposeArray(ALGEB p) {
    sfixn *pp;
    pp = (sfixn *)MapleToPointer(KV, p);
    if ( pp ) { 
        my_free(pp); 
	// printf("DisposeArray called to free\n");
    }
}

static void M_DECL NopFunction() {
    /* do nothing */
    // printf("NopFunction called to free\n");
}

// Convert a maple DAG integer to sfxin 
static sfixn M_DECL ToSignedFixedNum(MKernelVector kv, ALGEB p) {
    #ifdef TRY64
    return (sfixn) MapleToInteger64(kv, p);
    #else
    return (sfixn) MapleToInteger32(kv, p);
    #endif
}

/* 
static sfixn * M_DECL ToSignedFixedPtr(MKernelVector kv, ALGEB p) {
    return (sfixn *) RTableDataBlock(kv, p);
}
*/

/* MapleGcArray:
 *
 * Allocate and deallocate an array in C, but visible to Maple.
 *
 * Calling sequences:
 *
 * (1)  val := MapleGcArray("malloc", size);
 * (2)  val := MapleGcArray("free", val);
 *
 * Typical usage:
 *
 * Initially, create a C-array and keep an handle in maple
 *
 *      val := MapleGcArray("malloc", 10);
 *
 * There are two ways to free the allocated memory in C.
 *
 * (1) Free it explicitly in Maple:
 *
 *     if (val <> 0) then val := MapleGcArray("free", val); end if;
 *
 *     where val is the return value during the array creation.
 *
 * (2) Do nothing. GC will call DisposeArray to free the memory.
 * 
 * No argument checking provided. Call only by modpn warp functions.
 *  
 */
#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
MapleGcArray_ALGEB(MKernelVector kv, ALGEB *args)
{
    M_INT argc;
    ALGEB  val;   /* the result DAG */
    sfixn *vec;   /* the first element of new allocated array */
    sfixn    n;   /* the size of new allocated array*/
    char *option; /* command type */

    argc = MapleNumArgs(kv, (ALGEB)args);
    if ( argc != 2 ) {
        MapleRaiseError(kv, "two arguments expected");
        return ToMapleNULLPointer(kv);
    }

    option = MapleToString(kv, args[1]);
    if ( strcmp(option, "malloc" ) == 0 ) {
        /* create a Gc array */
        n = ToSignedFixedNum(kv, args[2]);
        assert(n > 0);
        vec = (sfixn *)my_calloc(n, sizeof(sfixn));
        // printf("Malloc::n=%d\n", n);
        // printf("Malloc::total bytes=%d\n", n*sizeof(sfixn));
        // printf("Malloc::Addresss of vec = %p\n", vec);
        if ( vec == NULL )
            MapleRaiseError1(kv, "fail to allocate an array of size %1", args[1]);
        else {
            KV = kv;
            val = ToMaplePointer(kv, (void *)vec, (M_INT)&MarkArray);
            MaplePointerSetMarkFunction(kv, val, MarkArray);
            MaplePointerSetDisposeFunction(kv, val, DisposeArray);
            return val;
        }
    } else if ( strcmp(option, "free") == 0 ) {
        /* free a Gc array */
        if( !IsMaplePointer(kv, args[2]) ||
            MaplePointerType(kv, args[2]) != (M_INT)&MarkArray)
        {
            MapleRaiseError(kv, "gc array expected");
            return ToMapleNULLPointer(kv);
        }
        vec = (sfixn *)MapleToPointer(kv, args[2]);
        if ( vec ) {
            // printf("free the memory manually.\n");
            my_free(vec);
            /* do not call the default DisposeArray, hence free twice */
            MaplePointerSetDisposeFunction(kv, args[2], NopFunction);
            return ToMapleNULLPointer(kv);
        }

    } else {
        MapleRaiseError1(kv, "unrecognized option %1", args[1]);
    }

    return ToMapleNULLPointer(kv);
}

/*  FFT based SCUBE construction  */

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
SubResultantChains_ALGEB(MKernelVector kv, ALGEB *args)
// the input parameters are :
// sfixn N, sfixn w, sfixn Ssz, sfixn *S, sfixn *Edims1, sfixn E1sz,
// sfixn *E1, sfixn *Edims2, sfixn E2sz, sfixn*E2, sfixn p
// return : sfixn
// MaplePointers : S, E1, E2
{
    sfixn  N       = ToSignedFixedNum(kv, args[1]);
    sfixn  w       = ToSignedFixedNum(kv, args[2]);
    sfixn  Ssz     = ToSignedFixedNum(kv, args[3]);
    sfixn *S       = (sfixn *)MapleToPointer(kv, args[4]);
    sfixn *Edims1  = (sfixn *)RTableDataBlock(kv, args[5]);
 /* sfixn  E1sz    = ToSignedFixedNum(kv, args[6]); */
    sfixn *E1      = (sfixn *)MapleToPointer(kv, args[7]);
    sfixn *Edims2  = (sfixn *)RTableDataBlock(kv, args[8]);
 /* sfixn  E2sz    = ToSignedFixedNum(kv, args[9]); */
    sfixn *E2      = (sfixn *)MapleToPointer(kv, args[10]);
    sfixn  p       = ToSignedFixedNum(kv, args[11]);

    MONTP_OPT2_AS_GENE prime;
    sfixn mymsg = 0;

    #if DEBUG 
    if( ! IsMaplePointer(kv, args[4])  ||
        ! IsMaplePointer(kv, args[7])  ||
        ! IsMaplePointer(kv, args[10]) ||
        MaplePointerType(kv, args[4])  != (M_INT)&MarkArray ||
        MaplePointerType(kv, args[7])  != (M_INT)&MarkArray ||
        MaplePointerType(kv, args[10]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleInteger(kv, (long)(-10));
    }
    #endif 

    if (testPrimeNumber(p)==0) return ToMapleInteger(kv, (long)(-200));
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    getSubResultantChains(N, w, Ssz, S, Edims1, E1, Edims2, E2, &prime);
    if (Interrupted==1) mymsg += -10;
    if (PrimeError==1) mymsg += -100;
    return ToMapleInteger(kv, (long)mymsg);
}

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
// the input parameters are :
// sfixn *Nmp, sfixn *es, sfixn *dims, sfixn Esz, sfixn *E,
// sfixn *fdgs, sfixn fsz, sfixn *fBuffer, sfixn *rootsPtr
// return : void
// MaplePointers : E
DftMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args)
{
    sfixn *Nmp      = (sfixn *) RTableDataBlock(kv, args[1]); 
    sfixn *es       = (sfixn *) RTableDataBlock(kv, args[2]);
    sfixn *dims     = (sfixn *) RTableDataBlock(kv, args[3]);
 /* sfixn  Esz      = ToSignedFixedNum(kv, args[4]); */
    sfixn *E        = (sfixn *) MapleToPointer(kv, args[5]);
    sfixn *fdgs     = (sfixn *) RTableDataBlock(kv, args[6]);  
 /* sfixn  fsz      = ToSignedFixedNum(kv, args[7]); */
    sfixn *fBuffer  = (sfixn *) RTableDataBlock(kv, args[8]);
    sfixn *rootsPtr = (sfixn *) RTableDataBlock(kv, args[9]);

    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    preFFTRep *poly;
    MONTP_OPT2_AS_GENE prime;
    
    #if DEBUG 
    if( ! IsMaplePointer(kv, args[5]) ||
        MaplePointerType(kv, args[5]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleNULL(kv);
    }
    #endif 

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = createOneWrapperPoly(N, fdgs, fBuffer);
    fastDftMulti_test(m, es, dims, E, poly, rootsPtr, &prime);
    freeOneWrapperPoly(poly);

    return ToMapleNULL(kv);
}

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
InvDftMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args)
{
    sfixn *Nmp      = (sfixn *) RTableDataBlock(kv, args[1]); 
    sfixn *es       = (sfixn *) RTableDataBlock(kv, args[2]);
    sfixn *dims     = (sfixn *) RTableDataBlock(kv, args[3]);
 /* sfixn  Esz      = ToSignedFixedNum(kv, args[4]); */
    sfixn *E        = (sfixn *) MapleToPointer(kv, args[5]);
 /* sfixn  dVsz     = ToSignedFixedNum(kv, args[6]); */
    sfixn *pdegVec  = (sfixn *) RTableDataBlock(kv, args[7]);
 /* sfixn  cVsz     = ToSignedFixedNum(kv, args[8]); */
    sfixn *coefVec  = (sfixn *) RTableDataBlock(kv, args[9]);
    sfixn *rootsPtr = (sfixn *) RTableDataBlock(kv, args[10]);
    
     
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly;

    #if DEBUG 
    if( ! IsMaplePointer(kv, args[5]) ||
        MaplePointerType(kv, args[5]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleNULL(kv);
    }
    #endif 

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = fastInvDftMulti_test(N, m, es, dims, E, rootsPtr, &prime);
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);

    return ToMapleNULL(kv);
}

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
InvDftIthDthMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args)
{
    sfixn *Nmp           = (sfixn *) RTableDataBlock(kv, args[1]);
    sfixn ith            = ToSignedFixedNum(kv, args[2]);
    sfixn dth            = ToSignedFixedNum(kv, args[3]);
    sfixn w              = ToSignedFixedNum(kv, args[4]);
    sfixn subslicesz     = ToSignedFixedNum(kv, args[5]);
    sfixn *subslicees    = (sfixn *) RTableDataBlock(kv, args[6]);
    sfixn *subslicedims  = (sfixn *) RTableDataBlock(kv, args[7]);
    sfixn Ssz            = ToSignedFixedNum(kv, args[8]);
    sfixn *S             = (sfixn *) MapleToPointer(kv, args[9]);
 /* sfixn dVsz           = ToSignedFixedNum(kv, args[10]); */
    sfixn *pdegVec       = (sfixn *) RTableDataBlock(kv, args[11]);
 /* sfixn cVsz           = ToSignedFixedNum(kv, args[12]); */
    sfixn *coefVec       = (sfixn *) RTableDataBlock(kv, args[13]);
    sfixn *rootsPtr      = (sfixn *) RTableDataBlock(kv, args[14]);

    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly;

    #if DEBUG 
    if( ! IsMaplePointer(kv, args[9]) ||
        MaplePointerType(kv, args[9]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleNULL(kv);
    }
    #endif 
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = interpIthDthSliceDFT(ith, dth, N, m, w, subslicesz, subslicees,
        subslicedims, Ssz, S, rootsPtr, &prime);

    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);

    return ToMapleNULL(kv);
}

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
InvDftIthMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args)
{
    sfixn *Nmp           = (sfixn *) RTableDataBlock(kv, args[1]);
    sfixn ith            = ToSignedFixedNum(kv, args[2]);
    sfixn w              = ToSignedFixedNum(kv, args[3]);
    sfixn slicesz        = ToSignedFixedNum(kv, args[4]);
    sfixn *slicees       = (sfixn *) RTableDataBlock(kv, args[5]);
    sfixn *slicedims     = (sfixn *) RTableDataBlock(kv, args[6]);
    sfixn Ssz            = ToSignedFixedNum(kv, args[7]);
    sfixn *S             = (sfixn *) MapleToPointer(kv, args[8]);
 /* sfixn dVsz           = ToSignedFixedNum(kv, args[9]); */
    sfixn *pdegVec       = (sfixn *) RTableDataBlock(kv, args[10]);
 /* sfixn cVsz           = ToSignedFixedNum(kv, args[11]); */
    sfixn *coefVec       = (sfixn *) RTableDataBlock(kv, args[12]);
    sfixn *rootsPtr      = (sfixn *) RTableDataBlock(kv, args[13]);

    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly;

    #if DEBUG 
    if( ! IsMaplePointer(kv, args[8]) ||
        MaplePointerType(kv, args[8]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleNULL(kv);
    }
    #endif 
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = interpIthSliceDFT(ith, N, m, w, slicesz, slicees, slicedims,
                             Ssz, S, rootsPtr, &prime);
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);

    return ToMapleNULL(kv);
}

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
InvDftNextDefectiveLCMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args)
{
    sfixn *Nmp           = (sfixn *) RTableDataBlock(kv, args[1]);
    sfixn start          = ToSignedFixedNum(kv, args[2]);     
    sfixn w              = ToSignedFixedNum(kv, args[3]);
    sfixn subslicesz     = ToSignedFixedNum(kv, args[4]);
    sfixn *subslicees    = (sfixn *) RTableDataBlock(kv, args[5]);
    sfixn *subslicedims  = (sfixn *) RTableDataBlock(kv, args[6]);   
    sfixn Ssz            = ToSignedFixedNum(kv, args[7]);   
    sfixn *S             = (sfixn *) MapleToPointer(kv, args[8]);
 /* sfixn dVsz           = ToSignedFixedNum(kv, args[9]); */
    sfixn *pdegVec       = (sfixn *) RTableDataBlock(kv, args[10]);
 /* sfixn cVsz           = ToSignedFixedNum(kv, args[11]); */
    sfixn *coefVec       = (sfixn *) RTableDataBlock(kv, args[12]);
    sfixn *rootsPtr      = (sfixn *) RTableDataBlock(kv, args[13]);

    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly;
    int nexti, *nextiAddr=&nexti;
    
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    #if DEBUG 
    if( ! IsMaplePointer(kv, args[8]) ||
        MaplePointerType(kv, args[8]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleInteger(kv, (long)(-1));
    }
    #endif 
    poly = interpNextCandidateSliceLCDFTDefective(nextiAddr, start, N, m, w,
              subslicesz, subslicees,subslicedims, Ssz, S, rootsPtr, &prime);

    if (poly==NULL) return ToMapleInteger(kv, (long)(-1));
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);

    return ToMapleInteger(kv, (long)nexti);
}

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
InvDftNextLCMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args)
{
    sfixn *Nmp           = (sfixn *) RTableDataBlock(kv, args[1]);
    sfixn start          = ToSignedFixedNum(kv, args[2]);     
    sfixn w              = ToSignedFixedNum(kv, args[3]);
    sfixn subslicesz     = ToSignedFixedNum(kv, args[4]);
    sfixn *subslicees    = (sfixn *) RTableDataBlock(kv, args[5]);
    sfixn *subslicedims  = (sfixn *) RTableDataBlock(kv, args[6]);   
    sfixn Ssz            = ToSignedFixedNum(kv, args[7]);   
    sfixn *S             = (sfixn *) MapleToPointer(kv, args[8]);
 /* sfixn dVsz           = ToSignedFixedNum(kv, args[9]); */
    sfixn *pdegVec       = (sfixn *) RTableDataBlock(kv, args[10]);
 /* sfixn cVsz           = ToSignedFixedNum(kv, args[11]); */
    sfixn *coefVec       = (sfixn *) RTableDataBlock(kv, args[12]);
    sfixn *rootsPtr      = (sfixn *) RTableDataBlock(kv, args[13]);

    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly;
    int nexti, *nextiAddr = &nexti;

    #if DEBUG 
    if( ! IsMaplePointer(kv, args[8]) ||
        MaplePointerType(kv, args[8]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleInteger(kv, (long)(-1));
    }
    #endif 
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = interpNextCandidateSliceLCDFT(nextiAddr, start, N, m, w, subslicesz,
                             subslicees, subslicedims, Ssz, S, rootsPtr, &prime);

    if (poly==NULL) return ToMapleInteger(kv, (long)(-1));
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);

    return ToMapleInteger(kv, (long)nexti);
}

/*  TFT based SCUBE construction  */

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
SubResultantChainsTFT_ALGEB(MKernelVector kv, ALGEB *args)
// the input parameters are :
// sfixn N, sfixn M, sfixn w, sfixn Ssz, sfixn *S, sfixn *ls1, sfixn E1sz, 
// sfixn *E1, sfixn *ls2, sfixn E2sz, sfixn *E2, sfixn p;
// return : sfixn
// MaplePointers : E1, E2, S
{
    sfixn  N    = ToSignedFixedNum(kv, args[1]);
    sfixn  M    = ToSignedFixedNum(kv, args[2]);
    sfixn  w    = ToSignedFixedNum(kv, args[3]);
    sfixn  Ssz  = ToSignedFixedNum(kv, args[4]);
    sfixn *S    = (sfixn *)MapleToPointer(kv, args[5]);
    sfixn *ls1  = (sfixn *)RTableDataBlock(kv, args[6]);
    sfixn  E1sz = ToSignedFixedNum(kv, args[7]);   
    sfixn *E1   = (sfixn *)MapleToPointer(kv, args[8]);
    sfixn *ls2  = (sfixn *)RTableDataBlock(kv, args[9]);
    sfixn  E2sz = ToSignedFixedNum(kv, args[10]);   
    sfixn *E2   = (sfixn *)MapleToPointer(kv, args[11]);
    sfixn  p    = ToSignedFixedNum(kv, args[12]);
    
    MONTP_OPT2_AS_GENE prime;
    sfixn mymsg = 0;
    sfixn i;

    #if DEBUG 
    if( ! IsMaplePointer(kv, args[5])  ||
        ! IsMaplePointer(kv, args[8])  ||
        ! IsMaplePointer(kv, args[11]) ||
        MaplePointerType(kv, args[5])  != (M_INT)&MarkArray ||
        MaplePointerType(kv, args[8])  != (M_INT)&MarkArray ||
        MaplePointerType(kv, args[11]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleInteger(kv, (long)(-10));
    }
    for (i=0; i<Ssz; ++i) assert(S[i]==0);
    #endif 

    if (testPrimeNumber(p)==0) return ToMapleInteger(kv, (long)-200);
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    assert(N == M+1);

    getSubResultantChainsTFT(N, M, w, Ssz, S, ls1, E1, ls2, E2, &prime);

    if (Interrupted==1) mymsg += -10;
    if (PrimeError==1) mymsg += -100;
    return ToMapleInteger(kv, (long)mymsg);
}

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
TFTMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args)
// the input parameters are :
// sfixn *Nmp, sfixn *es, sfixn *ls, sfixn Esz, sfixn *E,
// sfixn *fdgs, sfixn fsz, sfixn *fBuffer, sfixn *rootsPtr;
// return : void
// MaplePointers : E
{
    sfixn *Nmp      = (sfixn *) RTableDataBlock(kv, args[1]); 
    sfixn *es       = (sfixn *) RTableDataBlock(kv, args[2]);
    sfixn *ls       = (sfixn *) RTableDataBlock(kv, args[3]);
    sfixn  Esz      = ToSignedFixedNum(kv, args[4]); 
    sfixn *E        = (sfixn *) MapleToPointer(kv, args[5]);
    sfixn *fdgs     = (sfixn *) RTableDataBlock(kv, args[6]);  
    sfixn  fsz      = ToSignedFixedNum(kv, args[7]);   
    sfixn *fBuffer  = (sfixn *) RTableDataBlock(kv, args[8]);
    sfixn *rootsPtr = (sfixn *) RTableDataBlock(kv, args[9]);

    sfixn N = Nmp[0], M = Nmp[1], p = Nmp[2];
    preFFTRep *poly;
    MONTP_OPT2_AS_GENE prime;
    sfixn i;
    #if DEBUG 
    if( ! IsMaplePointer(kv, args[5]) ||
        MaplePointerType(kv, args[5]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleNULL(kv);
    }
    for (i=0; i<Esz; ++i) assert(E[i]==0);
    #endif 

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = createOneWrapperPoly(N, fdgs, fBuffer);
    TFTEvalMultiD(M, es, ls, E, poly, rootsPtr, &prime);
    freeOneWrapperPoly(poly);

    return ToMapleNULL(kv);
}

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
InvTFTMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args)
// the input parameters are :
// sfixn *Nmp, sfixn *es, sfixn *ls, sfixn Esz, sfixn *E, sfixn dVsz,
// sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr;
// return : void
// MaplePointers : E
{
    sfixn *Nmp      = (sfixn *) RTableDataBlock(kv, args[1]);
    sfixn *es       = (sfixn *) RTableDataBlock(kv, args[2]);
    sfixn *ls       = (sfixn *) RTableDataBlock(kv, args[3]);
 /* sfixn  Esz      = ToSignedFixedNum(kv, args[4]); */
    sfixn *E        = (sfixn *) MapleToPointer(kv, args[5]);
 /* sfixn  dVsz     = ToSignedFixedNum(kv, args[6]); */
    sfixn *pdegVec  = (sfixn *) RTableDataBlock(kv, args[7]);
 /* sfixn cVsz      = ToSignedFixedNum(kv, args[8]); */
    sfixn *coefVec  = (sfixn *) RTableDataBlock(kv, args[9]);
    sfixn *rootsPtr = (sfixn *) RTableDataBlock(kv, args[10]);

    sfixn N = Nmp[0], M = Nmp[1], p = Nmp[2];
    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly;

    #if DEBUG 
    if( ! IsMaplePointer(kv, args[5]) ||
        MaplePointerType(kv, args[5]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleNULL(kv);
    }
    #endif 

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = TFTInterpMultiD(N, M, es, ls, E, rootsPtr, &prime);

    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);
    
    return ToMapleNULL(kv);
}

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
InvTFTIthDthMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args)
// the input parameters are :
// sfixn *Nmp, sfixn ith, sfixn dth, sfixn w, sfixn subslicesz,
// sfixn *subslicees, sfixn *subslicedims, sfixn Ssz, sfixn *S,
// sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr;
// return : void
// MaplePointers : S
{   
    sfixn *Nmp           = (sfixn *) RTableDataBlock(kv,  args[1]);
    sfixn ith            = ToSignedFixedNum(kv, args[2]);
    sfixn dth            = ToSignedFixedNum(kv, args[3]);
    sfixn w              = ToSignedFixedNum(kv, args[4]);
    sfixn subslicesz     = ToSignedFixedNum(kv, args[5]);
    sfixn *subslicees    = (sfixn *) RTableDataBlock(kv, args[6]);
    sfixn *subslicedims  = (sfixn *) RTableDataBlock(kv, args[7]);
    sfixn Ssz            = ToSignedFixedNum(kv, args[8]);
    sfixn *S             = (sfixn *) MapleToPointer(kv, args[9]);
 /* sfixn dVsz           = ToSignedFixedNum(kv, args[10]); */
    sfixn *pdegVec       = (sfixn *) RTableDataBlock(kv, args[11]);
 /* sfixn cVsz           = ToSignedFixedNum(kv, args[12]); */
    sfixn *coefVec       = (sfixn *) RTableDataBlock(kv, args[13]);
    sfixn *rootsPtr      = (sfixn *) RTableDataBlock(kv, args[14]);

    MONTP_OPT2_AS_GENE prime;
    preFFTRep* poly;
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];

    #if DEBUG 
    if( ! IsMaplePointer(kv, args[9]) ||
        MaplePointerType(kv, args[9]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleNULL(kv);
    }
    #endif 

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = interpIthDthSliceTFT(ith, dth, N, m, w, subslicesz, subslicees,
                                subslicedims, Ssz, S, rootsPtr, &prime);
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);

    freePoly(poly);
    my_free(poly);
    return ToMapleNULL(kv);
}

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
InvTFTIthMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args)
// the input parameters are :
// sfixn *Nmp, sfixn ith, sfixn w, sfixn slicesz, sfixn *slicees,
// sfixn *slicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
// sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr;
// return : void
// MaplePointers :  S
{
    sfixn *Nmp           = (sfixn *) RTableDataBlock(kv, args[1]);
    sfixn ith            = ToSignedFixedNum(kv, args[2]);
    sfixn w              = ToSignedFixedNum(kv, args[3]);
    sfixn slicesz        = ToSignedFixedNum(kv, args[4]);
    sfixn *slicees       = (sfixn *) RTableDataBlock(kv, args[5]);
    sfixn *slicedims     = (sfixn *) RTableDataBlock(kv, args[6]);
    sfixn Ssz            = ToSignedFixedNum(kv, args[7]);
    sfixn *S             = (sfixn *) MapleToPointer(kv, args[8]);
 /* sfixn dVsz           = ToSignedFixedNum(kv, args[9]); */
    sfixn *pdegVec       = (sfixn *) RTableDataBlock(kv, args[10]);
 /* sfixn cVsz           = ToSignedFixedNum(kv, args[11]); */
    sfixn *coefVec       = (sfixn *) RTableDataBlock(kv, args[12]);
    sfixn *rootsPtr      = (sfixn *) RTableDataBlock(kv, args[13]);

    MONTP_OPT2_AS_GENE prime;
    preFFTRep* poly;
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];

    #if DEBUG 
    if( ! IsMaplePointer(kv, args[8]) ||
        MaplePointerType(kv, args[8]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleNULL(kv);
    }
    #endif 

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = interpIthSliceTFT(ith, N, m, w, slicesz, slicees, slicedims,
                             Ssz, S, rootsPtr, &prime);
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);

    freePoly(poly);
    my_free(poly);
    return ToMapleNULL(kv);
}

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
InvTFTNextDefectiveLCMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args)
// the input parameters are :
// sfixn *Nmp, sfixn start, sfixn w, sfixn subslicesz, sfixn *subslicees,
// sfixn *subslicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
// sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr;
// return : sfixn 
// MaplePointers : S
{
    sfixn *Nmp           = (sfixn *) RTableDataBlock(kv, args[1]);
    sfixn start          = ToSignedFixedNum(kv, args[2]);     
    sfixn w              = ToSignedFixedNum(kv, args[3]);
    sfixn subslicesz     = ToSignedFixedNum(kv, args[4]);
    sfixn *subslicees    = (sfixn *) RTableDataBlock(kv, args[5]);
    sfixn *subslicedims  = (sfixn *) RTableDataBlock(kv, args[6]);   
    sfixn Ssz            = ToSignedFixedNum(kv, args[7]);   
    sfixn *S             = (sfixn *) MapleToPointer(kv, args[8]);
 /* sfixn dVsz           = ToSignedFixedNum(kv, args[9]); */
    sfixn *pdegVec       = (sfixn *) RTableDataBlock(kv, args[10]);
 /* sfixn cVsz           = ToSignedFixedNum(kv, args[11]); */
    sfixn *coefVec       = (sfixn *) RTableDataBlock(kv, args[12]);
    sfixn *rootsPtr      = (sfixn *) RTableDataBlock(kv, args[13]);

    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly;
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    int nexti, *nextiAddr = &nexti;

    #if DEBUG 
    if( ! IsMaplePointer(kv, args[8]) ||
        MaplePointerType(kv, args[8]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleInteger(kv, (long) (-1));
    }
    #endif 

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = interpNextCandidateSliceLCTFTDefective(nextiAddr, start, N, m, w,
        subslicesz, subslicees, subslicedims, Ssz, S, rootsPtr, &prime);

    if (poly==NULL) return ToMapleInteger(kv, (long) (-1));
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);

    return ToMapleInteger(kv, (long)nexti);;
}

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
InvTFTNextLCMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args)
// the input parameters are :
// sfixn *Nmp, sfixn start, sfixn w, sfixn subslicesz, sfixn *subslicees,
// sfixn *subslicedims, sfixn Ssz, sfixn *S, sfixn dVsz, sfixn *pdegVec,
// sfixn cVsz, sfixn *coefVec, sfixn *rootsPtr;
// return : sfixn
// MaplePointers : S
{
    sfixn *Nmp           = (sfixn *) RTableDataBlock(kv, args[1]);
    sfixn start          = ToSignedFixedNum(kv, args[2]);     
    sfixn w              = ToSignedFixedNum(kv, args[3]);
    sfixn subslicesz     = ToSignedFixedNum(kv, args[4]);
    sfixn *subslicees    = (sfixn *) RTableDataBlock(kv, args[5]);
    sfixn *subslicedims  = (sfixn *) RTableDataBlock(kv, args[6]);   
    sfixn Ssz            = ToSignedFixedNum(kv, args[7]);   
    sfixn *S             = (sfixn *) MapleToPointer(kv, args[8]);
 /* sfixn dVsz           = ToSignedFixedNum(kv, args[9]); */
    sfixn *pdegVec       = (sfixn *) RTableDataBlock(kv, args[10]);
 /* sfixn cVsz           = ToSignedFixedNum(kv, args[11]); */
    sfixn *coefVec       = (sfixn *) RTableDataBlock(kv, args[12]);
    sfixn *rootsPtr      = (sfixn *) RTableDataBlock(kv, args[13]);

    MONTP_OPT2_AS_GENE prime;
    preFFTRep *poly;
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    int nexti, *nextiAddr = &nexti;

    #if DEBUG 
    if( ! IsMaplePointer(kv, args[8]) ||
        MaplePointerType(kv, args[8]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleInteger(kv, (long) (-1));
    }
    #endif 

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    poly = interpNextCandidateSliceLCTFT(nextiAddr, start, N, m, w, subslicesz, 
        subslicees, subslicedims, Ssz, S, rootsPtr, &prime);

    if (poly==NULL) return ToMapleInteger(kv, (long) (-1));
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);

    return ToMapleInteger(kv, (long)nexti);
}

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
FastEvalMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args)
{
    sfixn *Nmp        = (sfixn *) RTableDataBlock(kv, args[1]);  
    sfixn *ptPHWDSZ   = (sfixn *) RTableDataBlock(kv, args[2]);
    sfixn *bounds     = (sfixn *) RTableDataBlock(kv, args[3]);
    sfixn *pts_s      = (sfixn *) RTableDataBlock(kv, args[4]);
    sfixn *h_s        = (sfixn *) RTableDataBlock(kv, args[5]);
    sfixn *W_s        = (sfixn *) RTableDataBlock(kv, args[6]);
    sfixn *NoNodes_s  = (sfixn *) RTableDataBlock(kv, args[7]);
    sfixn *Bases_s    = (sfixn *) RTableDataBlock(kv, args[8]);
    sfixn *data_s     = (sfixn *) RTableDataBlock(kv, args[9]);
    sfixn *dims       = (sfixn *) RTableDataBlock(kv, args[10]);
 /* sfixn Esz         = ToSignedFixedNum(kv, args[11]); */
    sfixn *E          = (sfixn *) MapleToPointer(kv, args[12]);
    sfixn *fdgs       = (sfixn *) RTableDataBlock(kv, args[13]);
 /* sfixn fsz         = ToSignedFixedNum(kv, args[14]); */
    sfixn *fBuffer    = (sfixn *) RTableDataBlock(kv, args[15]);

    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    sfixn pts_sSz = ptPHWDSZ[0], h_sSz = ptPHWDSZ[1];
    sfixn WNB_sSz = ptPHWDSZ[2], data_sSz = ptPHWDSZ[3]; 
    preFFTRep *poly;
    MONTP_OPT2_AS_GENE prime;
    PTS_TREE *pts_tree;
    sfixn mymsg = 0;

    #if DEBUG 
    if( ! IsMaplePointer(kv, args[12]) ||
        MaplePointerType(kv, args[12]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleInteger(kv, (long) (-1));
    }
    #endif 

    if (testPrimeNumber(p)==0) return ToMapleInteger(kv, (long)(-200));
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    pts_tree = createWrapperPTS_TREE(N, m, bounds, pts_sSz, pts_s, h_sSz, h_s,
                       WNB_sSz, W_s, NoNodes_s, Bases_s, data_sSz, data_s, p);
    poly = createOneWrapperPoly(N, fdgs, fBuffer);
    fastEvalMulti_test(m, dims, E, poly, pts_tree->trees, &prime);
    freeOneWrapperPoly(poly);
    freeWrapperPTS_TREE(pts_tree);
    if (Interrupted==1) mymsg += -10;
    if (PrimeError==1) mymsg += -100;
    return ToMapleInteger(kv, (long)mymsg);
}

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
FastInterpMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args)
{
    sfixn *Nmp        = (sfixn *)RTableDataBlock(kv, args[1]);  
    sfixn *ptPHWDSZ   = (sfixn *)RTableDataBlock(kv, args[2]);
    sfixn *bounds     = (sfixn *)RTableDataBlock(kv, args[3]);
    sfixn *pts_s      = (sfixn *)RTableDataBlock(kv, args[4]);
    sfixn *h_s        = (sfixn *)RTableDataBlock(kv, args[5]);
    sfixn *W_s        = (sfixn *)RTableDataBlock(kv, args[6]);
    sfixn *NoNodes_s  = (sfixn *)RTableDataBlock(kv, args[7]);
    sfixn *Bases_s    = (sfixn *)RTableDataBlock(kv, args[8]);
    sfixn *data_s     = (sfixn *)RTableDataBlock(kv, args[9]);
    sfixn *dims       = (sfixn *)RTableDataBlock(kv, args[10]);
 /* sfixn Esz         = ToSignedFixedNum(kv, args[11]); */
    sfixn *E          = (sfixn *)MapleToPointer(kv, args[12]);
 /* sfixn dVsz        = ToSignedFixedNum(kv, args[13]); */
    sfixn *pdegVec    = (sfixn *)RTableDataBlock(kv, args[14]);
 /* sfixn cVsz        = ToSignedFixedNum(kv, args[15]); */
    sfixn *coefVec    = (sfixn *)RTableDataBlock(kv, args[16]);

    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    sfixn pts_sSz = ptPHWDSZ[0], h_sSz = ptPHWDSZ[1];
    sfixn WNB_sSz = ptPHWDSZ[2], data_sSz = ptPHWDSZ[3]; 
    MONTP_OPT2_AS_GENE prime;
    PTS_TREE *pts_tree;
    preFFTRep *poly;
    sfixn mymsg = 0;

    #if DEBUG 
    if( ! IsMaplePointer(kv, args[12]) ||
        MaplePointerType(kv, args[12]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleInteger(kv, (long) (-1));
    }
    #endif 

    if (testPrimeNumber(p)==0) return ToMapleInteger(kv, (long)(-200));
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    pts_tree = createWrapperPTS_TREE(N, m, bounds, pts_sSz, pts_s, h_sSz, h_s,
                       WNB_sSz, W_s, NoNodes_s, Bases_s, data_sSz, data_s, p);

    poly = fastInterpMulti_test(N, m, dims, E, pts_tree->ptsPtr, pts_tree->trees, &prime);

    if (poly==NULL) {
        freeWrapperPTS_TREE(pts_tree);
        if (Interrupted==1) mymsg+=-10;
        if (PrimeError==1) mymsg+=-100;
        return ToMapleInteger(kv, (long)mymsg);
    }
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);
    freeWrapperPTS_TREE(pts_tree);
    if (Interrupted==1) mymsg += -10;
    if (PrimeError==1) mymsg += -100;
    return ToMapleInteger(kv, (long)mymsg);
}

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
InterpIthDthMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args)
{
    sfixn *Nmp          = (sfixn *)RTableDataBlock(kv, args[1]); 
    sfixn *ptPHWDSZ     = (sfixn *)RTableDataBlock(kv, args[2]);
    sfixn ith           = ToSignedFixedNum(kv, args[3]);
    sfixn dth           = ToSignedFixedNum(kv, args[4]);
    sfixn *bounds       = (sfixn *)RTableDataBlock(kv, args[5]);
    sfixn *pts_s        = (sfixn *)RTableDataBlock(kv, args[6]);
    sfixn *h_s          = (sfixn *)RTableDataBlock(kv, args[7]);
    sfixn *W_s          = (sfixn *)RTableDataBlock(kv, args[8]);
    sfixn *NoNodes_s    = (sfixn *)RTableDataBlock(kv, args[9]);
    sfixn *Bases_s      = (sfixn *)RTableDataBlock(kv, args[10]);
    sfixn *data_s       = (sfixn *)RTableDataBlock(kv, args[11]);
    sfixn w             = ToSignedFixedNum(kv, args[12]);
    sfixn subslicesz    = ToSignedFixedNum(kv, args[13]);
    sfixn *subslicedims = (sfixn *)RTableDataBlock(kv, args[14]);
    sfixn Ssz           = ToSignedFixedNum(kv, args[15]);
    sfixn *S            = (sfixn *)MapleToPointer(kv, args[16]);
 /* sfixn dVsz          = ToSignedFixedNum(kv, args[17]); */
    sfixn *pdegVec      = (sfixn *)RTableDataBlock(kv, args[18]);
 /* sfixn cVsz          = ToSignedFixedNum(kv, args[19]); */
    sfixn *coefVec      = (sfixn *)RTableDataBlock(kv, args[20]);

    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    sfixn pts_sSz = ptPHWDSZ[0], h_sSz = ptPHWDSZ[1]; 
    sfixn WNB_sSz = ptPHWDSZ[2], data_sSz = ptPHWDSZ[3]; 
    MONTP_OPT2_AS_GENE prime;
    PTS_TREE *pts_tree;
    preFFTRep *poly;

    #if DEBUG 
    if( ! IsMaplePointer(kv, args[16]) ||
        MaplePointerType(kv, args[16]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleNULL(kv);
    }
    #endif 

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    pts_tree = createWrapperPTS_TREE(N, m, bounds, pts_sSz, pts_s, h_sSz, h_s,
        WNB_sSz, W_s, NoNodes_s, Bases_s, data_sSz, data_s, p);

    poly = interpIthDthSlice(ith, dth, N, m, w, subslicesz, subslicedims,
        Ssz, S, pts_tree, &prime);

    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);
    freeWrapperPTS_TREE(pts_tree);
    return ToMapleNULL(kv);
}

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
InterpIthMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args)
{
    sfixn *Nmp       = (sfixn *)RTableDataBlock(kv, args[1]); 
    sfixn *ptPHWDSZ  = (sfixn *)RTableDataBlock(kv, args[2]);
    sfixn ith        = ToSignedFixedNum(kv, args[3]);
    sfixn *bounds    = (sfixn *)RTableDataBlock(kv, args[4]);
    sfixn *pts_s     = (sfixn *)RTableDataBlock(kv, args[5]);
    sfixn *h_s       = (sfixn *)RTableDataBlock(kv, args[6]);
    sfixn *W_s       = (sfixn *)RTableDataBlock(kv, args[7]);
    sfixn *NoNodes_s = (sfixn *)RTableDataBlock(kv, args[8]);
    sfixn *Bases_s   = (sfixn *)RTableDataBlock(kv, args[9]);
    sfixn *data_s    = (sfixn *)RTableDataBlock(kv, args[10]);
    sfixn w          = ToSignedFixedNum(kv, args[11]);
    sfixn slicesz    = ToSignedFixedNum(kv, args[12]);
    sfixn *slicedims = (sfixn *)RTableDataBlock(kv, args[13]);
    sfixn Ssz        = ToSignedFixedNum(kv, args[14]);
    sfixn *S         = (sfixn *)MapleToPointer(kv, args[15]);
 /* sfixn dVsz       = ToSignedFixedNum(kv, args[16]); */
    sfixn *pdegVec   = (sfixn *)RTableDataBlock(kv, args[17]);
 /* sfixn cVsz       = ToSignedFixedNum(kv, args[18]); */
    sfixn *coefVec   = (sfixn *)RTableDataBlock(kv, args[19]);
                     
    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    sfixn pts_sSz = ptPHWDSZ[0], h_sSz = ptPHWDSZ[1];
    sfixn WNB_sSz = ptPHWDSZ[2], data_sSz = ptPHWDSZ[3]; 
    MONTP_OPT2_AS_GENE prime;
    PTS_TREE *pts_tree;
    preFFTRep *poly;

    #if DEBUG 
    if( ! IsMaplePointer(kv, args[15]) ||
        MaplePointerType(kv, args[15]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleNULL(kv);
    }
    #endif 

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);
    pts_tree = createWrapperPTS_TREE(N, m, bounds, pts_sSz, pts_s, h_sSz, h_s,
        WNB_sSz, W_s, NoNodes_s, Bases_s, data_sSz, data_s, p);

    poly = interpIthSlice(ith, N, m, w, slicesz, slicedims, Ssz, S, pts_tree, &prime);
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);
    freeWrapperPTS_TREE(pts_tree);
    return ToMapleNULL(kv);
}

#ifdef WINDOWS
__declspec(dllexport) ALGEB __stdcall
#else
ALGEB
#endif
// the input parameters are :
// sfixn *Nmp, sfixn *ptPHWDSZ, sfixn start, sfixn *bounds, sfixn *pts_s,
// sfixn *h_s, sfixn *W_s, sfixn *NoNodes_s, sfixn *Bases_s, sfixn *data_s,
// sfixn w, sfixn subslicesz, sfixn *subslicedims, sfixn Ssz, sfixn *S,
// sfixn dVsz, sfixn *pdegVec, sfixn cVsz, sfixn *coefVec;
// return : sfixn
// MaplePointers : S
InterpNextLCMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args)
{
    sfixn *Nmp          = (sfixn *)RTableDataBlock(kv, args[1]); 
    sfixn *ptPHWDSZ     = (sfixn *)RTableDataBlock(kv, args[2]);
    sfixn start         = ToSignedFixedNum(kv, args[3]);
    sfixn *bounds       = (sfixn *)RTableDataBlock(kv, args[4]);
    sfixn *pts_s        = (sfixn *)RTableDataBlock(kv, args[5]);
    sfixn *h_s          = (sfixn *)RTableDataBlock(kv, args[6]);
    sfixn *W_s          = (sfixn *)RTableDataBlock(kv, args[7]);
    sfixn *NoNodes_s    = (sfixn *)RTableDataBlock(kv, args[8]);
    sfixn *Bases_s      = (sfixn *)RTableDataBlock(kv, args[9]);
    sfixn *data_s       = (sfixn *)RTableDataBlock(kv, args[10]);
    sfixn w             = ToSignedFixedNum(kv, args[11]);
    sfixn subslicesz    = ToSignedFixedNum(kv, args[12]);
    sfixn *subslicedims = (sfixn *)RTableDataBlock(kv, args[13]);
    sfixn Ssz           = ToSignedFixedNum(kv, args[14]);
    sfixn *S            = (sfixn *)MapleToPointer(kv, args[15]);
 /* sfixn dVsz          = ToSignedFixedNum(kv, args[16]); */
    sfixn *pdegVec      = (sfixn *)RTableDataBlock(kv, args[17]);
 /* sfixn cVsz          = ToSignedFixedNum(kv, args[18]); */
    sfixn *coefVec      = (sfixn *)RTableDataBlock(kv, args[19]);

    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    sfixn pts_sSz = ptPHWDSZ[0], h_sSz = ptPHWDSZ[1];
    sfixn WNB_sSz = ptPHWDSZ[2], data_sSz = ptPHWDSZ[3];
    MONTP_OPT2_AS_GENE prime;
    PTS_TREE *pts_tree;
    preFFTRep *poly;
    int nexti, *nextiAddr = &nexti;

    #if DEBUG 
    if( ! IsMaplePointer(kv, args[15]) ||
        MaplePointerType(kv, args[15]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleInteger(kv, (long)(-1));
    }
    #endif 

    EX_MontP_Init_OPT2_AS_GENE(&prime, p);  
    pts_tree = createWrapperPTS_TREE(N, m, bounds, pts_sSz, pts_s, h_sSz, h_s,
        WNB_sSz, W_s, NoNodes_s, Bases_s, data_sSz, data_s, p);

    poly = interpNextCandidateSliceLC(nextiAddr, start, N, m, w, subslicesz,
        subslicedims, Ssz, S, pts_tree, &prime);

    if (poly==NULL) return ToMapleInteger(kv, (long)(-1));
    create_pdeg_coef_Vec (pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);
    freeWrapperPTS_TREE(pts_tree);
    return ToMapleInteger(kv, (long)nexti);
}

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
// sfixn cVsz, sfixn *coefVec;
// return : sfixn
// MaplePointers : S
InterpNextDefectiveLCMultiWrapCN_ALGEB(MKernelVector kv, ALGEB *args) 
{
    sfixn *Nmp           = (sfixn *)RTableDataBlock(kv, args[1]); 
    sfixn *ptPHWDSZ      = (sfixn *)RTableDataBlock(kv, args[2]);
    sfixn start          = ToSignedFixedNum(kv, args[3]);
    sfixn *bounds        = (sfixn *)RTableDataBlock(kv, args[4]);
    sfixn *pts_s         = (sfixn *)RTableDataBlock(kv, args[5]);
    sfixn *h_s           = (sfixn *)RTableDataBlock(kv, args[6]);
    sfixn *W_s           = (sfixn *)RTableDataBlock(kv, args[7]);
    sfixn *NoNodes_s     = (sfixn *)RTableDataBlock(kv, args[8]);
    sfixn *Bases_s       = (sfixn *)RTableDataBlock(kv, args[9]);
    sfixn *data_s        = (sfixn *)RTableDataBlock(kv, args[10]);
    sfixn w              = ToSignedFixedNum(kv, args[11]);
    sfixn subslicesz     = ToSignedFixedNum(kv, args[12]);
    sfixn *subslicedims  = (sfixn *)RTableDataBlock(kv, args[13]);
    sfixn Ssz            = ToSignedFixedNum(kv, args[14]);
    sfixn *S             = (sfixn *)MapleToPointer(kv, args[15]);
 /* sfixn dVsz           = ToSignedFixedNum(kv, args[16]); */
    sfixn *pdegVec       = (sfixn *)RTableDataBlock(kv, args[17]);
 /* sfixn cVsz           = ToSignedFixedNum(kv, args[18]); */
    sfixn *coefVec       = (sfixn *)RTableDataBlock(kv, args[19]);

    sfixn N = Nmp[0], m = Nmp[1], p = Nmp[2];
    sfixn pts_sSz = ptPHWDSZ[0], h_sSz = ptPHWDSZ[1];
    sfixn WNB_sSz = ptPHWDSZ[2], data_sSz = ptPHWDSZ[3];
    MONTP_OPT2_AS_GENE prime;
    PTS_TREE *pts_tree;
    preFFTRep *poly;
    int nexti, *nextiAddr = &nexti;

    #if DEBUG 
    if( ! IsMaplePointer(kv, args[15]) ||
        MaplePointerType(kv, args[15]) != (M_INT)&MarkArray )
    {
        MapleRaiseError(kv, "gc array expected");
        return ToMapleNULL(kv);
    }
    #endif 
    EX_MontP_Init_OPT2_AS_GENE(&prime, p);  
    pts_tree = createWrapperPTS_TREE(N, m, bounds, pts_sSz, pts_s, h_sSz, h_s,
        WNB_sSz, W_s, NoNodes_s, Bases_s, data_sSz, data_s, p);

    poly = interpNextCandidateSliceLCDefective(nextiAddr, start, N, m, w, subslicesz,
        subslicedims, Ssz, S, pts_tree, &prime);

    if (poly==NULL) return ToMapleInteger(kv, (long)(-1));
    create_pdeg_coef_Vec(pdegVec, coefVec, poly);
    freePoly(poly);
    my_free(poly);
    freeWrapperPTS_TREE(pts_tree);

    return ToMapleInteger(kv, (long)nexti);
}
#endif // _mcompile_

/**
 * CUDA related functions
 */

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
is_cuda_tag_enabled() { return CUDA_TAG; }

#ifdef WINDOWS
__declspec(dllexport) sfixn __stdcall
#else
sfixn
#endif
enable_cuda_tag(sfixn tag) { 
    sfixn oldtag = CUDA_TAG; 
    CUDA_TAG = tag;
    return oldtag;
}

/********************************* EOF ****************************************/

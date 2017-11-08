/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#include "MPMMTS.h"

extern int Interrupted;

sfixn MulCut=42;
sfixn DivCut1=146;
sfixn DivCut2=5;
sfixn DivCut3=3;
sfixn DivCutN=2;
sfixn UniGcdCut=500;
extern sfixn topN;
extern double timeVec[];
extern double timeVec2[];
extern double top2MulTime1;
extern double top2MulTime2;

extern double time_cls, time_fft, time_mulmod;

struct exception_context the_exception_context[1];

// The declaration of the four functions below
// MUST remain here, for compilaion issues.

//  Fast division in PF[x], destructive to APtr.
static void UniFastMod_1(sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, 
        sfixn Lbound, sfixn * BRevInvPtr,  MONTP_OPT2_AS_GENE * pPtr);

// Fast division in PF[x_1, ... , x_n][y], destructive to inPtr.
static void MultiUniFastMod_1(sfixn N,  preFFTRep * tmpIn,  sfixn deg, 
        preFFTRep * inPtr, TriSet * ts, TriRevInvSet * tris,  
        MONTP_OPT2_AS_GENE * pPtr);

// Plain division in PF[x_1, ... , x_n][y]/<TS>, destructive to FPtr.
static preFFTRep * MultiUniPlainMod_1(sfixn N, sfixn d1, preFFTRep * FPtr, 
        sfixn d2, preFFTRep * GPtr, TriSet * ts, TriRevInvSet * tris, 
        MONTP_OPT2_AS_GENE * pPtr);


void NewtonRevInverse(sfixn N, preFFTRep * tRIPtrtmp, preFFTRep * tRIPtr, 
        preFFTRep * tPtr, TriSet * ts, TriRevInvSet * tris,
        MONTP_OPT2_AS_GENE * pPtr);

/**
 * getDenseSiz:
 * @N: number of variables.
 * @poly: A C-Cube polynomial.
 * @buszsN: BUSZSI('poly', 'N').
 * @data: The data field of 'poly'.
 * @cumN: CUMI('poly', 'N').
 *
 * Data size of the C-Cube polynomial 'poly' minus all leading zeros in it.
 * 
 * Return value: SIZ(poly) - number of all leading zeros in the least variable. 
 **/
sfixn getDenseSiz(sfixn N, preFFTRep *poly, sfixn buszsN, sfixn *data, sfixn cumN)
{
    sfixn deg, siz=0;  
    deg = shrinkDeg(buszsN, data, cumN);       
    if(N==1) return deg +1;
    if(deg>0) return (deg+1)*CUMI(poly, N);
    if(deg ==0){
        siz=getDenseSiz(N-1, poly, BUSZSI(poly, N-1), data, CUMI(poly, N-1));
        return siz; 
    }
    return -1;
}

// test if all coefficients are just numbers.
// return 1, TRUE.
// return 0, FALSE
int IsAllNumberCoeffs(preFFTRep *poly){
    int res=1, i;
    sfixn N=N(poly), d;
    backupData(poly);
    d=shrinkDeg(BUSZSI(poly, N), DAT(poly), CUMI(poly, N));
    decreaseOneDim(poly); 
    for(i=0; i<=d; i++){
        if(! constantPolyp(poly)) res=0;    
        nextCoefData(poly, N);
    }
    increaseOneDim(poly);
    restoreData(poly);
    return res;
}

// all coefficients are constant polynomials.
// And both input polynomials have the same allocated size.
preFFTRep * QuoAsUni(preFFTRep *f1, preFFTRep *f2, MONTP_OPT2_AS_GENE * pPtr){
    sfixn d1, d2, N=N(f1);
    sfixn *vec1, *vec2, *dat1, *dat2, *Q, degQ;
    int i, offset;
    preFFTRep *resPoly;
    d1=shrinkDeg(BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    d2=shrinkDeg(BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
    vec1=(sfixn *)my_calloc(d1+1, sizeof(sfixn));
    vec2=(sfixn *)my_calloc(d2+1, sizeof(sfixn));
    for(i=0, offset=0; i<=d1; i++){
        vec1[i] = (DAT(f1))[offset];
        offset+=CUMI(f1, N);
    }
    for(i=0, offset=0; i<=d2; i++){
        vec2[i] = (DAT(f2))[offset];
        offset+=CUMI(f2, N);
    }
    Q = EX_UniQuo(&degQ, d1, vec1, d2, vec2, pPtr);

    resPoly = EX_InitOnePoly(N(f1), BUSZS(f1));

    for(i=0, offset=0; i<=degQ; i++){
        (DAT(resPoly))[offset]=Q[i];
        offset+=CUMI(resPoly, N);
    }

    my_free(Q);
    return resPoly;
}




// all coefficients are constant polynomials.
// And both input polynomials have the same allocated size.
preFFTRep * GcdAsUni(preFFTRep *f1, preFFTRep *f2, MONTP_OPT2_AS_GENE * pPtr){
    sfixn d1, d2, N=N(f1);
    sfixn *vec1, *vec2, *dat1, *dat2, *G, degG;
    int i, offset;
    preFFTRep *resPoly;
    d1=shrinkDeg(BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    d2=shrinkDeg(BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
    vec1=(sfixn *)my_calloc(d1+1, sizeof(sfixn));
    vec2=(sfixn *)my_calloc(d2+1, sizeof(sfixn));
    for(i=0, offset=0; i<=d1; i++){
        vec1[i] = (DAT(f1))[offset];
        offset+=CUMI(f1, N);
    }
    for(i=0, offset=0; i<=d2; i++){
        vec2[i] = (DAT(f2))[offset];
        offset+=CUMI(f2, N);
    }

    G = EX_GCD_UNI(&degG, vec1, d1, vec2, d2, pPtr);

    resPoly = EX_InitOnePoly(N(f1), BUSZS(f1));

    for(i=0, offset=0; i<=degG; i++){
        (DAT(resPoly))[offset]=G[i];
        offset+=CUMI(resPoly, N);
    }

    my_free(G);
    return resPoly;
}




/**
 * copyPolyPointers:
 * @D: Destination C-cube polynomial.
 * @S: Source C-cube polynomial.
 * 
 * Assume 'D' has type of preFFTRep *, but its sub-fields 
 * have not been initialized.  
 * This function copies S's fields to 'D'.
 *
 * Return value: 
 **/
    void 
copyPolyPointers(preFFTRep *D, preFFTRep *S)
{
    D->N=S->N; 
    D->defN=S->defN;
    D->bufSizs=S->bufSizs;
    D->cuts=S->cuts; 
    D->accum=S->accum;
    D->size=S->size;  
    D->defSize=S->defSize;
    D->offset=S->offset;
    D->data=S->data;
    D->tmpData=S->tmpData; 
    D->defData=S->defData; 
}

//===================================================
// To check the degrees in a given triangular set.
// -1 means bounds of TS is NOT ok.
//  0 means ok.
//===================================================
static int
checkDgsOfST(sfixn N, TriSet * ts){
    register int i;
    for(i=1; i<=N; i++) if(BDSI(ts, i)<0) return -1;
    return 0;
}

//============================================
// Univariate Plain division.
// type: in-place.
// output: the remainder.
//============================================

/**
 * UniPlainMod_1:
 * @degA: The degree of the univeriate polynomial 'A'.
 * @APtr: The coefficient vector of univeriate polynomial 'A'.
 * @degB: The degree of the univeriate polynomial 'B'.
 * @BPtr: The coefficient vector of univeriate polynomial 'B'.
 * @pPtr: The information of the prime.
 * 
 * Compute the remainder R, where A = QB+R mod p.
 * The content of input 'A' will be replaced by 'R'. 
 * Return value: 
 **/
static void UniPlainMod_1(sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, 
        MONTP_OPT2_AS_GENE * pPtr )
{
    register sfixn i,j,p=pPtr->P;
    sfixn tmp;
    if((degA-degB)<0) return;
    if(degB>10){
        for(i=degA-degB; i>=0; i--){
            tmp = MontMulMod_OPT2_AS_GENE(APtr[degB+i], pPtr->R2BRsft, pPtr)<<(pPtr->Base_Rpow);
            for(j=0; j<degB; j++) 
                APtr[i+j]=SubMod(APtr[i+j], MontMulMod_OPT2_AS_GENE(BPtr[j],tmp,pPtr), p);
            APtr[i+degB]=0;
        }
    } else{
        for(i=degA-degB; i>=0; i--){
            for(j=0; j<degB; j++) 
                APtr[i+j]=SubMod(APtr[i+j], MulMod(BPtr[j],APtr[degB+i],p), p);
            APtr[i+degB]=0;
        }
    }
}

// return 1 if the coef is zero.
/**
 * isOnep:
 * @data: A sfixn vector.
 * @size: The size of the data vector 'data'.
 * 
 *  To test if the data[0]=1, and data[i]=0, for 1<i<size.
 * In other words, it tests whether a dense univariate polynomial is 1.
 *
 * Return value: 1 for true. 0 for false.
 **/
int isOnep(sfixn *data, sfixn size){
    register int i;
    for(i=size-1;i>1;i--)
        if (data[i]) return 0;
    if(data[0]!=1) return 0;
    return 1;
}


/**
 * onePolyp:
 * @polyPtr: A C-CUBE polynomial.
 * 
 * To test if 'polyPtr' is a C-Cube polynomial equal to one. 
 *
 * Return value: 1 for true, 0 for false.
 **/
sfixn
onePolyp(preFFTRep * polyPtr){
    register int i;
    if((!(N(polyPtr))) && ((DATI(polyPtr, 0))==1)) return 1; 
    for(i=1;i<(SIZ(polyPtr));i++){ if (DATI(polyPtr, i)) return 0;}
    if(DATI(polyPtr, 0)!=1) return 0;
    return 1;
}


//=============================================
// TO compare the data vectors of p1 and p2.
// according to p1's size
// 1 means equal, 0 means not equal.
//=============================================
int comparePolyData(preFFTRep * p1, preFFTRep * p2){
    return compareVec(SIZ(p1)-1, DAT(p1), DAT(p2));
}

//=================================================================
// To check if the multivariate polynomial is monic in its variable X_N.
// if monic return 0.
// otherwise return 1.
//=================================================================
/**
 * monicPolyp:
 * @N: X_N is the main variable.
 * @f: A C-Cube polynomial.
 *
 * To check if 'f' is monic or not. 
 * By monic, we mean that either the polynomial is the constant 1
 * the polynomial isnon-constant and its initial is one.
 *
 * Return value: 1 for true, 0 for fail. 
 **/
static int monicPolyp(sfixn N, preFFTRep* f){
    sfixn d;
    d=shrinkDeg(BUSZSI(f, N), DAT(f), CUMI(f, N));
    backupData(f);
    decreaseOneDim(f); 
    nextMCoefData(f,N,d);

    if(onePolyp(f)) 
    {  increaseOneDim(f);
        restoreData(f);
        return 1;}
        increaseOneDim(f);
        restoreData(f);
        return 0;
}



/**
 * groundPoly:
 * @polyPtr: A C-Cube polynomial.
 * 
 * If 'polyPtr' is not a constant polynomial, returns -1, otherwise
 * strip this constant polynomial 'polyPtr' into an actual constant.
 *
 * Return value: return the constant of the constant polynomial.
 *               or -1 if the operaton failed. 
 **/
static sfixn groundPoly(preFFTRep * polyPtr){
    register int i,j;
    if(!(N(polyPtr))) return DATI(polyPtr, 0);  
    for(i=(SIZ(polyPtr))-1, j=i; i>=0; i--, --j){ if (DATI(polyPtr, i)) break;}
    if(j>0) return -1;
    return DATI(polyPtr, 0);  
}



/**
 * getCoefMulti:
 * @N: Number of variables.
 * @co: (output) the coefficient of 'f' in degree `i` w.r.t. `X_N`.
 * @f: A C-Cube polynomial in `N` variables
 * @i: An index.
 * Make a copy of the i-th coefficient of 'f' and save the copy in co.
 * Return value: The copy of i-th coefficient of 'f'. 
 **/
static preFFTRep *
getCoefMulti(sfixn N, preFFTRep* co, preFFTRep* f, sfixn i){
    decreaseOneDim(f);
    backupData(f); 
    nextMCoefData(f,N,i);
    fromtofftRep(N-1,  CUM(co), DAT(co), CUM(f), BUSZS(f), DAT(f));
    increaseOneDim(f);
    restoreData(f);
    return co;
}

/**
 * CopyOnePoly:
 * @rPtr: Destination polynomial.
 * @fPtr: Source polynomial.
 * 
 * Make a copy of 'fPtr' and save the copy in 'rPtr'.
 *
 * Return value: 'rPtr'.
 **/
void CopyOnePoly(preFFTRep * rPtr, preFFTRep * fPtr ){  
    register int j;
    sfixn N=N(rPtr);
    N(rPtr)=N;
    SIZ(rPtr)=1;
    for(j=1;j<=N;j++){
        BUSZSI(rPtr, j)=BUSZSI(fPtr, j);    
        CUMI(rPtr, j)=CUMI(fPtr, j);
        SIZ(rPtr)=SIZ(rPtr)*(BUSZSI(rPtr, j)+1);
    } 
    OFST(rPtr)=0;
    for(j=0; j<SIZ(rPtr);j++) DATI(rPtr, j)=DATI(fPtr, j);
    //BUSZSI(rPtr, 0)=BUSZSI(rPtr, N);
    DFN(rPtr)=N(rPtr);
    DFSIZ(rPtr)=SIZ(rPtr);
    DEFDAT(rPtr)=DAT(rPtr);
}

/**
 * EX_CopyOnePoly:
 * @fPtr: Source polynomial.
 * 
 * Make a deep copy of 'fPtr'.
 *
 * Return value: The copy of 'fPtr'.
 **/
preFFTRep *
EX_CopyOnePoly(preFFTRep * fPtr ){
    preFFTRep *rPtr;
    rPtr=(preFFTRep *)my_malloc(sizeof(preFFTRep));
    InitOnePoly(rPtr, N(fPtr), BUSZS(fPtr));
    CopyOnePoly(rPtr, fPtr);
    return rPtr;
}

/**
 * EX_CopyOnePolyList:
 * @no: number of polynomials in the list.
 * @PolyList: A list of polynomials.
 *
 * Make a deep copy of 'PolyList'.
 *
 * Return value: The copy of 'PolyList'.
 **/
preFFTRep ** EX_CopyOnePolyList(sfixn no, preFFTRep **PolyList ){
    int i;
    preFFTRep **CopyPolyList = (preFFTRep **) my_malloc(no * sizeof(preFFTRep *));
    for(i=0; i<no; i++){
        CopyPolyList[i] = EX_CopyOnePoly(PolyList[i]);
    }
    return CopyPolyList;
}

void CopyOnePolyOffset(preFFTRep * rPtr, preFFTRep * fPtr, sfixn m ){  
    register int j;
    sfixn N=N(rPtr), offset;
    N(rPtr)=N;
    SIZ(rPtr)=1;
    for(j=1;j<=N;j++){
        BUSZSI(rPtr, j)=BUSZSI(fPtr, j);
        if(j==N) BUSZSI(rPtr, j) += m;     
        CUMI(rPtr, j)=CUMI(fPtr, j);
        SIZ(rPtr)=SIZ(rPtr)*(BUSZSI(rPtr, j)+1);
    } 
    OFST(rPtr)=0;
    offset = CUMI(fPtr,N) * m;
    if (offset > 0){
        for(j=0; j<SIZ(fPtr);j++) DATI(rPtr, j+offset )=DATI(fPtr, j);
    }
    else{
        for(j=-offset; j<(SIZ(fPtr));j++) DATI(rPtr, j+offset )=DATI(fPtr, j);
    }

    DFN(rPtr)=N(rPtr);
    DFSIZ(rPtr)=SIZ(rPtr);
    DEFDAT(rPtr)=DAT(rPtr);
}

/**
 * direvativeMulti:
 * @poly: A C-Cube polynomial.
 * @pPtr: Infomration for the prime.
 * 
 * Compute the derivative of 'poly' in x_1, 
 * where x_1 is the least variable of 'poly'.
 *   
 * Return value: 
 **/
preFFTRep * direvativeMulti(preFFTRep * poly, MONTP_OPT2_AS_GENE * pPtr){
    preFFTRep * rPoly;
    sfixn i, offset=0, N=N(poly);
    rPoly = EX_ShiftPoly(poly, -1);

    for(i=1; i<=BUSZSI(rPoly, N); i++){
        offset+=CUMI(rPoly, N); 
        coMulVec_1(i+1, CUMI(rPoly, N)-1, DAT(rPoly)+offset, pPtr);
    }
    return rPoly;
}

/**
 * ShiftPoly:
 * @fPtr: A C-Cube polynomial.
 * @m: a sfixn.
 * 
 * "Shift" 'fPtr' to right by m.
 * Namely, compute (X_N)^m * fPtr, 
 *     where X_N is the main variable of 'fPtr'. 
 * X_1 < X_2 < ... < X_N.
 *  
 *  
 * Return value: (X_N)^m * fPtr.
 **/
preFFTRep * EX_ShiftPoly(preFFTRep * fPtr, int m){
    preFFTRep *rPtr;
    sfixn N=N(fPtr), tmp;
    if(m==0) { return EX_CopyOnePoly(fPtr );}
    if (((BUSZSI(fPtr, N)+1) +  m) < 0) {
        printf("shifting out of range!\n");
        fflush(stdout);
        Throw 1111;
    }
    tmp=BUSZSI(fPtr,N);
    BUSZSI(fPtr,N)+=m;
    rPtr=(preFFTRep *)my_malloc(sizeof(preFFTRep));
    InitOnePoly(rPtr, N, BUSZS(fPtr));
    BUSZSI(fPtr,N)=tmp;
    CopyOnePolyOffset(rPtr, fPtr, m);
    return rPtr;
}

/**
 * EX_FreeOnePoly:
 * @rPtr: A C-Cube polynomial.
 * de-allocate 'rPtr'.
 * Return value:  
 **/
void EX_FreeOnePoly(preFFTRep * rPtr){
    if(rPtr){
        freePoly(rPtr);
        my_free(rPtr);
    }
}

sfixn *
bounds2dgs(TriSet *ts){
    int i;
    sfixn N=N(ts);
    sfixn *dgs;
    dgs=(sfixn *)my_calloc(N, sizeof(sfixn));
    for(i=0; i<N; i++){
        dgs[i]=BDSI(ts, i+1)+1;
    }
    return dgs;
}

// only top-level is different.
static void CopyOnePoly_deg(preFFTRep * rPtr, preFFTRep * fPtr, sfixn dg){  
    register int i;
    sfixn N=N(rPtr);
    for(i=0;i<DFSIZ(rPtr);i++) DATI(rPtr, i)=0;
    BUSZSI(rPtr, N)=dg;
    SIZ(rPtr)=CUMI(rPtr, N)*(dg+1);
    OFST(rPtr)=0;
    for(i=0; i<SIZ(rPtr);i++) DATI(rPtr, i)=DATI(fPtr, i);
}

/**
 * printPolyStru:
 * Ptr@: a C-Cube polynomial
 * 
 * Prints the entire C-Cube polynomial data structure
 * 
 * Return value: 
 **/   
void printPolyStrut(preFFTRep * Ptr){
    printf("\n{\n   N=%ld\n",N(Ptr)); 
    printf("   defN=%ld\n", DFN(Ptr));
    printf("   buff=");
    printVec(N(Ptr), BUSZS(Ptr));
    printf("   dgs=");
    printVec(N(Ptr), CUTS(Ptr));
    printf("   accum=");
    printVec(N(Ptr), CUM(Ptr)); 
    printf("   size=%ld\n",SIZ(Ptr));
    printf("   defSize=%ld\n",DFSIZ(Ptr));
    printf("   offset=%ld\n",OFST(Ptr));
    printf("   data=");
    if (N(Ptr)==0)  printf("%ld\n", DATI(Ptr, 0));   else printPoly(Ptr);
    printf("\n");
    printf("}\n");
}

// inputPolydegs is dgs.
// useful data is dgs[1..N].
/**
 * getRevInvTiSet:
 * @dgs: The partial degrees of the triangular set.
 * @N: The number of variables.
 * @tris: (output) the inverses (reverse ordered) of ts.
 * @ts: The triangular set.
 * @pPtr: The information of the prime.
 *
 * This is used for fast division (or normal form) modulo a triangular set.
 * We pre-compute with this functionn the inverse of the polynomial `ts.i`
 * w.r.t. 'Xi' modulo the underlying triangular set.
 * See Xin Li, Marc Moreno Maza, Eric Schost (ISSAC 2007).  
 *
 * Compute the inverses (reverse ordered) of given triangular set 'ts'.
 *
 * Return value: 
 **/
void getRevInvTiSet(sfixn *dgs, sfixn N, TriRevInvSet * tris, TriSet * ts,  
        MONTP_OPT2_AS_GENE * pPtr)
{
    register int i;
    preFFTRep tRIPtrtmp;
    for(i=1;i<=N;i++){
        InitOneRevInv(i, &tRIPtrtmp, BDS(ts), dgs[i]);
        if(!EXSTI(tris, i))
            NewtonRevInverse(i, &tRIPtrtmp, ELEMI(tris, i), ELEMI(ts, i), ts, tris, pPtr );
        freePoly(&tRIPtrtmp);}
}


// Subroutine of the next next next function
void InitOneMonicRandomPolys(sfixn * bounds, sfixn N, preFFTRep * p1Ptr,  
        MONTP_OPT2_AS_GENE * pPtr, sfixn seed)
{
    register int j;
    N(p1Ptr)=N;
    BUSZS(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUTS(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUM(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn)); 
    CUMI(p1Ptr, 1)= 1;
    SIZ(p1Ptr)=1;
    OFST(p1Ptr)=0; 
    //srand(getSeed());
    for(j=1;j<=N;j++){
        BUSZSI(p1Ptr, j)=bounds[j];
        CUTSI(p1Ptr, j)=bounds[j];
        if(j==N) { BUSZSI(p1Ptr, j)=bounds[j]+1;
            CUTSI(p1Ptr, j)=bounds[j]+1;
        }
        SIZ(p1Ptr)*=BUSZSI(p1Ptr, j)+1;
        if(j>=2){
            CUMI(p1Ptr, j)=CUMI(p1Ptr, j-1)*(BUSZSI(p1Ptr, j-1)+1);
        }
    }
    DAT(p1Ptr)=(sfixn * )my_calloc(SIZ(p1Ptr),sizeof(sfixn));
    randomMonicPoly(p1Ptr, pPtr->P);
    DFN(p1Ptr)=N(p1Ptr);
    DFSIZ(p1Ptr)=SIZ(p1Ptr);
    DEFDAT(p1Ptr)=DAT(p1Ptr);
    //BUSZSI(p1Ptr, 0)=BUSZSI(p1Ptr, N);
}

// Generates a polynomial whose coefficients are all one, for debugging
void InitOneMonicRandomPolys_allOne(sfixn * bounds, sfixn N, preFFTRep * p1Ptr,
        MONTP_OPT2_AS_GENE * pPtr, sfixn seed)
{
    register int j;
    N(p1Ptr)=N;
    BUSZS(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUTS(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUM(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn)); 
    CUMI(p1Ptr, 1)= 1;
    SIZ(p1Ptr)=1;
    OFST(p1Ptr)=0; 
    //srand(getSeed());
    for(j=1;j<=N;j++){
        BUSZSI(p1Ptr, j)=bounds[j];
        CUTSI(p1Ptr, j)=bounds[j];
        if(j==N) { BUSZSI(p1Ptr, j)=bounds[j]+1;  CUTSI(p1Ptr, j)=bounds[j]+1;}
        SIZ(p1Ptr)*=BUSZSI(p1Ptr, j)+1;
        if(j>=2){
            CUMI(p1Ptr, j)=CUMI(p1Ptr, j-1)*( BUSZSI(p1Ptr, j-1)+1);
        }
    }
    DAT(p1Ptr)=(sfixn * )my_calloc(SIZ(p1Ptr),sizeof(sfixn));
    randomMonicPoly_allOne(p1Ptr, pPtr->P);
    DFN(p1Ptr)=N(p1Ptr);
    DFSIZ(p1Ptr)=SIZ(p1Ptr);
    DEFDAT(p1Ptr)=DAT(p1Ptr);
    //BUSZSI(p1Ptr, 0)=BUSZSI(p1Ptr, N);
}

// Subroutine of the next function
void initRandomTriSet( sfixn N, sfixn dgbound, TriSet * tPtr,
        MONTP_OPT2_AS_GENE * pPtr)
{
    int i;

    N(tPtr)=N;
    NMLZ(tPtr)=0;
    ELEM(tPtr)=(preFFTRep **)my_calloc((N+1),sizeof(preFFTRep *) );
    BDS(tPtr)=(sfixn *) my_calloc((N+1),sizeof(sfixn));
    //srand(getSeed());
    for(i=1;i<=N;i++) {while(! BDSI(tPtr,i) ) BDSI(tPtr, i)=rand()%dgbound;}
    if(checkDgsOfST(N, tPtr)==-1) Throw 0;
    for(i=1; i<=N; i++){
        ELEMI(tPtr, i)=(preFFTRep *)my_calloc(1,sizeof(preFFTRep));
        InitOneMonicRandomPolys(BDS(tPtr), i, ELEMI(tPtr, i), pPtr, i);
    }
}




/**
 * EX_initRandomTriSet:
 * @N: Number of variables.
 * @dgbound: A sfixn which is a strict upper bound for all partial
 *           degrees in the output triangular set
 * @pPtr: Information of the prime number.
 * Create an randome Lazard trianuglar set in dimension zero
 *       with `N` variables and partial degrees strictly bounded by `degbound`
 * Return value:  an randome Lazard trianuglar set in dimension 
 **/
TriSet *
EX_initRandomTriSet( sfixn N, sfixn dgbound, MONTP_OPT2_AS_GENE * pPtr){
    TriSet *tPtr;
    tPtr=(TriSet *)my_calloc(1,sizeof(TriSet));
    initRandomTriSet(N, dgbound, tPtr, pPtr);
    return tPtr;
}

// Depreciated code
void NormalizeTriSetBDS(sfixn N, TriSet * ts){
    int i;
    for(i=1; i<=N; i++){
        BDSI(ts,i)=BUSZSI(ELEMI(ts, i), i) - 1 ;
    }
}

// Subroutine of the next function
void initRandomNonMonicTriSet( sfixn N, sfixn dgbound, TriSet * tPtr,  
        MONTP_OPT2_AS_GENE * pPtr)
{
    int i;
    N(tPtr)=N;
    NMLZ(tPtr)=0;
    ELEM(tPtr)=(preFFTRep **)my_calloc((N+1),sizeof(preFFTRep *) );
    BDS(tPtr)=(sfixn *) my_calloc((N+1),sizeof(sfixn));
    //srand(getSeed());
    for(i=1;i<=N;i++) {while(! BDSI(tPtr,i) ) BDSI(tPtr, i)=rand()%dgbound;}
    if(checkDgsOfST(N, tPtr)==-1) Throw 0;
    for(i=1; i<=N; i++){
        BDSI(tPtr, i)++;
        ELEMI(tPtr, i)=EX_randomPoly(i, BDS(tPtr), pPtr->P);
        BDSI(tPtr, i)--;
    }
}

/**
 * EX_initRandomNonMonicTriSet:
 * @N: The number of variables.
 * @dgbound: The bound of degrees of the triangular set.
 * @tPtr: (output) the triangular set.
 * @pPtr: The information of the prime.
 * 
 * To create a random non-monic, zero-dimensional triangular set.
 * In particular, initials can be non-constant polynomials
 * Return value: The newly created random triangular set. 
 **/
TriSet * EX_initRandomNonMonicTriSet(sfixn N, sfixn dgbound, 
        MONTP_OPT2_AS_GENE * pPtr)
{
    TriSet *tPtr=(TriSet *)my_malloc(sizeof(TriSet));
    //printf("Entering initRandomNonMonicTriSet...\n");
    //fflush(stdout);
    initRandomNonMonicTriSet( N, dgbound, tPtr, pPtr);
    //printf("Leaveing initRandomNonMonicTriSet...\n");
    //fflush(stdout);
    return tPtr;
}


/**
 * initRandomDenseTriSet:
 * @N: The number of variables.
 * @dgs: the partial degrees in the output triangular set
 * @tPtr: (output) the triangular set.
 * @pPtr: The information of the prime.
 * 
 * To create a random monic, zero-dimensional triangular set,
 * with prescribed partial degrees.
 * Return value: The newly created random triangular set. 
 **/
void initRandomDenseTriSet( sfixn N, sfixn * dgs, TriSet * tPtr,
        MONTP_OPT2_AS_GENE * pPtr)
{
    int i;
    N(tPtr)=N;
    ELEM(tPtr)=(preFFTRep **)my_calloc((N+1),sizeof(preFFTRep *) );
    BDS(tPtr)=(sfixn *) my_calloc((N+1),sizeof(sfixn));
    //srand(getSeed());
    for(i=1;i<=N;i++)  BDSI(tPtr, i)=dgs[i-1]-1;

    if(checkDgsOfST(N, tPtr)==-1) Throw 0;

    for(i=1; i<=N; i++){
        ELEMI(tPtr, i)=(preFFTRep *)my_calloc(1,sizeof(preFFTRep));
        InitOneMonicRandomPolys(BDS(tPtr), i, ELEMI(tPtr, i), pPtr, i);
    }

}

// Subroutine for the next next function
void InitOneMonicPolys(sfixn * bounds, sfixn N, preFFTRep * p1Ptr, 
        MONTP_OPT2_AS_GENE * pPtr, sfixn seed)
{
    register int j;
    N(p1Ptr)=N;
    BUSZS(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUTS(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUM(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn)); 
    CUMI(p1Ptr, 1)= 1;
    SIZ(p1Ptr)=1;
    OFST(p1Ptr)=0; 
    //srand(getSeed());
    for(j=1;j<=N;j++){
        BUSZSI(p1Ptr, j)=bounds[j];
        CUTSI(p1Ptr, j)=bounds[j];
        if(j==N) { BUSZSI(p1Ptr, j)=bounds[j]+1;
            CUTSI(p1Ptr, j)=bounds[j]+1;
        }
        SIZ(p1Ptr)*=BUSZSI(p1Ptr, j)+1;
        if(j>=2){
            CUMI(p1Ptr, j)=CUMI(p1Ptr, j-1)*(BUSZSI(p1Ptr, j-1)+1);
        }
    }
    DAT(p1Ptr)=(sfixn * )my_calloc(SIZ(p1Ptr),sizeof(sfixn));
    DFN(p1Ptr)=N(p1Ptr);
    DFSIZ(p1Ptr)=SIZ(p1Ptr);
    DEFDAT(p1Ptr)=DAT(p1Ptr);
    //BUSZSI(p1Ptr, 0)=BUSZSI(p1Ptr, N);
}



// Subroutine for the next function
void initDenseTriSet( sfixn N, sfixn * dgs, TriSet * tPtr,  
        MONTP_OPT2_AS_GENE * pPtr)
{
    int i;

    N(tPtr)=N;
    NMLZ(tPtr)=0;
    ELEM(tPtr)=(preFFTRep **)my_calloc((N+1),sizeof(preFFTRep *) );
    BDS(tPtr)=(sfixn *) my_calloc((N+1),sizeof(sfixn));
    //srand(getSeed());
    for(i=1;i<=N;i++)  BDSI(tPtr, i)=dgs[i-1]-1;

    if(checkDgsOfST(N, tPtr)==-1) Throw 0;

    for(i=1; i<=N; i++){
        ELEMI(tPtr, i)=(preFFTRep *)my_calloc(1,sizeof(preFFTRep));
        InitOneMonicPolys(BDS(tPtr), i, ELEMI(tPtr, i), pPtr, i);
    }
}

/**
 * Ex_InitDenseTriSet:
 * @N: The number of variables.
 * @dgs: The degrees of the triangular set.
 * @pPtr: The information of the prime.
 * 
 * Return value: A buffer (or template) for a triangular set
 * with the specified partial degrees. All coefficients are
 * set to zero. So, mathematically, this is not *yet*
 * a triangular set.
 **/
TriSet * Ex_InitDenseTriSet( sfixn N, sfixn * dgs, MONTP_OPT2_AS_GENE * pPtr){
    TriSet *ts=(TriSet *)my_malloc(sizeof(TriSet)); 
    initDenseTriSet(N, dgs, ts, pPtr);
    return ts;
}

/**
 * EX_CopyOneTriSet:
 * @srcts: A one-dimensional triangular set.
 * 
 * Make a deep copy of the input triangular set 'srcts'.
 *
 * Return value: A deep copy of 'srcts'.
 **/
TriSet *
EX_CopyOneTriSet(TriSet * srcts){
    int i;
    TriSet *dests=(TriSet *)my_malloc(sizeof(TriSet));
    N(dests)=N(srcts);
    NMLZ(dests)=NMLZ(srcts);
    BDS(dests)=(sfixn *) my_calloc((N(dests)+1),sizeof(sfixn));
    for(i=1;i<=N(dests);i++)  BDSI(dests, i)=BDSI(srcts, i);
    ELEM(dests)=(preFFTRep **)my_calloc((N(dests)+1),sizeof(preFFTRep *) );
    for(i=1; i<=N(dests); i++){
        ELEMI(dests,i)=  EX_CopyOnePoly( ELEMI(srcts, i));
    }  
    return dests;
}

/**
 * EX_getLowerTriSet:
 * @M: An index.
 * @srcts: A triangular set.
 * Make a deep copy of the lower-part of the input triangular set 'srcts'.
 * Namely, if srcts is [T_1, T_2, ..., T_N].
 * This routine creates another triangular set [T_1, T_2,..., T_M].
 *
 * Of course we assume that M is less or equal to N (the number
 * of variables in srcts).
 *
 * Return value: A copy of the lower part of 'srcts'.
 **/
TriSet *
EX_getLowerTriSet(sfixn M, TriSet * srcts){
    int i;
    TriSet *dests=(TriSet *) my_malloc(sizeof(TriSet));
    assert(M<=N(srcts));
    N(dests)=M;
    NMLZ(dests)=NMLZ(srcts);
    BDS(dests)=(sfixn *) my_calloc((N(dests)+1),sizeof(sfixn));
    for(i=1;i<=N(dests);i++)  BDSI(dests, i)=BDSI(srcts, i);
    ELEM(dests)=(preFFTRep **)my_calloc((N(dests)+1),sizeof(preFFTRep *) );
    for(i=1; i<=N(dests); i++){
        ELEMI(dests,i) = EX_CopyOnePoly( ELEMI(srcts, i));
    }  
    return dests;
}

// Generates a particular example for testing the lifting
// y0=10;
void init_example_1_DenseTriSetForLifting_y0_10(sfixn N, sfixn * dgs, 
        TriSet * tPtr,  MONTP_OPT2_AS_GENE * pPtr)
{
    // Make this a parameter later.
    int i;
    N(tPtr)=N;
    NMLZ(tPtr)=0;
    ELEM(tPtr)=(preFFTRep **)my_calloc((N+1),sizeof(preFFTRep *) );
    BDS(tPtr)=(sfixn *) my_calloc((N+1),sizeof(sfixn));
    //srand(getSeed());
    for(i=1;i<=N;i++)  BDSI(tPtr, i)=dgs[i-1]-1;

    if(checkDgsOfST(N, tPtr)==-1) Throw 0;

    for(i=1; i<=N; i++){
        ELEMI(tPtr, i)=(preFFTRep *)my_calloc(1,sizeof(preFFTRep));
        InitOneMonicRandomPolys(BDS(tPtr), i, ELEMI(tPtr, i), pPtr, i);
    }

    // t = 0;
    // Set the first monomial T_1 in the Triangular Set to be:
    //     t^d+y0, where d=dgs[0] and dgs[0] is the degree of T_1.
    PolyCleanData(ELEMI(tPtr, 1));
    DATI(ELEMI(tPtr, 1), dgs[0]) = 1;

    PolyCleanData(ELEMI(tPtr, 2));
    DATI(ELEMI(tPtr, 2), 0) = 168912189;
    DATI(ELEMI(tPtr, 2), 1) = 108862386;
    DATI(ELEMI(tPtr, 2), 2) = 57295705;
    DATI(ELEMI(tPtr, 2), 3) = 222454163;
    DATI(ELEMI(tPtr, 2), 4) = 1;
    PolyCleanData(ELEMI(tPtr, 3));
    DATI(ELEMI(tPtr, 3), 0) = 440060203;
    DATI(ELEMI(tPtr, 3), 1) = 156933537;
    DATI(ELEMI(tPtr, 3), 2) = 289935951;
    DATI(ELEMI(tPtr, 3), 3) = 16038265;
    DATI(ELEMI(tPtr, 3), 4) = 1;

}

// Subroutine for next function
// Only for poly in Triandular Set.
void InitOneMonicForTriSet(sfixn * bounds, sfixn N, preFFTRep * p1Ptr,  
        MONTP_OPT2_AS_GENE * pPtr)
{
    register int j;
    N(p1Ptr)=N;

    BUSZS(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUTS(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUM(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn)); 
    CUMI(p1Ptr, 1)= 1;
    SIZ(p1Ptr)=1;
    OFST(p1Ptr)=0; 
    //srand(getSeed());
    for(j=1;j<=N;j++){
        BUSZSI(p1Ptr, j)=bounds[j];
        CUTSI(p1Ptr, j)=bounds[j];
        if(j==N) { BUSZSI(p1Ptr, j)=bounds[j]+1;
            CUTSI(p1Ptr, j)=bounds[j]+1;
        }
        SIZ(p1Ptr)*=BUSZSI(p1Ptr, j)+1;
        if(j>=2){
            CUMI(p1Ptr, j)=CUMI(p1Ptr, j-1)*(BUSZSI(p1Ptr, j-1)+1);
        }
    }
    DAT(p1Ptr)=(sfixn * )my_calloc(SIZ(p1Ptr),sizeof(sfixn));
    // randomMonicPoly(p1Ptr, pPtr->P);
    DFN(p1Ptr)=N(p1Ptr);
    DFSIZ(p1Ptr)=SIZ(p1Ptr);
    DEFDAT(p1Ptr)=DAT(p1Ptr);
    //BUSZSI(p1Ptr, 0)=BUSZSI(p1Ptr, N);
}


//================================================================
// see symbolic Newton lift algorithm.
// Subroutine for lifting one polynomial in a triangular set
//================================================================
void LiftTinTriSet( sfixn N, TriSet * tPtr, int ii, MONTP_OPT2_AS_GENE * pPtr){
    int i;
    preFFTRep * tmpPtr;
    sfixn * newbounds=(sfixn *)my_calloc((N+1),sizeof(sfixn));
    sfixn dgs[2]={0,0};
    dgs[1]=1<<(ii+1);
    freePoly(ELEMI(tPtr, 1)); my_free(ELEMI(tPtr, 1));
    ELEMI(tPtr, 1)=(preFFTRep *)my_calloc(1,sizeof(preFFTRep));
    InitOnePoly(ELEMI(tPtr, 1), 1, dgs);
    DATI(ELEMI(tPtr, 1), dgs[1]) = 1;
    //DATI(ELEMI(tPtr, 1), 0) = (pPtr->P)-1;
    newbounds[1]=dgs[1]-1;

    for(i=2; i<=N; i++){
        newbounds[i]=BDSI(tPtr,i);
        tmpPtr=(preFFTRep *)my_calloc(1,sizeof(preFFTRep));
        InitOneMonicForTriSet(newbounds, i, tmpPtr, pPtr);
        fromtofftRep(i, CUM(tmpPtr), DAT(tmpPtr), CUM(ELEMI(tPtr, i)),  
                BUSZS(ELEMI(tPtr, i)), DAT(ELEMI(tPtr, i)));
        //printf("dead-0.2.%d\n", i-2);
        //fflush(stdout);
        freePoly(ELEMI(tPtr, i)); my_free(ELEMI(tPtr, i));
        //printf("dead-0.3.%d\n", i-2);
        //fflush(stdout);
        ELEMI(tPtr, i)=tmpPtr;
    }


    //fflush(stdout);

    BDSI(tPtr,1)= newbounds[1];
    my_free(newbounds);
}

// Debugging purpose
void initRandomTriSet_dgs_allOne( sfixn N, sfixn * dgs, TriSet * tPtr, 
        MONTP_OPT2_AS_GENE * pPtr)
{
    int i;
    N(tPtr)=N;
    NMLZ(tPtr)=0;
    ELEM(tPtr)=(preFFTRep **)my_calloc((N+1),sizeof(preFFTRep *) );
    BDS(tPtr)=(sfixn *) my_calloc((N+1),sizeof(sfixn));
    //srand(getSeed());
    for(i=1;i<=N;i++)  BDSI(tPtr, i)=dgs[i-1]-1;
    if(checkDgsOfST(N, tPtr)==-1) Throw 0;
    for(i=1; i<=N; i++){
        ELEMI(tPtr, i)=(preFFTRep *)my_calloc(1,sizeof(preFFTRep));
        InitOneMonicRandomPolys_allOne(BDS(tPtr), i, ELEMI(tPtr, i), pPtr, i);
    }
}

// Subroutine for the next one
void freeTriSet(TriSet * tPtr){
    register int i;
    if(ELEM(tPtr)!=NULL){
        for(i=1; i<=N(tPtr); i++){
            if(ELEMI(tPtr, i) != NULL){
                freePoly(ELEMI(tPtr, i));
                my_free(ELEMI(tPtr, i));
                ELEMI(tPtr, i)=NULL;
            }
        }
        my_free(ELEM(tPtr));
        ELEM(tPtr)=NULL;
    }
    if(BDS(tPtr) !=NULL) {my_free(BDS(tPtr)); BDS(tPtr)=NULL;}
}

/**
 * EX_freeTriSet:
 * @tPtr: A trinagulsr set.
 * 
 * Deallocate a triangular set.
 *
 * Return value: 
 **/
void EX_freeTriSet(TriSet * tPtr){
    if(tPtr){
        freeTriSet(tPtr);
        my_free(tPtr);
    }
}

/**
 * InitOneRevInv:
 * @N: number of variables for the output polynomial
 * @tRIPtr: (output) buffer for the inverse (in reversed order)  
 *          of the N-th polynomial in a triangular set 
 *          with partial degrees strictly bounded by `bounds`
 * @bounds: vector 
 * @di: the truncation degree for the inverse 
 * 
 *   Initializes a buffer for a modular inverse computation
 *   of a polynomial modulo a triangular set.
 * 
 * Return value: 
 **/   
// degree of RevInv is always power of 2 which is slightly more than enough.
void
InitOneRevInv(sfixn N, preFFTRep * tRIPtr, sfixn * bounds, sfixn di){
    register int j;
    sfixn eidi2;
    N(tRIPtr)=N;
    BUSZS(tRIPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUTS(tRIPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUM(tRIPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn)); 
    CUMI(tRIPtr, 1)=1;
    for(j=1;j<=N;j++){
        BUSZSI(tRIPtr, j)=bounds[j];
        CUTSI(tRIPtr, j)=bounds[j];
        if(j>=2){
            CUMI(tRIPtr, j)=CUMI(tRIPtr, j-1)*(BUSZSI(tRIPtr, j-1)+1);}
    }
    //if (bounds[N]<4) BUSZSI(tRIPtr, N)=(1<<(logceiling(bounds[N])+2))-1;
    //else BUSZSI(tRIPtr, N)=(1<<(logceiling(bounds[N])+1))-1;
    ////BUSZSI(tRIPtr, N)=(1<<((logceiling(bounds[N]+1))+1))-1;

    eidi2 = di ;

    if(eidi2<=0) {printf("the eidi2 is <=0. \n"); fflush(stdout); Throw 121;}


    BUSZSI(tRIPtr, N)=(1<<((logceiling(eidi2))))-1;

    //BUSZSI(tRIPtr, N)=(1<<((logceiling(bounds[N]+1))+1))-1;   

    CUTSI(tRIPtr, N)=BUSZSI(tRIPtr, N);
    SIZ(tRIPtr)=CUMI(tRIPtr, N)*(BUSZSI(tRIPtr, N)+1);
    OFST(tRIPtr)=0;
    DAT(tRIPtr)=(sfixn * )my_calloc(SIZ(tRIPtr),sizeof(sfixn));
    DFN(tRIPtr)=N(tRIPtr);
    DFSIZ(tRIPtr)=SIZ(tRIPtr);
    DEFDAT(tRIPtr)=DAT(tRIPtr);
    //BUSZSI(tRIPtr, 0)=BUSZSI(tRIPtr, N);
}


// Subroutine for next function
void
initCopyOneRevInv(preFFTRep * outPtr, preFFTRep * inPtr){
    register int j;
    N(outPtr)=N(inPtr);
    BUSZS(outPtr)=(sfixn * )my_calloc(N(inPtr)+1,sizeof(sfixn)); 
    CUTS(outPtr)=(sfixn * )my_calloc(N(inPtr)+1,sizeof(sfixn)); 
    CUM(outPtr)=(sfixn * )my_calloc(N(inPtr)+1,sizeof(sfixn)); 
    for(j=1;j<=N(inPtr);j++){ BUSZSI(outPtr, j)=BUSZSI(inPtr, j);
        CUTSI(outPtr, j)=CUTSI(inPtr, j);
        CUMI(outPtr, j)=CUMI(inPtr, j);}
        SIZ(outPtr)=SIZ(inPtr);
        OFST(outPtr)=OFST(inPtr);
        DAT(outPtr)=(sfixn * )my_calloc(SIZ(outPtr),sizeof(sfixn));
        DFN(outPtr)=N(outPtr);
        DFSIZ(outPtr)=SIZ(outPtr);
        DEFDAT(outPtr)=DAT(outPtr);
        //BUSZSI(outPtr, 0)=BUSZSI(outPtr, N(outPtr));
}



/**
 * initTriRevInvSet:
 * @dgs: A sfixn vector which keeps the truncation degrees for the 
 *         modular inverses. Except for normal form computations,
 *        these should be paritial degrees of the input triangualr set 'tPtr' .
 * @N: Number of variables.
 * @tRevInvPtr: (output) The buffers for the inverses (reverse-ordered) 
 *            of the input triangular set 'tPtr'.
 * @tPtr: A triangular set in dimension-zero.
 * 
 * Create the buffers for the inverses (reverse-ordered) of the intput
 *         triangular set 'tPtr'.
 *         See Xi Lin et al. (ISSAC 2007)
 *
 * Return value: 
 **/
void initTriRevInvSet(sfixn *dgs, sfixn N, TriRevInvSet *tRevInvPtr, TriSet * tPtr){
    register int i;
    N(tRevInvPtr)=N;
    EXST(tRevInvPtr)=(sfixn *)my_calloc(N+1,sizeof(sfixn) );
    NLB(tRevInvPtr)=(sfixn *)my_calloc(N+1,sizeof(sfixn) );
    NSB(tRevInvPtr)=(sfixn *)my_calloc(N+1,sizeof(sfixn) );
    ELEM(tRevInvPtr)=(preFFTRep **)my_calloc((N+1),sizeof(preFFTRep *) );
    for(i=1; i<=N; i++){
        ELEMI(tRevInvPtr, i)=(preFFTRep *)my_calloc(1,sizeof(preFFTRep));
        InitOneRevInv(i, ELEMI(tRevInvPtr, i), BDS(tPtr), dgs[i]);    
        (NLB(tRevInvPtr))[i]= BUSZSI(ELEMI(tRevInvPtr, i), i);
        (NSB(tRevInvPtr))[i]=BDSI(tPtr, i)-1;
    }
}



/**
 * initTriRevInvSet:
 * @dgs: A sfixn vector which keeps the truncation degrees for the 
 *        modular inverses. Except for normal form computations,
 *        these should be paritial degrees of the input triangualr set 'tPtr' .
 * @N: Number of variables.
 * @tPtr: A monic triangular set in dimension-zero.
 * 
 * Returns the buffers for the inverses (reverse-ordered) 
 *            of the input triangular set 'tPtr'.
 *         See Xi Lin et al. (ISSAC 2007)
 *
 * Return value: 
 **/
TriRevInvSet *
EX_initTriRevInvSet( sfixn *dgs, sfixn N, TriSet * tPtr){
    TriRevInvSet * tRevInvPtr;
    tRevInvPtr=(TriRevInvSet *) my_malloc(sizeof(TriRevInvSet));
    initTriRevInvSet(dgs, N, tRevInvPtr, tPtr);
    return tRevInvPtr;
}



/**
 * freeTriRevInvSet:
 * @tRIPtr: Inverses (reverse ordered) of a trinagular set.
 *  Free these inverses in 'tRIPtr'.
 * Return value: 
 **/
void freeTriRevInvSet(TriRevInvSet * tRIPtr){
    register int i;
    if(EXST(tRIPtr) !=NULL){
        my_free(EXST(tRIPtr));
        EXST(tRIPtr)=NULL;
    }
    if(NLB(tRIPtr) !=NULL){
        my_free(NLB(tRIPtr));
        NLB(tRIPtr)=NULL;
    }
    if(NSB(tRIPtr) !=NULL){
        my_free(NSB(tRIPtr));
        NSB(tRIPtr)=NULL;
    }
    if(ELEM(tRIPtr)!=NULL){
        for(i=1; i<=tRIPtr->N; i++){
            if(ELEMI(tRIPtr, i) != NULL){
                freePoly(ELEMI(tRIPtr, i));
                my_free(ELEMI(tRIPtr, i));
                ELEMI(tRIPtr, i)=NULL;
            }
        }
        my_free(ELEM(tRIPtr));
        ELEM(tRIPtr)=NULL;
    }
}


/**
 * freeTriRevInvSet:
 * @tRIPtr: Inverses (reverse ordered) of a trinagular set.
 *  Free these inverses in 'tRIPtr' and 'tRIPts' (a C struct) itself.
 * Return value: 
 **/
void EX_freeTriRevInvSet(TriRevInvSet * tRIPtr){
    if(tRIPtr){
        freeTriRevInvSet(tRIPtr);
        my_free(tRIPtr);
    }
}



// Subroutine for the next function
void printPoly_inner(sfixn N, sfixn * dgs, sfixn * accum, sfixn * data){
    int i, offset=0;
    if(N==1){
        for(i=0; i<dgs[N]; i++){
            //if(data[i]!=0){
            printf("%ld",data[i]);
            printf("*%c^%d",letters[N],i);
            printf("+");
            //}

        }

        printf("%ld*%c^%ld ",data[dgs[N]], letters[N], dgs[N]);

        return;

    }
    for(i=0; i<dgs[N]; i++){
        offset=accum[N]*i;
        // if (!zeroCoefp(data+offset, accum[N])){

        printf("(");
        printPoly_inner(N-1, dgs, accum, data+offset);
        printf(")");
        if(i>0) printf("*%c^%d",letters[N],i);
        printf("+");
        //}
    } 
    offset=accum[N]*dgs[N];
    printf("(");
    printPoly_inner(N-1, dgs, accum, data+offset);
    printf(")");
    if (dgs[N]>0) printf("*%c^%ld", letters[N],dgs[N]);
}


/**
 * printPoly:
 * @Ptr: A C-Cube polynomial.
 * Pretty Printer for a polynomial.
 * This is a 1-D representation which can be used as input
 * in Maple or MAGMA 
 * Return value: 
 **/
void printPoly(preFFTRep * Ptr){

    if(Ptr==NULL) {printf("NULL polynomial.\n"); return;}
    if(!zeroPolyp(Ptr)){
        if (N(Ptr)==0)  {printf("%ld\n", DATI(Ptr, 0)); return;}
        printPoly_inner(N(Ptr), BUSZS(Ptr), CUM(Ptr), DAT(Ptr));
        printf("\n");
    }
    else printf("0\n");

}


// Subroutine of next function
void fprintPoly_inner(FILE *file, sfixn N, sfixn * dgs, sfixn * accum, sfixn * data){
    int i, offset=0;

    if(N==1){
        for(i=0; i<dgs[N]; i++){
            if(data[i]!=0){
                fprintf(file, "%ld",data[i]);
                if(i>0) fprintf(file, "*%c^%d",letters[N],i);
                fprintf(file, "+");
            }
        }
        if(data[dgs[N]] !=0) fprintf(file, "%ld",data[dgs[N]]);
        if(dgs[N]>0) fprintf(file,"*%c^%ld",letters[N],dgs[N]);
        return;
    }
    for(i=0; i<dgs[N]; i++){
        offset=accum[N]*i;

        if (!zeroCoefp(data+offset, accum[N])){
            fprintf(file,"(");
            fprintPoly_inner(file, N-1, dgs, accum, data+offset);
            fprintf(file, ")");
            if(i>0) fprintf(file, "*%c^%d",letters[N],i);
            fprintf(file, "+");
        }
    } 
    offset=accum[N]*dgs[N];
    fprintf(file, "(");
    fprintPoly_inner(file, N-1, dgs, accum, data+offset);
    fprintf(file, ")");
    if (dgs[N]>0) fprintf(file, "*%c^%ld", letters[N],dgs[N]);
}


/**
 * fprintPoly:
 * @file: A handler of a file.
 * @Ptr: A C-Cube polynomial.
 * 
 * Print 'Ptr' into the tex FILE 'file'.
 * This is a 1-D representation which can be parsed by
 * Maple or MAGMA 
 * Return value: 
 **/
void fprintPoly(FILE *file, preFFTRep * Ptr){

    if(!zeroPolyp(Ptr)){
        if (N(Ptr)==0)  {fprintf(file, "%ld\n", DATI(Ptr, 0)); return;}
        fprintPoly_inner(file, N(Ptr), BUSZS(Ptr), CUM(Ptr), DAT(Ptr));
        fprintf(file, "\n");
    }
    else fprintf(file, "0\n");

}

// Subroutine for next function
void randomPoly_inner(sfixn N, sfixn * dgs, sfixn * accum, sfixn * data, sfixn p){
    int i, offset=0;

    if(N==1){
        //srand(getSeed());
        for(i=0; i<=dgs[N]; i++){
            data[i]=(rand()%p);
        }
        if (! data[dgs[N]]) data[dgs[N]]=1;
        return;
    }


    for(i=0; i<=dgs[N]; i++){
        offset=accum[N]*i;
        randomPoly_inner(N-1, dgs, accum, data+offset, p);
    } 

}


/**
 * randomPoly:
 * @Ptr: A C-Cube polynomial.
 * @p: A prime number.
 * 
 * The input 'Ptr' is an 'clean' C-Cube polynomial, i.e. 
 * 'Ptr' is initialized with zero coefficients and partial degrees.
 * This routine fills random numbers in Z/pZ into 'Ptr's 
 * coefficient vector.
 *
 * Return value: 
 **/
void randomPoly(preFFTRep * Ptr, sfixn p){
    randomPoly_inner(N(Ptr), BUSZS(Ptr), CUM(Ptr), DAT(Ptr), p);  
}

void sparsify(preFFTRep * Ptr){
    int i=0;
    //srand(getSeed());

    for(i=0; i<SIZ(Ptr); i++){
        if(rand()%2) DATI(Ptr, i)=0;
    }

}

void randomSparsePoly(preFFTRep * Ptr, sfixn p){
    randomPoly(Ptr, p);
    sparsify(Ptr);

}

// Subroutine for the next function
void randomMonicPoly_inner(sfixn N, sfixn N1, sfixn * dgs, sfixn * accum, 
        sfixn * data, sfixn p)
{
    int i, offset=0;
    if(N1==1){
        //srand(getSeed() );
        for(i=0; i<=dgs[N1]; i++){ data[i]=(rand()%p); }
        if (! data[dgs[N1]]) data[dgs[N1]]=1;
        if(N==1) data[dgs[N]]=1;
        return;
    }

    if(N==N1){
        for(i=0; i<dgs[N1]; i++){
            offset=accum[N1]*i;
            randomMonicPoly_inner(N, N1-1, dgs, accum, data+offset, p);
        }
        (data+accum[N]*dgs[N])[0]=1;}

    else{
        for(i=0; i<=dgs[N1]; i++){
            offset=accum[N1]*i;
            randomMonicPoly_inner(N, N1-1, dgs, accum, data+offset, p);
        }

    }
}


void randomMonicPoly(preFFTRep * Ptr, sfixn p){
    randomMonicPoly_inner(N(Ptr), N(Ptr), BUSZS(Ptr), CUM(Ptr), DAT(Ptr), p);  
}


/**
 * EX_randomMonicPoly:
 * @N: Number of variables.
 * @dgs: The degree vector of polynomials.
 * @p: A prime number.
 *  
 * Create a random C-Cube polynomial where all univariate
 *        polynomial coefficients are monic
 *
 * Return value: 
 **/
preFFTRep *
EX_randomMonicPoly(sfixn N, sfixn * dgs, sfixn p){
    preFFTRep *poly;
    poly = EX_InitOnePoly(N, dgs);
    randomMonicPoly(poly, p);
    return poly;
}


// Testing purpose only
void randomMonicPoly_inner_allOne(sfixn N, sfixn N1, sfixn * dgs, 
        sfixn * accum, sfixn * data, sfixn p)
{
    int i, offset=0;
    if(N1==1){
        for(i=0; i<=dgs[N1]; i++){ data[i]=2; }
        if (! data[dgs[N1]]) data[dgs[N1]]=1;
        if(N==1) data[dgs[N]]=1;
        return;
    }

    for(i=0; i<dgs[N1]; i++){
        offset=accum[N1]*i;
        randomMonicPoly_inner_allOne(N, N1-1, dgs, accum, data+offset, p);
    }
    // make the leadingCoef monic.
    if(N==N1) (data+accum[N]*dgs[N])[0]=1;
}

void randomMonicPoly_allOne(preFFTRep * Ptr, sfixn p){
    randomMonicPoly_inner_allOne(N(Ptr), N(Ptr), BUSZS(Ptr), CUM(Ptr), DAT(Ptr), p);  
}

/**
 * printTriSet:
 * @tPtr: a triangular set 
 * 
 * Pretty printer for the input triangular set.
 * The output looks like a Maple list.
 * 
 * Return value: 
 **/   
void printTriSet(TriSet * tPtr){
    register int i;
    printf("[\n");
    for(i=1;i<N(tPtr);i++) {
        if( ELEMI(tPtr, i) ){
            printPoly(ELEMI(tPtr, i));
            printf(", \n");
        }
    }

    if( ELEMI(tPtr, N(tPtr)) ){
        printPoly(ELEMI(tPtr, N(tPtr)));
        printf("]\n");
    }
}

/**
 * printTriRevInvSet:
 * @triPt: the modular inverses of the triangular set
 * 
 * Pretty printer for these polynomials.
 * The output looks like a Maple list.
 * 
 * Return value: 
 **/   
void printTriRevInvSet(TriRevInvSet * triPtr){
    register int i;
    printf("[ ");
    for(i=1;i<N(triPtr);i++) { 
        if(EXSTI(triPtr, i)){
            printPoly(ELEMI(triPtr, i));
            printf(", \n");}
    }
    if(EXSTI(triPtr, N(triPtr))) printPoly(ELEMI(triPtr, N(triPtr)));
    printf(" ]\n");
}



/**
 * freePoly:
 * @x: A C-Cube polynomial.
 * 
 * Free a C-Cube polynomial.
 *
 * Return value: 
 **/
void freePoly(preFFTRep * x){

    if(BUSZS(x) !=NULL) {my_free(BUSZS(x)); BUSZS(x)=NULL;}
    if(CUTS(x) !=NULL) {my_free(CUTS(x)); CUTS(x)=NULL;}
    if(CUM(x) !=NULL) {my_free(CUM(x)); CUM(x)=NULL;}
    if(DAT(x) !=NULL) {my_free(DAT(x)); DAT(x)=NULL;}


}

// free a Kronecker data-structure
void freeKroFFTRep(KroFFTRep * x){
    int i;
    if(KROOTS(x) != NULL) {my_free(KROOTS(x)); KROOTS(x)=NULL;}
    if(CUM(x) !=NULL) {my_free(CUM(x)); CUM(x)=NULL;}
    if(ES(x) !=NULL) {my_free(ES(x)); ES(x)=NULL;}
    if(DIMS(x) !=NULL) {my_free(DIMS(x)); DIMS(x)=NULL;}
    if(DATS(x) != NULL){
        for(i=0; i<M(x); i++) 
            if(DATSI(x, i) !=NULL) {my_free(DATSI(x, i)); DATSI(x, i)=NULL;}
        my_free(DATS(x));
        DATS(x)=NULL;}
}


/**
 * InitKroFFTRep:
 * @kPtr: (output) A data structure to prepare the Kronecker-based FFT multiplication.
 * @resDgs: The degree vector of the result polynomial.
 * @N: The number of variables.
 * @M: The number of polynomials to multiply (poly1* poly2 * ... * polyM.
 * @pPtr: The information of the prime p.
 * 
 * Create 'kPtr' -- a data structure of type KroFFTRep.
 * This data structure will be used for Kronecker-based FFT multiplicaiton.
 *
 * Return value: 
 **/
void InitKroFFTRep(KroFFTRep * kPtr, sfixn * resDgs, sfixn N, sfixn M,
        MONTP_OPT2_AS_GENE * pPtr)
{
    register int j;
    N(kPtr)=N;     
    M(kPtr)=M;
    ES(kPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn)); 
    DIMS(kPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUM(kPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUMI(kPtr, 1)= 1; SIZ(kPtr)=1;
    for(j=1;j<=N;j++){
        ESI(kPtr, j)=logceiling(resDgs[j]+1);
        DIMSI(kPtr, j)=1<<(ESI(kPtr, j));
        SIZ(kPtr)*=DIMSI(kPtr, j); 
        if(j>=2){CUMI(kPtr, j)=CUMI(kPtr, j-1)*DIMSI(kPtr, j-1);}
    }

    DATS(kPtr)=(sfixn **)my_calloc(M,sizeof(sfixn *));
    for(j=0;j<M;j++){
        DATSI(kPtr, j)=(sfixn * )my_calloc(SIZ(kPtr),sizeof(sfixn));
    }
    KN(kPtr)=1; KE(kPtr)=0;
    for(j=1;j<=N;j++) {KN(kPtr)*=DIMSI(kPtr, j); KE(kPtr)+=ESI(kPtr, j);}
    KROOTS(kPtr)=NULL;
    kPtr->Defsize=SIZ(kPtr);
}

// cannot decrease dimension, only es, dims, siz, aacum.
void decreaseKroFFTRep(KroFFTRep * kPtr, sfixn * resDgs){
    register int i,j;
    CUMI(kPtr, 1)= 1; SIZ(kPtr)=1;
    for(j=1;j<=N(kPtr);j++){
        ESI(kPtr, j)=logceiling(resDgs[j]+1);
        DIMSI(kPtr, j)=1<<(ESI(kPtr, j));
        SIZ(kPtr)*=DIMSI(kPtr, j); 
        if(j>=2){CUMI(kPtr, j)=CUMI(kPtr, j-1)*DIMSI(kPtr, j-1);}
    }

    KN(kPtr)=1; KE(kPtr)=0;
    for(j=1;j<=N(kPtr);j++) {KN(kPtr)*=DIMSI(kPtr, j); KE(kPtr)+=ESI(kPtr, j);}

    for(i=0;i<M(kPtr);i++){
        for(j=0;j<kPtr->Defsize;j++){
            (DATSI(kPtr, i))[j]=0;}}
    if(KROOTS(kPtr)!=NULL) {my_free(KROOTS(kPtr)); KROOTS(kPtr)=NULL;}
}



/**
 * InitResPoly:
 * @rPtr: (output) A C-Cube polynomial.
 * @N: Number of variables.
 * @p1dgs: A degree vector for some C-Cube polynomial p1.
 * @p2dgs: A degree vector for some C-Cube polynomial p2.
 * 
 * Initialized a 'clean' (all zero-coefficients) C-Cube polynomial whoes
 * data space can exactly keep the product of p1 and p2.
 *
 * Return value: 
 **/
void InitResPoly(preFFTRep * rPtr, sfixn N, sfixn * p1dgs, sfixn * p2dgs){  
    register int j;
    N(rPtr)=N;
    BUSZS(rPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUTS(rPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    //
    SIZ(rPtr)=1;
    CUM(rPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    //
    CUMI(rPtr, 1)= 1;
    for(j=1;j<=N;j++){
        BUSZSI(rPtr, j)=p1dgs[j]+p2dgs[j];
        CUTSI (rPtr, j)=   BUSZSI(rPtr, j);
        SIZ(rPtr)=SIZ(rPtr)*(BUSZSI(rPtr, j)+1);
        if(j>=2){
            CUMI(rPtr, j)=CUMI(rPtr, j-1)*(BUSZSI(rPtr, j-1)+1);
        }
    }
    OFST(rPtr)=0;
    DAT(rPtr)=(sfixn * )my_calloc( SIZ(rPtr),sizeof(sfixn));
    //
    DFN(rPtr)=N(rPtr);
    DFSIZ(rPtr)=SIZ(rPtr);
    DEFDAT(rPtr)=DAT(rPtr);
    //BUSZSI(rPtr, 0)=BUSZSI(rPtr, N);
}

/**
 * InitResPoly:
 * @N: Number of variables.
 * @p1dgs: A degree vector for some C-Cube polynomial p1.
 * @p2dgs: A degree vector for some C-Cube polynomial p2.
 * 
 * Initialized a 'clean' (all zero-coefficients) C-Cube polynomial whoes
 * data spece can exactly keep the product of p1 and p2.
 *
 * Return value: A newly created C-Cube polynomial whoes
 * data spece can exactly keep the product of p1 and p2.   
 **/
preFFTRep *
EX_InitResPoly(sfixn N, sfixn * p1dgs, sfixn * p2dgs){

    preFFTRep *rPtr;
    rPtr=(preFFTRep *) my_malloc(sizeof(preFFTRep));
    InitResPoly(rPtr, N, p1dgs, p2dgs);
    return rPtr;
}

// Depreciated code
// the same as InitOneReducedPoly()
void InitOneReducedPoly(preFFTRep * rPtr, sfixn N, sfixn * p1dgs){  
    register int j; 
    N(rPtr)=N;
    BUSZS(rPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUTS(rPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    SIZ(rPtr)=1;
    CUM(rPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUMI(rPtr, 1)= 1;
    for(j=1;j<=N;j++){
        BUSZSI(rPtr, j)=p1dgs[j];
        CUTSI (rPtr, j)=p1dgs[j];   
        SIZ(rPtr)=SIZ(rPtr)*(BUSZSI(rPtr,j)+1);
        if(j>=2){
            CUMI(rPtr, j)=CUMI(rPtr, j-1)*(BUSZSI(rPtr,j-1)+1);
        }
    }
    OFST(rPtr)=0;
    DAT(rPtr)=(sfixn * )my_calloc( SIZ(rPtr),sizeof(sfixn));  
    //
    //BUSZSI(rPtr, 0)=BUSZSI(rPtr, N);
    DFN(rPtr)=N(rPtr);
    DFSIZ(rPtr)=SIZ(rPtr);
    DEFDAT(rPtr)=DAT(rPtr);
}

// Subroutine for next one
void InitOnePoly(preFFTRep * rPtr, sfixn N, sfixn * p1dgs){  
    register int j; 
    N(rPtr)=N;
    BUSZS(rPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUTS(rPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    SIZ(rPtr)=1;
    CUM(rPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUMI(rPtr, 1)= 1;
    for(j=1;j<=N;j++){
        BUSZSI(rPtr, j)=p1dgs[j];
        CUTSI (rPtr, j)=p1dgs[j];   
        SIZ(rPtr)=SIZ(rPtr)*(BUSZSI(rPtr,j)+1);
        if(j>=2){
            CUMI(rPtr, j)=CUMI(rPtr, j-1)*(BUSZSI(rPtr,j-1)+1);
        }
    }
    OFST(rPtr)=0;
    DAT(rPtr)=(sfixn * )my_calloc( SIZ(rPtr),sizeof(sfixn));  
    //
    //BUSZSI(rPtr, 0)=BUSZSI(rPtr, N);
    DFN(rPtr)=N(rPtr);
    DFSIZ(rPtr)=SIZ(rPtr);
    DEFDAT(rPtr)=DAT(rPtr);
}


/**
 * EX_InitOnePoly:
 * @N: Number of variables.
 * @dgs: A partial-degree vector of size N.
 * 
 * To initialize a C-Cube polynomial whoes partial 
 * degrees are defined in 'dgs'.
 *
 * Return value: A newly created C-Cube polynomial
 **/
preFFTRep *
EX_InitOnePoly(sfixn N, sfixn * dgs){
    preFFTRep *poly= (preFFTRep *) my_malloc(sizeof(preFFTRep));
    InitOnePoly(poly, N, dgs);
    return poly;
}

/**
 * EX_randomPoly:
 * @N: Number of variables.
 * @dgs: A partial-degree vector of size N.
 * @p: A prime number. 
 *
 * To initialize a C-Cube polynomial whoes partial degrees are defined in 'dgs'.
 * Then generate random coefficients in Z/pZ for this C-Cube polynomial.
 * Return value: The newly created C-Cube polynomial
 **/
preFFTRep *
EX_randomPoly(sfixn N, sfixn * dgs, sfixn p){
    preFFTRep *poly;
    poly = EX_InitOnePoly(N, dgs);
    randomPoly(poly, p);
    return poly;
}


// Initiazes two C-cube polynomials. Not used anymore
void InitTwoRandomReducedInputPolys(sfixn * bounds, sfixn N, preFFTRep * p1Ptr, 
        preFFTRep * p2Ptr,  MONTP_OPT2_AS_GENE * pPtr)
{
    register int j;

    BUSZS(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn)); 
    CUTS(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn)); 
    //
    CUM(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    //
    BUSZS(p2Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    CUTS(p2Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    //
    CUM(p2Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    //
    CUMI(p1Ptr, 1)= CUMI(p2Ptr, 1)= 1;
    SIZ(p1Ptr)=SIZ(p2Ptr)=1; 
    for(j=1;j<=N;j++){
        BUSZSI(p1Ptr, j)=bounds[j];
        BUSZSI(p2Ptr, j)=bounds[j];
        CUTSI(p1Ptr, j)=bounds[j];
        CUTSI(p2Ptr, j)=bounds[j];
        SIZ(p1Ptr)*=BUSZSI(p1Ptr, j)+1;
        SIZ(p2Ptr)*=BUSZSI(p2Ptr, j)+1;    
        if(j>=2){
            CUMI(p1Ptr, j)=CUMI(p1Ptr, j-1)*(BUSZSI(p1Ptr, j-1)+1);
            CUMI(p2Ptr, j)=CUMI(p2Ptr, j-1)*(BUSZSI(p2Ptr, j-1)+1);

        }
    }
    DAT(p1Ptr)=(sfixn * )my_calloc(SIZ(p1Ptr),sizeof(sfixn));
    //
    DAT(p2Ptr)=(sfixn * )my_calloc(SIZ(p2Ptr),sizeof(sfixn));
    //
    OFST(p1Ptr)=0;
    OFST(p2Ptr)=0;
    randomPoly(p1Ptr, pPtr->P);
    randomPoly(p2Ptr, pPtr->P);
    DFN(p1Ptr)=N(p1Ptr);
    DFSIZ(p1Ptr)=SIZ(p1Ptr);
    DEFDAT(p1Ptr)=DAT(p1Ptr);
    DFN(p2Ptr)=N(p2Ptr);
    DFSIZ(p2Ptr)=SIZ(p2Ptr);
    DEFDAT(p2Ptr)=DAT(p2Ptr);
    //BUSZSI(p1Ptr, 0)=BUSZSI(p1Ptr, N);
    //BUSZSI(p2Ptr, 0)=BUSZSI(p2Ptr,N);
}


// Depreciated code
void InitOneRandomReducedInputPoly(sfixn * bounds, sfixn N, preFFTRep * p1Ptr,
        MONTP_OPT2_AS_GENE * pPtr)
{
    register int j;

    BUSZS(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn)); 
    CUTS(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn)); 
    //
    CUM(p1Ptr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
    //
    CUMI(p1Ptr, 1)= 1;
    SIZ(p1Ptr)=1; 
    for(j=1;j<=N;j++){
        BUSZSI(p1Ptr, j)=bounds[j];
        CUTSI(p1Ptr, j)=bounds[j];
        SIZ(p1Ptr)*=BUSZSI(p1Ptr, j)+1;
        if(j>=2){
            CUMI(p1Ptr, j)=CUMI(p1Ptr, j-1)*(BUSZSI(p1Ptr, j-1)+1);
        }
    }
    DAT(p1Ptr)=(sfixn * )my_calloc(SIZ(p1Ptr),sizeof(sfixn));
    //
    randomPoly(p1Ptr, pPtr->P);
    //BUSZSI(p1Ptr, 0)=BUSZSI(p1Ptr, N); 
    DFN(p1Ptr)=N(p1Ptr);
    DFSIZ(p1Ptr)=SIZ(p1Ptr);
    DEFDAT(p1Ptr)=DAT(p1Ptr);
}

static void KroneckerCleanData(KroFFTRep * kPtr){
    register int i,j;
    for(i=0;i<M(kPtr);i++){
        for(j=0;j<kPtr->Defsize;j++){
            (DATSI(kPtr, i))[j]=0;
        }
    }
}

//============================================================
// rPtr = p1Ptr * p2Ptr
//============================================================

/**
 * mulPoly_FFT:
 * @N: Number of variables.
 * @kPtr: The Kronecker data structure.
 * @rPtr: (output) The output prod of 'p1Ptr' and 'p2Ptr'.
 * @p1Ptr: A C-Cube polynomial.
 * @p2Ptr: A C-Cube polynomial.
 * @pPtr: Information for the Fourier prime number p. 
 *
 * Compute the product of p1Ptr and p2Ptr  by either Kronecker-based univariate 
 * FFT approach or by direct Multi-dimensional FFT.
 *
 * Return value: 
 **/
void mulPoly_FFT(sfixn N, KroFFTRep * kPtr, preFFTRep * rPtr, preFFTRep * p1Ptr, 
        preFFTRep * p2Ptr,  MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn kdg1=0, kdg2=0;
    int switcher=0;

    if ((KE(kPtr)) > (pPtr->Npow)) { 
        printf("FFT size larger than the prime %ld can handle!", pPtr->P);
        switcher = 1;
    }

    fromtofftRepMultiD(N,  CUM(kPtr), DATSI(kPtr, 0), CUM(p1Ptr), BUSZS(p1Ptr), DAT(p1Ptr));
    fromtofftRepMultiD(N,  CUM(kPtr), DATSI(kPtr, 1), CUM(p2Ptr), BUSZS(p2Ptr), DAT(p2Ptr));

    if ((switcher) || (KN(kPtr) < 4)) {
        fftMultiD_test(DATSI(kPtr, 0), DATSI(kPtr, 1), N, ES(kPtr), DIMS(kPtr), pPtr);
    } else {
        kdg1 = (BUSZSI(p1Ptr, N)+1)*CUMI(kPtr, N)-1;
        kdg2 = (BUSZSI(p2Ptr, N)+1)*CUMI(kPtr, N)-1;
        KROOTS(kPtr)=(sfixn *)my_calloc(KN(kPtr), sizeof(sfixn));
        EX_Mont_GetNthRoots_OPT2_AS_GENE(KE(kPtr), KN(kPtr), KROOTS(kPtr),pPtr);
        EX_KN_Mont_FFTMul_OPT2_AS_GENE_1(KN(kPtr), KE(kPtr), 0, KROOTS(kPtr), 
                kdg1, DATSI(kPtr, 0), kdg2, DATSI(kPtr, 1), pPtr);
    }

    fromtofftRepMultiD(N,  CUM(rPtr), DAT(rPtr), CUM(kPtr), BUSZS(rPtr), DATSI(kPtr, 0));
}

// 1 using Multi-D FFT.
// 0 using Kronecker FFT.
void mulPoly_FFT_select(sfixn N, KroFFTRep * kPtr, preFFTRep * rPtr, 
        preFFTRep * p1Ptr, preFFTRep * p2Ptr, 
        MONTP_OPT2_AS_GENE * pPtr, int switcher)
{
    sfixn kdg1=0, kdg2=0;

    if((KE(kPtr))>(pPtr->Npow)) {   
        printf("FFT size larger than the prime %ld can handle!", pPtr->P);
        switcher = 1;
    }

    fromtofftRepMultiD(N,  CUM(kPtr), DATSI(kPtr, 0), CUM(p1Ptr), BUSZS(p1Ptr), DAT(p1Ptr));
    fromtofftRepMultiD(N,  CUM(kPtr), DATSI(kPtr, 1), CUM(p2Ptr), BUSZS(p2Ptr), DAT(p2Ptr));

    if (switcher) {
        fftMultiD_test(DATSI(kPtr, 0), DATSI(kPtr, 1), N, ES(kPtr), DIMS(kPtr), pPtr);
    } else{
        kdg1= (BUSZSI(p1Ptr, N)+1)*CUMI(kPtr, N)-1;
        kdg2= (BUSZSI(p2Ptr, N)+1)*CUMI(kPtr, N)-1;
        KROOTS(kPtr)=(sfixn *)my_calloc(KN(kPtr), sizeof(sfixn));
        EX_Mont_GetNthRoots_OPT2_AS_GENE(KE(kPtr), KN(kPtr), KROOTS(kPtr),pPtr);
        EX_KN_Mont_FFTMul_OPT2_AS_GENE_1(KN(kPtr), KE(kPtr), 0, KROOTS(kPtr), 
                kdg1, DATSI(kPtr, 0), kdg2, DATSI(kPtr, 1), pPtr);
    }
    fromtofftRepMultiD(N,  CUM(rPtr), DAT(rPtr), CUM(kPtr), BUSZS(rPtr), DATSI(kPtr, 0));
}

/**
 * EX_mulPoly_FFT:
 * @N: Number of variables.
 * @resPtr: (output) The output prod of 'p1Ptr' and 'p2Ptr'.
 * @f1: A C-Cube polynomial.
 * @f2: A C-Cube polynomial.
 * @pPtr: Information for the Fourier prime number p. 
 *
 * Compute the product of f1 and f2  by either Kronecker-based univariate FFT 
 * approach or by direct Multi-dimensional FFT.
 *
 * Return value: The product of f1 and f2.
 **/
void EX_mulPoly_FFT(sfixn N, preFFTRep * resPtr, preFFTRep * f1, preFFTRep * f2,  
        MONTP_OPT2_AS_GENE *pPtr)
{
    KroFFTRep kro;
    InitKroFFTRep(&kro, BUSZS(resPtr), N, 2, pPtr);
    mulPoly_FFT(N, &kro, resPtr, f1, f2,  pPtr);
    freeKroFFTRep(&kro);
    return;
}

/**
 * EX_mulPoly:
 * @N: Number of variables.
 * @resPtr: (output) The output prod of 'p1Ptr' and 'p2Ptr'.
 * @f1: A C-Cube polynomial.
 * @f2: A C-Cube polynomial.
 * @pPtr: Information for the Fourier prime number p. 
 *
 * Compute the product of f1 and f2  by either Kronecker-based univariate FFT 
 * approach or by direct Multi-dimensional FFT or by classical multiplication.
 *
 * Return value:.
 **/
void EX_mulPoly(sfixn N, preFFTRep * resPtr, preFFTRep * f1, preFFTRep * f2,  
        MONTP_OPT2_AS_GENE *pPtr)
{
    sfixn sz1, sz2;
    sz1 = getDenseSiz(N, f1, BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    sz2 = getDenseSiz(N, f2, BUSZSI(f2, N), DAT(f2), CUMI(f2, N));

    if((sz1>=MulCut)||(sz2>=MulCut)){
        EX_mulPoly_TFTFFT(N, resPtr, f1, f2, pPtr);
    } else {
        plainMultiDMul(N, CUM(resPtr), DAT(resPtr), CUM(f1), BUSZS(f1), 
                CUM(f2), BUSZS(f2), DAT(f1), DAT(f2), pPtr);
    }
}

/**
 * EX_mulPoly:
 * @N: Number of variables.
 * @f1: A C-Cube polynomial.
 * @f2: A C-Cube polynomial.
 * @pPtr: Information for the Fourier prime number p. 
 *
 * Compute the product of 'f1' and 'f2' by either Kronecker-based univariate 
 * FFT approach or by direct Multi-dimensional FFT or by classical multiplication.
 *
 * Return value:The product of 'f1' and 'f2'.
 **/
preFFTRep * EX_EX_mulPoly(sfixn N,  preFFTRep * f1, preFFTRep * f2,
        MONTP_OPT2_AS_GENE *pPtr)
{
    sfixn sz1, sz2;
    preFFTRep * resPtr;

    resPtr = EX_InitResPoly(N,  BUSZS(f1), BUSZS(f2));

    sz1=getDenseSiz(N, f1, BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    sz2=getDenseSiz(N, f2, BUSZSI(f2, N), DAT(f2), CUMI(f2, N));

    if((sz1>=MulCut)||(sz2>=MulCut)){
        EX_mulPoly_TFTFFT(N, resPtr, f1, f2, pPtr);
    }else{
        plainMultiDMul(N, CUM(resPtr), DAT(resPtr), CUM(f1), BUSZS(f1), 
                CUM(f2), BUSZS(f2), DAT(f1), DAT(f2), pPtr);
    }
    return resPtr;
}

preFFTRep * EX_EX_TFTFFTMulPoly(sfixn N,  preFFTRep * f1, preFFTRep * f2,
        MONTP_OPT2_AS_GENE *pPtr)
{
    sfixn sz1, sz2;
    preFFTRep * resPtr;

    resPtr = EX_InitResPoly(N,  BUSZS(f1), BUSZS(f2));

    sz1=getDenseSiz(N, f1, BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    sz2=getDenseSiz(N, f2, BUSZSI(f2, N), DAT(f2), CUMI(f2, N));

    EX_mulPoly_TFTFFT(N, resPtr, f1, f2, pPtr);

    return resPtr;
}

preFFTRep * EX_EX_PlainMulPoly(sfixn N,  preFFTRep * f1, preFFTRep * f2,
        MONTP_OPT2_AS_GENE *pPtr)
{
    sfixn sz1, sz2;
    preFFTRep * resPtr;

    resPtr = EX_InitResPoly(N,  BUSZS(f1), BUSZS(f2));

    sz1=getDenseSiz(N, f1, BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    sz2=getDenseSiz(N, f2, BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
    plainMultiDMul(N, CUM(resPtr), DAT(resPtr), CUM(f1), BUSZS(f1), 
            CUM(f2), BUSZS(f2), DAT(f1), DAT(f2), pPtr);

    return resPtr;
}


void EX_mulPoly_FFT_select(sfixn N, preFFTRep * resPtr, preFFTRep * f1, 
        preFFTRep * f2,  MONTP_OPT2_AS_GENE *pPtr, int switcher)
{
    KroFFTRep kro;
    InitKroFFTRep(&kro, BUSZS(resPtr), N, 2, pPtr);
    mulPoly_FFT_select(N, &kro, resPtr, f1, f2,  pPtr, switcher);
    freeKroFFTRep(&kro);
    return;
}

//===========================================================
// rPtr = p1Ptr ^ 2
//===========================================================


/**
 * squarePoly_FFT:
 * @N: number of variables.
 * @kPtr: the Kronecker presentation.
 * @rPtr: (output) C-Cube polynomial to keep the square (p1)^2.
 * @p1Ptr: C-Cube prolynomial p1.
 * @pPtr: Information for the prime number.
 * 
 * Return value: 
 **/
void squarePoly_FFT(sfixn N, KroFFTRep * kPtr, preFFTRep * rPtr, preFFTRep *p1Ptr,
        MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn kdg1;
    int switcher=0;

    if((KE(kPtr))>(pPtr->Npow)) { switcher=1;} 
    fromtofftRepMultiD(N,  CUM(kPtr), DATSI(kPtr, 0), CUM(p1Ptr), BUSZS(p1Ptr), DAT(p1Ptr));
    if(switcher){
        fftMultiD_square_test(DATSI(kPtr, 0), N, ES(kPtr), DIMS(kPtr), pPtr); }
    else{
        kdg1= (BUSZSI(p1Ptr, N)+1)*CUMI(kPtr, N)-1;
        KROOTS(kPtr)=(sfixn *)my_calloc(KN(kPtr), sizeof(sfixn));
        EX_Mont_GetNthRoots_OPT2_AS_GENE(KE(kPtr), KN(kPtr), KROOTS(kPtr),pPtr);
        EX_KN_Mont_FFTSQUARE_OPT2_AS_GENE_1(KN(kPtr), KE(kPtr), 0, KROOTS(kPtr), kdg1, DATSI(kPtr, 0), pPtr);
    }
    fromtofftRepMultiD(N,  CUM(rPtr), DAT(rPtr), CUM(kPtr), BUSZS(rPtr), DATSI(kPtr, 0));
}

void compareVecLx(sfixn N, sfixn * v1, sfixn * v2){
    sfixn i;
    for(i=1; i<=N; i++)
        if(v1[i]<v2[i]) printf("error v1[%ld]=%ld, v2[%ld]=%ld\n",  i, v1[i], i, v2[i]);
}


static 
void
message(sfixn N, const char * msg){
    printf("N=%ld   ", N);
    printf(msg);
    printf("\n");
}



static 
int productDegsOk(sfixn N, preFFTRep * productPtr,  TriSet * ts, TriRevInvSet * tris){
    register int i;
    sfixn n=0,m=0, b=0;
    for(i=1;i<=N;i++){
        n=BUSZSI(productPtr, i);
        m=BDSI(ts, i)+1;
        b=(NLB(tris))[i];
        if((n-m)>b) {
            printf("n=%ld, m=%ld, b=%ld. %d is bigger than this program can handle.\n", 
                    n,m,b,i); return(0);
        }
    }  
    return 1;
}



//===================================================
// MulMod
// 
// Distructive to f1 if selector == 1;
// Distructive to f2 if selector != 1;
//===================================================

/**
 * mul_Reduced_1:
 * @resPtr: (Output) the product:  'f1' * 'f2'.
 * @N: The number of variables.
 * @f1: C-Cube polynomial.
 * @f2: C-Cube polynomial.
 * @ts: A triangular set.
 * @tris: The modular inverses of 'ts'
 * @pPtr: The information for the prime.
 * @selector: Set to be 1 then 'f1' is distructed , otherwise 'f2' is distructed.
 *            The distructed one keeps the final modular product 'f1' * 'f2' mod 'ts'.
 *
 * Return value: 
 **/
void mul_Reduced_1(preFFTRep * resPtr, sfixn N, preFFTRep * f1, preFFTRep * f2, 
        TriSet * ts, TriRevInvSet * tris, MONTP_OPT2_AS_GENE * pPtr, int selector)
{
    KroFFTRep kro;
    sfixn d1, d2, d3, d4, d5, sz1, sz2;

    d1=shrinkDeg(BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    d2=shrinkDeg(BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
    //sz1=(d1+1)*CUMI(f1, N);
    //sz2=(d2+1)*CUMI(f2, N);
    sz1=getDenseSiz(N, f1, BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    sz2=getDenseSiz(N, f2, BUSZSI(f2, N), DAT(f2), CUMI(f2, N));

    d3=BUSZSI(f1, N);
    d4=BUSZSI(f2, N);

    d5=BUSZSI(resPtr, N);

    if((sz1>=MulCut)||(sz2>=MulCut)){
        //

        BUSZSI(f1, N)=d1;
        BUSZSI(f2, N)=d2;   
        BUSZSI(resPtr, N)=d1+d2;
        InitKroFFTRep(&kro, BUSZS(resPtr), N, 2, pPtr); 

        //================================================================
        mulPoly_FFT(N, &kro, resPtr, f1, f2,  pPtr);
        //================================================================

        BUSZSI(f1, N)=d3;
        BUSZSI(f2, N)=d4;   
        BUSZSI(resPtr, N)=d1+d2;  

        freeKroFFTRep(&kro);}
    else{

        if(N==1){

            EX_Mont_PlainMul_OPT2_AS_GENE(BUSZSI(resPtr, 1), DAT(resPtr), 
                    d1, DAT(f1), d2, DAT(f2), pPtr);

        } else{

            plainMultiDMul(N, CUM(resPtr), DAT(resPtr), CUM(f1), BUSZS(f1), 
                    CUM(f2), BUSZS(f2), DAT(f1), DAT(f2), pPtr);
        }
    }

    if(selector==1){
        PolyCleanData(f1);
        MultiMod(N, f1, resPtr, ts, tris, pPtr);}
    else{
        PolyCleanData(f2);
        MultiMod(N, f2, resPtr, ts, tris, pPtr);}

}


//===================================================
// MulMod
// (1) resPtr = f1 * f2.
// (2) out = resPtr modulo TS.
//  * resPtr is pre-allocated and passed as a parameter.
//  * resPtr will be destructed during MultiMod operation.
//===================================================
/**
 * mul_Reduced:
 * @resPtr: (output) the product: 'f1' * 'f2'.
 * @N: the number of variables.
 * @out: (output) the modular product: 'f1' * 'f2' mod 'ts' .
 * @f1: C-Cube polynomial.
 * @f2: C-Cube polynomial.
 * @ts: the triangular set.
 * @tris: the inverses of 'ts'.
 * @pPtr: the information of the prime.
 * 
 * Return value: 
 **/
static preFFTRep * mul_Reduced(preFFTRep * resPtr, sfixn N, preFFTRep * out, 
        preFFTRep * f1, preFFTRep * f2, TriSet * ts, TriRevInvSet * tris, 
        MONTP_OPT2_AS_GENE * pPtr)
{
    KroFFTRep kro;

    sfixn d1,d2,d3,d4,d5,sz1, sz2;
    d1=shrinkDeg(BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    d2=shrinkDeg(BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
    //sz1=(d1+1)*CUMI(f1, N);
    //sz2=(d2+1)*CUMI(f2, N);
    sz1=getDenseSiz(N, f1, BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    sz2=getDenseSiz(N, f2, BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
    d3=BUSZSI(f1, N);
    d4=BUSZSI(f2, N);

    if((sz1>=MulCut)||(sz2>=MulCut)){
        d5= BUSZSI(resPtr, N);
        BUSZSI(f1, N)=d1;
        BUSZSI(f2, N)=d2;
        BUSZSI(resPtr, N)=d1+d2;
        InitKroFFTRep(&kro, BUSZS(resPtr), N, 2, pPtr); 
        //================================================================
        mulPoly_FFT(N, &kro, resPtr, f1, f2,  pPtr);
        //================================================================
        BUSZSI(f1, N)=d3;
        BUSZSI(f2, N)=d4;
        BUSZSI(resPtr, N)=d5;

        freeKroFFTRep(&kro);}
    else{

        if(N==1){
            EX_Mont_PlainMul_OPT2_AS_GENE(BUSZSI(resPtr, 1), DAT(resPtr), d1, DAT(f1), d2, DAT(f2), pPtr); 
        } else{
            plainMultiDMul(N, CUM(resPtr), DAT(resPtr), CUM(f1), BUSZS(f1), 
                    CUM(f2), BUSZS(f2), DAT(f1), DAT(f2), pPtr); }
    }

    MultiMod(N, out, resPtr, ts, tris, pPtr);
    return out;
}




//===================================================
// MulMod
// (1) resPtr = f1 * f2.
// (2) out = resPtr modulo TS.
//  * resPtr is allocated within the function.
//===================================================
/**
 * EX_mul_Reduced:
 * @N: Number of variables.
 * @out: (output) 'f1'*'f2' modulo 'ts'.
 * @f1: A input C-Cube polynomial.
 * @f2: A input C-Cube polynomial.
 * @ts: A triangular set.
 * @tris: Inverses (Reverse-ordered) of 'ts'.
 * @pPtr: Information for the prime p.
 * 
 * Return value: 
 **/
preFFTRep * EX_mul_Reduced(sfixn N, preFFTRep * out, preFFTRep * f1, preFFTRep * f2,
        TriSet * ts, TriRevInvSet * tris, MONTP_OPT2_AS_GENE * pPtr)
{
    preFFTRep res;
    //KroFFTRep kro;
    //double tmp;

    sfixn d1,d2,d3,d4,d5,sz1, sz2;
    d1=shrinkDeg(BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    d2=shrinkDeg(BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
    //sz1=getDenseSiz(f1);
    //sz2=getDenseSiz(f2);
    sz1=getDenseSiz(N, f1, BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    sz2=getDenseSiz(N, f2, BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
    d3=BUSZSI(f1, N);
    d4=BUSZSI(f2, N);
    InitResPoly(&res, N,  BUSZS(f1), BUSZS(f2));

    //printf("sz1=%ld.\n", sz1);
    //printf("sz2=%ld.\n", sz2);

    if((sz1>=MulCut)||(sz2>=MulCut)){

        //tmp=gettime();

        d5=BUSZSDI(res, N);
        BUSZSI(f1, N)=d1;
        BUSZSI(f2, N)=d2;
        BUSZSDI(res, N)=d1+d2;

        /*     InitKroFFTRep(&kro, BUSZSD(res), N, 2, pPtr);  */
        /*     //================================================================ */
        /*     mulPoly_FFT(N, &kro, &res, f1, f2,  pPtr); */
        /*     //================================================================ */
        /*     freeKroFFTRep(&kro); */

        EX_mulPoly_TFTFFT(N, &res, f1, f2, pPtr);



        BUSZSI(f1, N)=d3;
        BUSZSI(f2, N)=d4;
        BUSZSDI(res, N)=d5;

    }
    else{
        if(N==1){
            EX_Mont_PlainMul_OPT2_AS_GENE(BUSZSDI(res, 1), res.data, d1, DAT(f1), d2, DAT(f2), pPtr); 
        } else{
            plainMultiDMul(N, res.accum, res.data, CUM(f1), BUSZS(f1), 
                    CUM(f2), BUSZS(f2), DAT(f1), DAT(f2), pPtr); }
    }

    MultiMod(N, out, &res, ts, tris, pPtr);
    freePoly(&res);
    return out;
}


//===================================================
// MulMod
// (1) resPtr = f1 * f2.
// (2) out = resPtr modulo TS.
//  * resPtr is allocated within the function.
//===================================================
/**
 * EX_mul_Reduced_1:
 * @N: Number of variables.
 * @f1: A input C-Cube polynomial.
 * @f2: A input C-Cube polynomial.
 * @ts: A triangular set.
 * @tris: Inverses (Reverse-ordered) of 'ts'.
 * @pPtr: Information for the prime p.
 * 
 * Return value: 
 **/
preFFTRep * EX_mul_Reduced_1(sfixn N, preFFTRep * f1, preFFTRep * f2, TriSet * ts, 
        TriRevInvSet * tris, MONTP_OPT2_AS_GENE * pPtr)
{
    preFFTRep res;
    KroFFTRep kro;

    sfixn d1,d2,d3,d4,d5,sz1, sz2;
    d1=shrinkDeg(BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    d2=shrinkDeg(BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
    //sz1=(d1+1)*CUMI(f1, N);
    //sz2=(d2+1)*CUMI(f2, N);

    sz1=getDenseSiz(N, f1, BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    sz2=getDenseSiz(N, f2, BUSZSI(f2, N), DAT(f2), CUMI(f2, N));

    d3=BUSZSI(f1, N);
    d4=BUSZSI(f2, N);
    InitResPoly(&res, N,  BUSZS(f1), BUSZS(f2));

    if((sz1>=MulCut)||(sz2>=MulCut)){
        d5=BUSZSDI(res, N);
        BUSZSI(f1, N)=d1;
        BUSZSI(f2, N)=d2;
        BUSZSDI(res, N)=d1+d2;
        InitKroFFTRep(&kro, BUSZSD(res), N, 2, pPtr); 
        //================================================================
        mulPoly_FFT(N, &kro, &res, f1, f2,  pPtr);
        //================================================================
        BUSZSI(f1, N)=d3;
        BUSZSI(f2, N)=d4;
        BUSZSDI(res, N)=d5;
        freeKroFFTRep(&kro);}
    else{
        if(N==1){
            EX_Mont_PlainMul_OPT2_AS_GENE(BUSZSDI(res, 1), res.data, d1, DAT(f1), d2, DAT(f2), pPtr); 
        } else{
            plainMultiDMul(N, res.accum, res.data, CUM(f1), BUSZS(f1), CUM(f2), BUSZS(f2), DAT(f1), DAT(f2), pPtr); }
    }
    PolyCleanData(f1);
    MultiMod(N, f1, &res, ts, tris, pPtr);
    freePoly(&res);
    return f1;
}

//===========================================================
// Ptr = Ptr1 + Ptr2
//===========================================================
/**
 * addPoly:
 * @N: number of variables.
 * @Ptr: (output) 'P1' + 'P2' mod 'p'.
 * @Ptr1: C-Cube polynomial.
 * @Ptr2: C-Cube polynomial.
 * @p: the prime.
 * 
 * Return value: 
 **/
void addPoly(sfixn N, preFFTRep * Ptr,  preFFTRep * Ptr1, preFFTRep * Ptr2, sfixn p){
    CopyOnePoly(Ptr, Ptr1);
    addEqDgPoly_1(N, Ptr, Ptr2, p);
}

//===========================================================
// Ptr = Ptr1 - Ptr2
//===========================================================
/**
 * subPoly:
 * @N: number of variables.
 * @Ptr: (output) 'P1' - 'P2' mod 'p'.
 * @Ptr1: C-Cube polynomial.
 * @Ptr2: C-Cube polynomial.
 * @p: the prime.
 * 
 * Return value: 
 **/
void subPoly(sfixn N, preFFTRep * Ptr, preFFTRep * Ptr1, preFFTRep * Ptr2, sfixn p){
    CopyOnePoly(Ptr, Ptr1);
    subPoly_inner_1(N, CUM(Ptr), BUSZS(Ptr2), CUM(Ptr2), DAT(Ptr), DAT(Ptr2), p);
}

/**
 * EX_subPoly:
 * @N: number of variables.
 * @Ptr1: C-Cube polynomial  P1.
 * @Ptr2: C-Cube polynomial  P2.
 * @p: the prime.
 * 
 * Return value: 'P1' - 'P2' mod p. 
 **/
preFFTRep *
EX_SubPoly(preFFTRep * Ptr1, preFFTRep * Ptr2, MONTP_OPT2_AS_GENE * pPtr){
    preFFTRep *outPtr;
    sfixn N;
    N=N(Ptr1);
    assert(N=N(Ptr2));
    outPtr = EX_CopyOnePoly(Ptr1);
    subPoly_inner_1(N, CUM(outPtr), BUSZS(Ptr2), CUM(Ptr2), DAT(outPtr), DAT(Ptr2), pPtr->P);
    return outPtr;
}


// 1 -- yes they are euqal.
// 0 -- No they are NOT equal.
/**
 * EX_IsEqualPoly:
 * @Ptr1: a C-Cube polynomial 'P1'.
 * @Ptr2: a C-Cube polynomial 'P2'.
 * 
 * To compare if two polynomials are equal.
 *
 * Return value: if 'P1' is equal to 'P2', then return 1. Otherwise return 0.
 **/
int
EX_IsEqualPoly(preFFTRep * Ptr1, preFFTRep * Ptr2){
    if(N(Ptr1) != N(Ptr2)) return 0;
    if (! compareVec(N(Ptr1), BUSZS(Ptr1), BUSZS(Ptr2))) return 0;
    if (! compareVec(SIZ(Ptr1)-1, DAT(Ptr1), DAT(Ptr2))) return 0;
    return 1;
}



//===================================================
// powerPoly_1
// f = f^e
//
//===================================================

/**
 * powPoly_1:
 * @f: a C-Cube polynomial.
 * @e: the exponent.
 * @N: the number of variables.
 * @ts: the triangular set.
 * @tris: the modular inverses of 'ts'.
 * @pPtr: the information of the prime number.
 *
 *  
 * Compute 'f' to the 'e'-th power.
 * 
 * Return value:  'f' to the 'e'-th power.
 **/
preFFTRep * powPoly_1(preFFTRep *f, sfixn e, sfixn N, TriSet * ts, 
        TriRevInvSet * tris, MONTP_OPT2_AS_GENE * pPtr)
{
    int i;
    preFFTRep *poly = (preFFTRep *)my_malloc(sizeof(preFFTRep));
    preFFTRep *tmppoly;
    InitOnePoly(poly, N, ts->bounds);
    tmppoly = (preFFTRep *)my_malloc(sizeof(preFFTRep));


    if(e==0) {
        PolyCleanData(f);
        DATI(f,0) = 1;
        return f;
    }

    if(e==1) {
        return f;
    }

    InitOnePoly(tmppoly, N, ts->bounds);
    CopyOnePoly(tmppoly, f);
    for(i=2; i<=e; i++){
        PolyCleanData(poly);
        EX_mul_Reduced(N, poly, tmppoly, f, ts, tris, pPtr);
        CopyOnePoly(tmppoly, poly);
    }
    freePoly(tmppoly);
    my_free(tmppoly);
    CopyOnePoly(f, poly);
    freePoly(poly);
    my_free(poly);
    return f;
}


//===================================================
// f += f1 * f2;
//===================================================
    preFFTRep *
addMulPoly_1(preFFTRep *f, preFFTRep *f1, preFFTRep *f2, sfixn N, 
        TriSet * ts, TriRevInvSet * tris, MONTP_OPT2_AS_GENE * pPtr)
{

    preFFTRep *tmppoly = (preFFTRep *)my_malloc(sizeof(preFFTRep));
    InitOnePoly(tmppoly, N, ts->bounds);
    //    CopyOnePoly(tmppoly, f);
    EX_mul_Reduced(N, tmppoly, f1, f2, ts, tris, pPtr);
    addEqDgPoly_1(N, f, tmppoly, pPtr->P);
    freePoly(tmppoly);
    my_free(tmppoly);
    return f;
}


// f -= f1 * f2;
preFFTRep *
subMulPoly_1(preFFTRep *f, preFFTRep *f1, preFFTRep *f2, sfixn N, 
        TriSet * ts, TriRevInvSet * tris, MONTP_OPT2_AS_GENE * pPtr){

    preFFTRep *tmppoly = (preFFTRep *)my_malloc(sizeof(preFFTRep));
    //printf("f=");
    //printPoly(f);
    //printf("f1=");
    //printPoly(f1);
    //printf("f2=");
    //printPoly(f2);
    InitOnePoly(tmppoly, N, ts->bounds);
    //CopyOnePoly(tmppoly, f);
    EX_mul_Reduced(N, tmppoly, f1, f2, ts, tris, pPtr);
    //printf("tmppoly=");
    //printPoly(tmppoly);
    subEqDgPoly_1(N, f, tmppoly, pPtr->P,1);
    freePoly(tmppoly);
    my_free(tmppoly);
    return f;
}




/**
 * EX_mul_Coef_Reduced:
 * @N: number of variables of 'f1', 'f2'.
 * @out: (output)  'f1' * 'f2' modulo 'ts' where 'ts' has main variable X_{N-1}.
 * @f1: a C-Cube polynomial.
 * @f2: a C-Cube polynomial.
 * @ts: a triangular set.
 * @tris: inverses of 'ts'.
 * @pPtr: the information for the prime number.
 * 
 *  First, compute the product of 'f1' and 'f2'.
 *  Second, reduce the coefficients of the product w.r.t 'ts'.
 *
 * Return value: 
 **/
preFFTRep * EX_mul_Coef_Reduced(sfixn N, preFFTRep * out, preFFTRep * f1, 
        preFFTRep * f2, TriSet * ts, TriRevInvSet * tris, MONTP_OPT2_AS_GENE * pPtr)
{
    preFFTRep res;
    KroFFTRep kro;

    sfixn d1,d2,d3,d4,d5,sz1, sz2;
    d1=shrinkDeg(BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    d2=shrinkDeg(BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
    //sz1=(d1+1)*CUMI(f1, N);
    //sz2=(d2+1)*CUMI(f2, N);

    sz1=getDenseSiz(N, f1, BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    sz2=getDenseSiz(N, f2, BUSZSI(f2, N), DAT(f2), CUMI(f2, N));

    d3=BUSZSI(f1, N);
    d4=BUSZSI(f2, N);
    InitResPoly(&res, N,  BUSZS(f1), BUSZS(f2));

    if((sz1>=MulCut)||(sz2>=MulCut)){
        d5=BUSZSDI(res, N);
        BUSZSI(f1, N)=d1;
        BUSZSI(f2, N)=d2;
        BUSZSDI(res, N)=d1+d2;
        InitKroFFTRep(&kro, BUSZSD(res), N, 2, pPtr); 
        //================================================================
        mulPoly_FFT(N, &kro, &res, f1, f2,  pPtr);
        //================================================================
        BUSZSI(f1, N)=d3;
        BUSZSI(f2, N)=d4;
        BUSZSDI(res, N)=d5;
        freeKroFFTRep(&kro);}
    else{
        if(N==1){
            EX_Mont_PlainMul_OPT2_AS_GENE(BUSZSDI(res, 1), res.data, d1, DAT(f1), d2, DAT(f2), pPtr); 
        } else{
            plainMultiDMul(N, res.accum, res.data, CUM(f1), BUSZS(f1), CUM(f2), BUSZS(f2), DAT(f1), DAT(f2), pPtr); }
    }
    //MultiMod(N, out, &res, ts, tris, pPtr);

    reduceCoeffs(N, out, &res, ts, tris, pPtr);

    freePoly(&res);
    return out;
}



/**
 * MultiCoefPolyMul_1:
 * @N: the number of variables.
 * @co: a coefficient of 'f1', where 'f1' has the same main variable with 'f2'.
 * @f2: (output) a C-Cube polynomial.
 * @ts: the triangular set.
 * @tris: the modular inverses of 'ts'.
 * @pPtr: the information of the prime.
 * 
 * Return value: f2 = co * f2.
 **/
preFFTRep * MultiCoefPolyMul_1(sfixn N, preFFTRep * co, preFFTRep * f2, 
        TriSet * ts, TriRevInvSet * tris, MONTP_OPT2_AS_GENE * pPtr)
{
    register sfixn i;
    sfixn d2;
    preFFTRep res;

    //printf("------------->2.0.0\n");
    //fflush(stdout);

    backupData(f2);
    //printf("------------->2.0.1\n");
    //fflush(stdout);

    d2=shrinkDeg(BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
    decreaseOneDim(f2); 
    InitResPoly(&res, N-1,  BUSZS(f2), BUSZS(co));


    //printf("------------->2.0.2\n");
    //fflush(stdout);

    for(i=0; i<=d2; i++){
        //printf("------------->i=%ld\n", i);
        //fflush(stdout);

        PolyCleanData(&res);
        mul_Reduced_1(&res, N-1, f2, co, ts, tris, pPtr, 1);
        nextCoefData(f2, N);

    }
    freePoly(&res);
    increaseOneDim(f2);
    restoreData(f2);
    return f2;
}

    static preFFTRep * 
MultiCoefPolyMulMonicize_1(sfixn N, preFFTRep * co, preFFTRep * f2, TriSet * ts, 
        TriRevInvSet * tris, MONTP_OPT2_AS_GENE * pPtr)
{
    MultiCoefPolyMul_1(N, co, f2, ts, tris, pPtr);
    setLeadingCoefMultiOne(N,f2);
    return f2;
}

//====================================================================
//====================================================================
//     Reduce Coefficients.
//====================================================================
//===================================================================

/**
 * reduceCoeffs:
 * @N: number of variables.
 * @toPtr: (output) 'fromPtr' modulo 'ts'.
 * @fromPtr: a C-Cube polynomial.
 * @ts: a triangular set.
 * @tris: modular inverses of 'ts'.
 * @pPtr: information of the prime.
 * 
 * 'ts' has main variable N-1. 'fromPtr' has main variable N.
 * Reduce the coefficients of 'fromPtr' by 'ts'. 
 *
 * Return value: 
 **/
void reduceCoeffs(sfixn N, preFFTRep * toPtr, preFFTRep * fromPtr, 
        TriSet * ts, TriRevInvSet * tris,   MONTP_OPT2_AS_GENE * pPtr)
{ register int i;

    backupData(fromPtr);
    backupData(toPtr);
    decreaseOneDim(fromPtr);
    decreaseOneDim(toPtr);
    for(i=0; i<=BUSZSI(toPtr, N); i++){
        MultiMod(N-1, toPtr, fromPtr, ts, tris, pPtr);
        nextCoefData(fromPtr,N);
        nextCoefData(toPtr,N); 
    }
    increaseOneDim(fromPtr);
    increaseOneDim(toPtr);
    restoreData(fromPtr);
    restoreData(toPtr);
}

// x_N is mainvariable of fromPtr.
// reduce fromPtr's coefficient's
/**
 * Ex_ReduceCoeffs:
 * @N: the number of variables.
 * @fromPtr: a Cube polynomial.
 * @ts: a triangular set.
 * @tris: modular inverse of 'ts'.
 * 
 * To reduce the coefficients of 'fromPtr' by 'ts'.
 * Namely, 'fromPtr' has main variable of X_{N-1}
 * And 'ts' has one less variable.
 * Return value:  'fromPtr' modulo 'ts'. 
 **/
// from Ptr will be changed
preFFTRep * Ex_ReduceCoeffs(sfixn N, preFFTRep *fromPtr, TriSet *ts,  
        TriRevInvSet *tris, MONTP_OPT2_AS_GENE *pPtr)
{
    preFFTRep *toPtr;
    sfixn *dgs;
    int i;
    dgs=(sfixn *) my_calloc(N+1, sizeof(sfixn));
    for(i=1; i<N;i++){dgs[i]=BDSI(ts, i);}
    dgs[N] = BUSZSI(fromPtr, N);
    toPtr = EX_InitOnePoly(N, dgs);
    reduceCoeffs(N, toPtr, fromPtr, ts, tris, pPtr);
    my_free(dgs);
    return toPtr;
}



// fromPtr will not be changed
preFFTRep * EX_ReduceCoeffs(sfixn N, preFFTRep *fromPtr, TriSet *ts,
        TriRevInvSet *tris, MONTP_OPT2_AS_GENE *pPtr)
{
    preFFTRep *toPtr, *tmpPtr;
    sfixn *dgs;
    int i;
    dgs= (sfixn *) my_calloc(N+1, sizeof(sfixn));
    for(i=1; i<N;i++){dgs[i]=BDSI(ts, i);}
    dgs[N] = BUSZSI(fromPtr, N);
    toPtr = EX_InitOnePoly(N, dgs);
    tmpPtr=EX_CopyOnePoly(fromPtr);
    reduceCoeffs(N, toPtr, tmpPtr, ts, tris, pPtr);
    my_free(dgs);
    EX_FreeOnePoly(tmpPtr);
    return toPtr;
}

//====================================================================
//====================================================================
//     Newton Reverse Inverse.
//====================================================================
//====================================================================

// output tRIPt.
/**
 * NewtonRevInverse:
 * @N: number of variables of 'ts'.
 * @tRIPtrtmp: temporary buffer.
 * @tRIPtr:(output) the inverse of T_N modulo {T_1,...,T_{N-1}}.
 * @tPtr: a triangular set {T_1,...,T_N}.
 * @tris: the inverses of 'tPtr'.
 * @pPtr: the information of prime p.
 *
 * Compute the inverse of T_N modulo 'tPtr'.
 *
 * Return value: 
 **/
void NewtonRevInverse(sfixn N, preFFTRep * tRIPtrtmp, preFFTRep * tRIPtr, 
        preFFTRep * tPtr, TriSet * ts, TriRevInvSet * tris,  MONTP_OPT2_AS_GENE * pPtr)
{
    register int i; 
    sfixn e,n;
    sfixn * tmpPtr;
    sfixn cutDeg;
    sfixn degF;
    sfixn d1, d2, d3, d4, d5;
    KroFFTRep * krofftrep, * krofftrep2;
    preFFTRep * resPoly, * resPoly2;

    backupData(tPtr);
    if(EXSTI(tris, N)){ //if(DEBUG) printf("leaving NewtonRevInverse.\n"); 
        return;}

    if(N==1){
        n=(BUSZSI(tRIPtr, 1))+1;
        e=logceiling(n);
        tmpPtr=(sfixn *)my_calloc(((BUSZSI(tPtr, 1))+1),(sizeof(sfixn)));
        reverseUni(BUSZSI(tPtr, 1), tmpPtr, DAT(tPtr));
        degF=BUSZSI(tPtr, 1);
        modularInvPM(BUSZSI(tRIPtr, 1), DAT(tRIPtr), degF, tmpPtr, e, n, pPtr);
        my_free(tmpPtr);
        EXSTI(tris, N)=1;
        return;
    }
    krofftrep=(KroFFTRep *) my_calloc(1,sizeof(KroFFTRep));
    krofftrep2=(KroFFTRep *) my_calloc(1,sizeof(KroFFTRep));
    resPoly=(preFFTRep *) my_calloc(1,sizeof(preFFTRep));
    resPoly2=(preFFTRep *) my_calloc(1,sizeof(preFFTRep));
    n=(BUSZSI(tRIPtr, N))+1;
    e=logceiling(n);
    tmpPtr=(sfixn *)my_calloc(SIZ(tPtr),(sizeof(sfixn)));
    reverseMulti(BUSZSI(tPtr, N), CUMI(tPtr, N), tmpPtr, DAT(tPtr));
    setData(tPtr, tmpPtr);
    DATI(tRIPtr, 0)=1;
    InitResPoly(resPoly, N, BUSZS(tRIPtr) , BUSZS(tRIPtr));
    InitResPoly(resPoly2, N, BUSZS(tPtr) , BUSZS(tRIPtr));
    InitKroFFTRep(krofftrep, BUSZS(resPoly), N, 2, pPtr);
    InitKroFFTRep(krofftrep2, BUSZS(resPoly2), N, 2, pPtr);

    //=====================
    // Newton Iteration  ==
    //=====================

    d1=BUSZSI(tRIPtr, N);
    d2=BUSZSI(tPtr, N);
    d3=BUSZSI(resPoly, N);
    d4=BUSZSI(tRIPtrtmp, N);
    d5=BUSZSI(resPoly2, N);

    for (i=1; i<=e; i++){
        cutDeg=(1<<i)-1;


        if(d1>cutDeg) BUSZSI(tRIPtr, N)=cutDeg; else BUSZSI(tRIPtr, N)=d1;
        if(d2>cutDeg) BUSZSI(tPtr, N)=cutDeg; else BUSZSI(tPtr, N)=d2;
        if(i>1) PolyCleanData(resPoly);

        if(d3>(2*cutDeg)) BUSZSI(resPoly, N)=2*cutDeg; else BUSZSI(resPoly, N)=d3;

        decreaseKroFFTRep(krofftrep, BUSZS(resPoly));

        //=========================================================
        squarePoly_FFT(N, krofftrep, resPoly, tRIPtr, pPtr);
        // =============================================================

        PolyCleanData(tRIPtrtmp);

        if(d4> cutDeg) BUSZSI(tRIPtrtmp, N)=cutDeg; else BUSZSI(tRIPtrtmp, N)=d4;
        if(d3>cutDeg) BUSZSI(resPoly, N)=cutDeg; else BUSZSI(resPoly, N)=d3;

        reduceCoeffs(N, tRIPtrtmp, resPoly, ts, tris, pPtr);

        if(i>1)  KroneckerCleanData(krofftrep2);
        if(i>1) PolyCleanData(resPoly2);
        if(d5>2*cutDeg) BUSZSI(resPoly2, N)=2*cutDeg; else BUSZSI(resPoly2, N)=d5;
        decreaseKroFFTRep(krofftrep2, BUSZS(resPoly2));

        //====================================================================
        mulPoly_FFT(N, krofftrep2, resPoly2, tRIPtrtmp, tPtr, pPtr);
        //==================================================================== 

        if(d5>cutDeg) BUSZSI(resPoly2, N)=cutDeg; else BUSZSI(resPoly2, N)=d5;
        BUSZSI(tRIPtrtmp, N)=d4;
        PolyCleanData(tRIPtrtmp);
        BUSZSI(tRIPtrtmp, N)=cutDeg;

        reduceCoeffs(N, tRIPtrtmp, resPoly2, ts, tris, pPtr);

        //============================================
        addEqDgPoly_1(N, tRIPtr, tRIPtr, pPtr->P);
        //============================================
        //=================================================
        subEqDgPoly_1(N, tRIPtr, tRIPtrtmp, pPtr->P, 1);
        //=============================================

    }

    // actually free tmpPtr -- the reversal of tPtr->data.
    my_free(DAT(tPtr));
    restoreData(tPtr);


    BUSZSI(tRIPtr, N)=d1;
    BUSZSI(tPtr, N)=d2;
    BUSZSI(resPoly, N)=d3;
    BUSZSI(tRIPtrtmp, N)=d4;
    BUSZSI(resPoly2, N)=d5;

    freePoly(resPoly);  
    freePoly(resPoly2); 
    freeKroFFTRep(krofftrep);
    freeKroFFTRep(krofftrep2);
    my_free(krofftrep);
    my_free(krofftrep2);
    my_free(resPoly);
    my_free(resPoly2);
    EXSTI(tris, N)=1; 
    //
    return;
}




//============================================================
//
//
//
//  Univariate polynomial Fast Mod.
//
//
//=============================================================

// suppose the RevInvPtr is known. or Say the first poly's RevInv in TriSet is computed.
//destructive APtr into RPtr.

/**
 * UniFastMod_1:
 * @degA: the degree of univariate polynomial 'A'.
 * @APtr: the coefficient vector of polynomial 'A'.
 * @degB: the degree of univariate polynomial 'B'.
 * @BPtr: the coefficient vector of polynomial 'B'.
 * @Lbound: The next power of 2 of (degA-degB) .
 * @BRevInvPtr: the modular inverse of 'B'.
 * @pPtr: the information of prime number p.
 * 
 * Computer the remainder of A is divied by B.
 *
 * Return value: 
 **/
static void UniFastMod_1(sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, sfixn Lbound, 
        sfixn * BRevInvPtr,  MONTP_OPT2_AS_GENE * pPtr)
{
    register sfixn i, j;

    sfixn * FPtr,  *QPtr, * GPtr;
    sfixn degF, power2,power3, n2,n3,l1,l2,l3, tmp, sz;
    sfixn dg, da;
    sfixn degQ=degA-degB;


    degF=degQ;

    if(degF<0) {
        return;}

    l1=degF+1;

    n2=1; power2=0; l2=(l1<<1)-1;tmp=l2;
    while(tmp){tmp>>=1; n2<<=1; power2++;}
    n3=1; power3=0;  l3=degB+degQ+1, tmp=l3;
    while(tmp){tmp>>=1; n3<<=1; power3++;}

    dg=da=degF;

    sz=n2;
    if(n3>n2)sz=n3;

    FPtr=(sfixn * )my_calloc(sz ,sizeof(sfixn));
    GPtr=(sfixn * )my_calloc(sz ,sizeof(sfixn));


    for(i=0;i<=dg; i++) GPtr[i]=BRevInvPtr[i];

    FPtr=reverseUni(degB, FPtr, BPtr);

    FPtr=reverseUni(degA, FPtr, APtr);

    EX_Mont_FFTMul_OPT2_AS_GENE_1(n2, power2, da+dg, da, FPtr, dg, GPtr, pPtr);

    QPtr=(sfixn * ) my_calloc(degQ+1, sizeof(sfixn));

    QPtr=reverseUni(degQ, QPtr, FPtr);

    cleanVec(n3-1, FPtr);


    EX_Mont_FFTMul_OPT2_AS_GENE(n3, power3, degQ+degB, FPtr, degQ, QPtr, degB, BPtr, pPtr);



    for(j=0; j<l3; j++) APtr[j]=SubMod(APtr[j], FPtr[j],pPtr->P);
    for(j=l3;j<=degA;j++) APtr[j]=0;

    my_free(FPtr);
    my_free(GPtr);
    my_free(QPtr);

} 

//============================================================
//  Multiavriate (as univariate) polynomial Fast Mod.
//=============================================================
// Modulo poly by T_n, then by [T_{n-1}, T_{n-2}, ...,T_1] 
/**
 * MultiUniFastMod_1:
 * @N: number of variables.
 * @tmpIn: a temp buffer.
 * @n: The next power of 2 of (deg(inPtr, X_N)-deg(T_N, X_N)+1).
 * @inPtr:(output) a C-Cube polynomial.
 * @ts: a triangular set.
 * @tris: modular inverses of 'ts'.
 * @pPtr: the information of the prime p.
 * 
 * Return value: 
 **/
static void MultiUniFastMod_1(sfixn N, preFFTRep * tmpIn,  sfixn n, preFFTRep * inPtr, 
        TriSet * ts, TriRevInvSet * tris,  MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn m,e,d1,d2;
    sfixn cutDg;

    preFFTRep * resPoly, * resPoly2, * out;
    KroFFTRep * krofftrep, * krofftrep2;

    //
    m=BUSZSI((ELEMI(ts, N)), N);

    if(n<m){ // if(DEBUG) printf("Leaving MultiUniFastMod_1.\n"); 
        return;}

    resPoly=(preFFTRep *)my_calloc(1,sizeof(preFFTRep));
    resPoly2=(preFFTRep *)my_calloc(1,sizeof(preFFTRep));
    out=(preFFTRep *)my_calloc(1,sizeof(preFFTRep));

    krofftrep=(KroFFTRep *)my_calloc(1,sizeof(KroFFTRep));
    krofftrep2=(KroFFTRep *)my_calloc(1,sizeof(KroFFTRep));
    e=logceiling(n-m+1);
    if(! (EXSTI(tris, N))){  
        initCopyOneRevInv(out, ELEMI(tris, N));
        NewtonRevInverse(N, out, ELEMI(tris, N), ELEMI(ts, N), ts, tris, pPtr );
        freePoly(out);}
        cutDg=n-m;
        d1=BUSZSI(tmpIn, N);
        d2=BUSZSI(ELEMI(tris, N), N);
        BUSZSI(tmpIn, N)=cutDg;
        BUSZSI(ELEMI(tris, N), N)=cutDg;
        reverseMulti_1(n, CUMI(tmpIn, N), DAT(tmpIn));
        InitResPoly(resPoly, N, BUSZS(tmpIn), BUSZS(ELEMI(tris, N)));



        EX_mulPoly(N, resPoly, tmpIn, ELEMI(tris, N), pPtr);

        /*   InitKroFFTRep(krofftrep, BUSZS(resPoly), N, 2, pPtr); */
        /*   //===================================================================== */
        /*   mulPoly_FFT(N, krofftrep, resPoly, tmpIn, ELEMI(tris, N), pPtr); */
        /*   //============================================================ */
        /*   freeKroFFTRep(krofftrep); */



        reverseMulti_1(cutDg, CUMI(resPoly, N), DAT(resPoly));

        BUSZSI(tmpIn, N)=d1;
        BUSZSI(ELEMI(tris, N), N)=d2;

        PolyCleanData(tmpIn);

        BUSZSI(tmpIn, N)=cutDg;

        reduceCoeffs(N, tmpIn, resPoly, ts, tris, pPtr);

        freePoly(resPoly); 



        InitResPoly(resPoly2, N,  BUSZS(ELEMI(ts, N)), BUSZS(tmpIn));




        EX_mulPoly(N, resPoly2, ELEMI(ts, N), tmpIn,  pPtr);

        /*   InitKroFFTRep(krofftrep2, BUSZS(resPoly2), N, 2, pPtr);  */
        /*   //================================================================ */
        /*   mulPoly_FFT(N, krofftrep2, resPoly2, ELEMI(ts, N), tmpIn,  pPtr); */
        /*   //================================================================ */
        /*   freeKroFFTRep(krofftrep2); */



        BUSZSI(tmpIn, N)=d1;

        PolyCleanData(tmpIn);

        reduceCoeffs(N, tmpIn, resPoly2, ts, tris, pPtr);

        freePoly(resPoly2);

        subPoly_1(N, inPtr, tmpIn, pPtr->P);

        my_free(resPoly); 
        my_free(resPoly2);
        my_free(out);
        my_free(krofftrep); 
        my_free(krofftrep2);
}

//============================================================
//  Multiavriate polynomial Fast Mod. (recursive.)
//=============================================================
/**
 * MultiMod_1:
 * @N: number of variables.
 * @inPtr: a C-Cube polynomial.
 * @ts: a zero-dim trinagular set.
 * @tris: inverses of 'ts'.
 * @pPtr: the information of prime number p.
 * 
 * Reduce 'inPtr' by 'ts' in-place.
 * I.e. 'inPtr'= 'inPtr' modulo 'ts'.
 * Return value: 
 **/
void MultiMod_1(sfixn N, preFFTRep * inPtr, TriSet * ts, 
        TriRevInvSet * tris,  MONTP_OPT2_AS_GENE * pPtr)
{
    register int i;
    int deg=0, d;

    sfixn * restoreDat;
    sfixn restoreSiz;
    preFFTRep  tmpIn;

    signal(SIGINT,catch_intr);

    if(N==1){ 

        if(Interrupted==1) { return; }

        deg=shrinkDeg(BUSZSI(inPtr, N), DAT(inPtr), 1);

        if (!deg) return;
        if (((BDSI(ts, 1))>DivCut1)) {
            UniFastMod_1(deg, DAT(inPtr), BUSZSI((ELEMI(ts, 1)), 1), DAT(ELEMI(ts, 1)), 
                    (NLB(tris))[1], DAT(ELEMI(tris, 1)) ,pPtr);  }
        else{
            UniPlainMod_1(deg, DAT(inPtr), BUSZSI((ELEMI(ts, 1)), 1), 
                    DAT(ELEMI(ts, 1)), pPtr); }
        return;
    }



    deg=shrinkDeg(BUSZSI(inPtr, N), DAT(inPtr), CUMI(inPtr, N));
    restoreDat=DAT(inPtr);
    N(inPtr)=N-1;
    restoreSiz=SIZ(inPtr);  
    SIZ(inPtr)=CUMI(inPtr, N);

    for(i=0; i<=deg; i++){
        MultiMod_1(N-1, inPtr, ts, tris, pPtr);
        DAT(inPtr)+=CUMI(inPtr, N); 
    }
    N(inPtr)=N;
    DAT(inPtr)=restoreDat;
    SIZ(inPtr)=restoreSiz;
    d=BDSI(ts, N);


    copyVec_1_to_n(N-1, CUTS(inPtr), BDS(ts));
    deg=shrinkDeg(CUTSI(inPtr, N), DAT(inPtr), CUMI(inPtr, N));
    copyVec_1_to_n(N-1, CUTS(inPtr), BUSZS(inPtr));


    BDSI(ts, N)=deg;
    InitOneReducedPoly(&tmpIn, N, BDS(ts));  
    fromtofftRep(N, tmpIn.accum, tmpIn.data, CUM(inPtr),  BUSZSD(tmpIn), DAT(inPtr));
    BDSI(ts, N)=d;



    if((((N==2)&&(d>DivCut2)) || ((N==3)&&(d>DivCut3)) || ((N>3)&&(d>DivCutN)))){
        //printf("using MultiUniFastMod_1\n");
        //fflush(stdout);

        MultiUniFastMod_1(N, &tmpIn, deg, inPtr, ts, tris, pPtr);

    }
    else{
        //printf("using MultiUniPlainMod_1\n");
        //fflush(stdout);



        MultiUniPlainMod_1(N, deg, &tmpIn, BDSI(ts, N)+1,  ELEMI(ts, N), ts, tris, pPtr);
        fromtofftRep(N, CUM(inPtr), DAT(inPtr), tmpIn.accum,  BUSZSD(tmpIn), tmpIn.data);

    }

    freePoly(&tmpIn);       
}



//========================================================================
//   Multiavriate polynomial Fast Mod. (recursive.)
//   using FFT.
// 
// will destruct inPtr.
// N is the # of current dimensions.
// outPtr keeps the output of the reduction.
// -- The output is bounded by ts->bounds (inclusively).
// inPtr keeps the input to be reduced.
// -- The input is required bounded by 2 times ts->bounds (inslucively).
// ts is the triangular set.
// tris is the inverses of reverse-coef-ordered ts. 
//========================================================================

/**
 * MultiMod_DF:
 * @N: number of variables.
 * @outPtr: (output) the image of 'inPtr' modulo 'ts'.
 * @inPtr: a C-Cube polynomial.
 * @ts: a triangular set.
 * @tris: modular inverse of 'ts'.
 * @pPtr: the information of prime number p.
 * 
 * Reduce 'inPtr' w.r.t 'ts' and save the result in 'outPtr'.
 *
 * Return value: 
 **/
void MultiMod_DF(sfixn N, preFFTRep * outPtr, preFFTRep * inPtr, TriSet * ts, 
        TriRevInvSet * tris,  MONTP_OPT2_AS_GENE * pPtr)
{ 
    if (zeroCoefp(DAT(inPtr), SIZ(inPtr)) ) return;
    MultiMod_1(N, inPtr, ts, tris, pPtr);
    fromtofftRep(N, CUM(outPtr), DAT(outPtr), CUM(inPtr),  BUSZS(outPtr), DAT(inPtr));
}

//============================================================
//  Multiavriate polynomial Fast Mod. (recursive.)
//=============================================================
void MultiMod_1_DF_CLS(sfixn N, preFFTRep * inPtr, TriSet * ts, 
        TriRevInvSet * tris,  MONTP_OPT2_AS_GENE * pPtr)
{
    register int i;
    int deg=0, d;

    sfixn * restoreDat;
    sfixn restoreSiz;
    preFFTRep  tmpIn;


    signal(SIGINT,catch_intr);

    if(N==1){

        if(Interrupted==1) { return; }

        deg=shrinkDeg(BUSZSI(inPtr, N), DAT(inPtr), 1);

        if (!deg) return;

        UniPlainMod_1(deg, DAT(inPtr), BUSZSI((ELEMI(ts, 1)), 1), 
                DAT(ELEMI(ts, 1)), pPtr); 
        return;
    }




    deg=shrinkDeg(BUSZSI(inPtr, N), DAT(inPtr), CUMI(inPtr, N));
    restoreDat=DAT(inPtr);
    N(inPtr)=N-1;
    restoreSiz=SIZ(inPtr);  
    SIZ(inPtr)=CUMI(inPtr, N);

    for(i=0; i<=deg; i++){
        MultiMod_1_DF_CLS(N-1, inPtr, ts, tris, pPtr);
        DAT(inPtr)+=CUMI(inPtr, N); 
    }
    N(inPtr)=N;
    DAT(inPtr)=restoreDat;
    SIZ(inPtr)=restoreSiz;
    d=BDSI(ts, N);


    copyVec_1_to_n(N-1, CUTS(inPtr), BDS(ts));
    deg=shrinkDeg(CUTSI(inPtr, N), DAT(inPtr), CUMI(inPtr, N));
    copyVec_1_to_n(N-1, CUTS(inPtr), BUSZS(inPtr));


    BDSI(ts, N)=deg;
    InitOneReducedPoly(&tmpIn, N, BDS(ts));  
    fromtofftRep(N, tmpIn.accum, tmpIn.data, CUM(inPtr),  BUSZSD(tmpIn), DAT(inPtr));
    BDSI(ts, N)=d;




    MultiUniPlainMod_1(N, deg, &tmpIn, BDSI(ts, N)+1,  ELEMI(ts, N), ts, tris, pPtr);
    fromtofftRep(N, CUM(inPtr), DAT(inPtr), tmpIn.accum,  BUSZSD(tmpIn), tmpIn.data);


    freePoly(&tmpIn);       
}



//========================================================================
//   Multiavriate polynomial Fast Mod. (recursive.)
//   using FFT.
// 
// will destruct inPtr.
// N is the # of current dimensions.
// outPtr keeps the output of the reduction.
// -- The output is bounded by ts->bounds (inclusively).
// inPtr keeps the input to be reduced.
// -- The input is required bounded by 2 times ts->bounds (inslucively).
// ts is the triangular set.
// tris is the inverses of reverse-coef-ordered ts. 
//========================================================================
    void
MultiMod_DF_CLS(sfixn N, preFFTRep * outPtr, preFFTRep * inPtr, TriSet * ts, 
        TriRevInvSet * tris,  MONTP_OPT2_AS_GENE * pPtr)
{ 
    if (zeroCoefp(DAT(inPtr), SIZ(inPtr)) ) return;
    MultiMod_1_DF_CLS(N, inPtr, ts, tris, pPtr);
    fromtofftRep(N, CUM(outPtr), DAT(outPtr), CUM(inPtr),  BUSZS(outPtr), DAT(inPtr));
}

//========================================================================
//  f1 = f1 - co * X**e * f2
// 
//  
//========================================================================
    sfixn
fmecgEqDg_1(sfixn N, preFFTRep * f1, sfixn e, preFFTRep * co, preFFTRep * f2, 
        TriSet * ts, TriRevInvSet * tris, MONTP_OPT2_AS_GENE * pPtr)
{
    register sfixn i;
    sfixn d1,d2,d;
    preFFTRep out, res;
    InitOneReducedPoly(&out, N-1, BUSZS(f1)); 
    d1=shrinkDeg(BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
    d2=d1-e;
    backupData(f1);
    backupData(f2);
    decreaseOneDim(f1);  
    decreaseOneDim(f2);
    nextMCoefData(f1,N,e);  
    InitResPoly(&res, N-1,  BUSZS(co), BUSZS(f2));
    for(i=0; i<=d2; i++){
        PolyCleanData(&out);
        PolyCleanData(&res);


        mul_Reduced(&res, N-1, &out, co, f2, ts, tris, pPtr);

        // printf("f1=");
        //printPoly(f1);
        //printf("&out=");
        //printPoly(&out);

        subEqDgPoly_1(N-1, f1, &out, pPtr->P, 1);
        nextCoefData(f1, N);   
        nextCoefData(f2, N);
    }
    freePoly(&res);
    increaseOneDim(f1);
    increaseOneDim(f2);
    restoreData(f1);
    restoreData(f2);
    freePoly(&out);

    //printf("BUSZSI(f1, N)=%ld\n", BUSZSI(f1, N));
    //fflush(stdout);

    d=shrinkDeg(BUSZSI(f1, N), DAT(f1), CUMI(f1, N));

    //printf("d=%ld\n", d);
    //fflush(stdout);


    //printf("f1=");
    //printPoly(f1);

    //printf("(zeroCoefp(DAT(f1), CUMI(f1,N)))=%ld.\n", (zeroCoefp(DAT(f1), CUMI(f1,N))));
    //fflush(stdout);

    if((zeroCoefp(DAT(f1), CUMI(f1,N)))&& (d==0)) return -1; else return d;
}





//========================================================================
// FPtr = QPtr * GPtr + rPtr   
// ouput: rPtr (FPtr), QPtr
//========================================================================

/**
 * MultiUniPlainDivide_1:
 * @N: number of variables.
 * @QPtr: (output) save the quotient of 'FPtr' divided by 'GPtr'.
 * @FPtr: a C-Cube polynomial, the divident.
 * @GPtr: a C-Cube polynomial, the divisor.
 * @ts: a triangular set.
 * @tris: modular inverses of 'ts'.
 * @pPtr: the information of prime p.
 * 
 * In formula FPtr = QPtr * GPtr + rPtr,
 * 'FPtr' and 'GPtr' are input.
 * 'QPtr' is the output quotient. 
 * 'rPtr' is the output remainder. And 'rPtr' uses 'FPtr' as the in-place buffer.
 * Return value: the degree of the 'QPtr'.
 **/
    sfixn
MultiUniPlainDivide_1(sfixn N, preFFTRep * QPtr, preFFTRep * FPtr, preFFTRep * GPtr, 
        TriSet * ts, TriRevInvSet * tris, MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn d1, d2, d, a, b, dq;

    preFFTRep co;
    PolyCleanData(QPtr);
    d1=shrinkDeg(BUSZSI(FPtr, N), DAT(FPtr), CUMI(FPtr, N));
    d2=shrinkDeg(BUSZSI(GPtr, N), DAT(GPtr), CUMI(GPtr, N));

    // printf("N=%ld.\n", N);

    dq=d=d1-d2;
    if(N==0){
        a=DATI(FPtr, 0);
        b=DATI(GPtr, 0);
        DATI(QPtr, 0)=MulMod(inverseMod(b, pPtr->P),a,pPtr->P);
        DATI(FPtr, 0)=0;
        return dq;
    }


    if(N==1){
        if(d2>DivCut1){
            fastDiv_1(d, DAT(QPtr), d1, DAT(FPtr), d2, DAT(GPtr), DAT(ELEMI(tris, 1)) ,pPtr);

        }
        else{
            plainDivMonic_1(d, DAT(QPtr), d1, DAT(FPtr), d2, DAT(GPtr), pPtr);
        }

    }
    else{
        InitOneReducedPoly(&co, N-1, BUSZS(FPtr));

        while(d>=0){
            PolyCleanData(&co);
            getCoefMulti(N, &co, FPtr, d1);
            setCoefMulti(N, QPtr, &co, d);
            d1=fmecgEqDg_1(N, FPtr, d, &co, GPtr, ts, tris, pPtr);
            if(d1==-1) {
                freePoly(&co);
                return dq;}
                d=d1-d2;
        }

        freePoly(&co);
    }

    return dq;
}


//=================================================================
// FPtr = QPtr * GPtr + rPtr   
// ouput: rPtr (FPtr)
// N is 1 plus #TriSet
// cuts1=1 means fast;
// cuts1=0 means plain;
//=================================================================

/**
 * MultiUniPlainDivide_1:
 * @N: number of variables.
 * @d1: degree of 'FPtr'.
 * @FPtr: a C-Cube polynomial, the divident.
 * @d2: degree of 'GPtr'.
 * @GPtr: a C-Cube polynomial, the divisor.
 * @ts: a triangular set.
 * @tris: modular inverses of 'ts'.
 * @pPtr: the information of prime p.
 * 
 * In formula FPtr = QPtr * GPtr + rPtr,
 * 'FPtr' and 'GPtr' are input. 
 * 'rPtr' is the output remainder. And 'rPtr' uses 'FPtr' as the in-place buffer.
 * Return value: the degree of the 'QPtr'.
 **/
static preFFTRep * MultiUniPlainMod_1(sfixn N, sfixn d1, preFFTRep * FPtr, 
        sfixn d2, preFFTRep * GPtr, TriSet * ts, TriRevInvSet * tris, 
        MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn d, a, b;

    preFFTRep co;

    d1=shrinkDeg(BUSZSI(FPtr, N), DAT(FPtr), CUMI(FPtr, N));
    d2=shrinkDeg(BUSZSI(GPtr, N), DAT(GPtr), CUMI(GPtr, N));

    d=d1-d2;
    if(d<0) return FPtr;
    if(N==0){
        a=DATI(FPtr, 0);
        b=DATI(GPtr, 0);
        DATI(FPtr, 0)=0;
        return FPtr;
    }

    if(N==1){
        if((BDSI(ts, 1))>DivCut1){ 
            UniFastMod_1(d1, DAT(FPtr), d2, DAT(GPtr), 
                    (NLB(tris))[1], DAT(ELEMI(tris, 1)) ,pPtr);}
        else{
            UniPlainMod_1(d1, DAT(FPtr), d2, DAT(GPtr), pPtr);}
    }
    else{

        InitOneReducedPoly(&co, N-1, BUSZS(FPtr));
        while(d>=0){
            PolyCleanData(&co);
            getCoefMulti(N, &co, FPtr, d1);
            d1=fmecgEqDg_1(N, FPtr, d, &co, GPtr, ts, tris, pPtr);


            if(d1==-1) {freePoly(&co); return FPtr;}
            d=d1-d2;
        }


        freePoly(&co);
    } 
    return FPtr;

}

//=======================================================================
// Extended Euclidean algorithm.
//
// return -1 means failed.
// return 0 means good. 
// suppose fPtr and gPtr have same number of variables and 
// deg(fPtr)>=deg(gPtr);
//======================================================================
/**
 * MultiExEuclidean:
 * @N: number of variables.
 * @gcdPtr: the GCD of 'fPtr' and 'gPtr', C-Cube polynomial.
 * @uPtr: bezout coefficient, C-Cube polynomial.
 * @vPtr: bezout coefficient, C-Cube polynomial.
 * @fPtr: a C-Cube polynomial.
 * @gPtr: a C-Cube polynomial.
 * @ts: a trinagular set.
 * @tris: modular inverses of 'ts'.
 * @pPtr: the information of prime number p.
 * 
 *
 * In the formula  " uPtr*fPtr + vPtr*gPtr = gcdPtr ".
 * 'fPtr' and 'gPtr' are the input.
 * 'uPtr', 'vPtr', and 'gcdPtr' are the output. 
 *
 * Return value: -1 for failed, 0 for having a gcd. 
 **/
    sfixn
MultiExEuclidean(sfixn N, preFFTRep * gcdPtr, preFFTRep * uPtr, 
        preFFTRep * vPtr,  preFFTRep * fPtr, preFFTRep * gPtr, 
        TriSet * ts, TriRevInvSet * tris, MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn dC=0, dD=0, dG=0, dA, dB, i, invG, p=pPtr->P;
    sfixn reci, dtmp;
    preFFTRep v1, v2, tmp, co, w1, w2, Q, res;

    signal(SIGINT,catch_intr);
    if(Interrupted==1) { return -1; }
    if(N==1){
        dA=shrinkDeg(BUSZSI(fPtr, 1), DAT(fPtr), 1);
        dB=shrinkDeg(BUSZSI(gPtr, 1), DAT(gPtr), 1);
        PolyCleanData(vPtr);
        //if ((dA>UniGcdCut) && (dB>UniGcdCut)){
        if(0){
            printf("crash on some large example.");
            fflush(stdout);
            Throw 11;
            XGCD(DAT(uPtr), &dC, DAT(vPtr), &dD, DAT(gcdPtr), &dG, DAT(fPtr), 
                    dA, DAT(gPtr), dB, pPtr);
            invG=inverseMod(DATI(gcdPtr, dG), p);
            for(i=0; i<=dC; i++) DATI(uPtr, i)=MulMod(DATI(uPtr, i),invG,p);
            for(i=0; i<=dD; i++) DATI(vPtr, i)=MulMod(DATI(vPtr, i),invG,p);
            for(i=0; i<=dG; i++) DATI(gcdPtr, i)=MulMod(DATI(gcdPtr, i),invG,p);
            return 0;
        } else  {
            ExGcd_Uni(DAT(uPtr), &dC, DAT(vPtr), &dD, DAT(gcdPtr), &dG, DAT(fPtr), 
                    dA, DAT(gPtr), dB, pPtr);
            return 0;
        }
    }

    PolyCleanData(vPtr);
    fromtofftRep(N,  CUM(gcdPtr), DAT(gcdPtr), CUM(fPtr), BUSZS(fPtr), DAT(fPtr));
    BDSI(ts, N)+=1;
    InitOneReducedPoly(&v1, N, BDS(ts));
    BDSI(ts, N)-=1;

    v1.data[0]=1; 
    BDSI(ts, N)+=1;
    InitOneReducedPoly(&v2, N, BDS(ts));
    BDSI(ts, N)-=1;
    fromtofftRep(N,  v2.accum, v2.data, CUM(gPtr), BUSZS(gPtr), DAT(gPtr));
    BDSI(ts, N)+=1;
    BDSI(ts, N-1)+=1;
    InitOneReducedPoly(&tmp, N-1, BDS(ts));
    InitOneReducedPoly(&co, N-1, BDS(ts));
    BDSI(ts, N-1)-=1;
    InitOneReducedPoly(&w1, N, BDS(ts)); // initialized as 0
    InitOneReducedPoly(&w2, N, BDS(ts));
    InitOneReducedPoly(&Q, N, BDS(ts));
    BDSI(ts, N)-=1;

    InitResPoly(&res, N, Q.bufSizs, v1.bufSizs);

    while(1){
        PolyCleanData(&co);       
        getLeadingCoefMulti(N, &co, &v2);
        PolyCleanData(&tmp);

        //printf("Entering MultiRecip()\n");
        reci=MultiRecip(N-1, &tmp, &co, ts, tris, pPtr);
        //printf("Leaving MultiRecip()...%ld\n", reci);     

        if(reci==-1) {
            freePoly(&v1);
            freePoly(&v2);
            freePoly(&tmp);
            freePoly(&co);
            freePoly(&w1);
            freePoly(&w2);
            freePoly(&Q);
            freePoly(&res);
            return -1;
        }

        if(reci>0){
            MultiNumbPolyMul_1(reci, &v1,  pPtr);
            MultiNumbPolyMulMonicize_1(N, reci, &v2,  pPtr);
        }

        if(reci==0){
            MultiCoefPolyMul_1(N, &tmp, &v1, ts, tris, pPtr);
            MultiCoefPolyMulMonicize_1(N, &tmp, &v2, ts, tris, pPtr);
        }

        if ( ! monicPolyp(N, &v2) ) {printf("NOT MONIC!!!\n"); return -1;};

        dtmp=MultiUniPlainDivide_1(N, &Q, gcdPtr, &v2, ts, tris, pPtr);
        CopyOnePoly(&w1, vPtr);
        CopyOnePoly(&w2, gcdPtr);
        CopyOnePoly(vPtr, &v1);
        CopyOnePoly(gcdPtr, &v2);
        PolyCleanData(&res);
        mul_Reduced_1(&res, N, &Q, &v1, ts, tris, pPtr, 2);
        subEqDgPoly_1(N, &w1, &v1, pPtr->P, 2);
        CopyOnePoly(&v2, &w2);
        if(zeroPolyp(&v2)){
            freePoly(&v1);
            freePoly(&v2);
            freePoly(&tmp);
            freePoly(&co);
            freePoly(&w1);
            freePoly(&w2);
            freePoly(&Q);
            freePoly(&res);
            return 0;
        }
    }

    }

    //========================================================
    // Polynomial Inverse.
    //
    // return number >0 means a constant inverse.
    // return number =0 means a non-constant inverse.
    // This N is at cPtr's level !!!
    //========================================================
    // !!!!!!!!!! The buffer of cPtr (the poly to be inverted) MUST BE size of  BDS(ts).
    // !!!!!!!!!! The buffer of invPtr (the output inverse of cPtr) 
    //    MUST BE size of  BDS'(ts).   where BDS'(ts) is the same with BDS(ts) except 
    //    BDSI'(ts,N) = BDSI(ts, N) + 1.

    /**
     * MultiRecip:
     * @N: the number of variables.
     * @invPtr: (output) the inverse 'cPtr' modulo 'ts', a C-Cube polynomial.
     * @cPtr: a C-Cube polynomial.
     * @ts: a triangular set.
     * @tris: the modular inverses of 'ts.
     * @pPtr: the information of prime p.
     * 
     *
     * In the formula "invPtr*cPtr mod ts",
     * 'cPtr' and 'ts' is the input.
     * 'invPtr' is the output. 
     * However, if the inversion failed, the function returns -1.
     * Return value: -1 means failed; 0 means a non-constant inverse containing in 'invPtr',
     *               a positive int means this integer the constant inverse.
     **/
    sfixn MultiRecip(sfixn N, preFFTRep * invPtr, preFFTRep * cPtr, TriSet * ts, 
            TriRevInvSet * tris, MONTP_OPT2_AS_GENE * pPtr) 
    {

        sfixn co, ex, d=0;
        preFFTRep gcd, u;
        //  sfixn *tmpgcd;

        if(zeroPolyp(cPtr)) { return -1; }

        if(onePolyp(cPtr)) return 1;
        co=groundPoly(cPtr);
        if(co>0) return inverseMod(co, pPtr->P);

        BDSI(ts, N)+=1;
        InitOneReducedPoly(&gcd, N, BDS(ts));
        InitOneReducedPoly(&u,   N, BDS(ts));
        BDSI(ts, N)-=1;
        //printf("Entering gcd\n");
        ex=MultiExEuclidean(N, &gcd, &u, invPtr, ELEMI(ts, N), cPtr, ts, tris, pPtr);
        //printf("Leaving gcd\n");
        d=shrinkDeg(BUSZSDI(gcd, N), gcd.data, gcd.accum[N]);

        freePoly(&gcd);
        freePoly(&u);

        if ((ex==-1) || (d>0)){ return -1;}

        return 0;
    }


    // suppose the inversion always exceed.
    // returns 1 means good.
    // returns -1 means failed.
    // an In-place operation.
    /**
     * MonicizePoly_1:
     * @N: the number of variables.
     * @inPoly: a C-Cube polynomial.
     * @ts: a trinagular set.
     * @tris: the modular inverse of 'ts'.
     * @pPtr: the information of prime p.
     *
     * To make 'inPoly' monic. Namely multiply the invers of its leading cofficient. 
     * Ceratinly, all computations are over 'ts'.
     * Return value: -1 means failed, 1 means good. 
     **/
    int MonicizePoly_1(sfixn N, preFFTRep *inPoly, TriSet *ts, 
            TriRevInvSet *tris, MONTP_OPT2_AS_GENE *pPtr)
    {

        preFFTRep co, *cPtr=&co, inv, *invPtr=&inv;
        sfixn reci;

        BDSI(ts, N-1)++;

        //InitOneReducedPoly(cPtr, N-1, BDS(ts));
        InitOneReducedPoly(invPtr, N-1, BDS(ts));

        BDSI(ts, N-1)--;

        InitOneReducedPoly(cPtr, N-1, BDS(ts));

        getLeadingCoefMulti(N, cPtr, inPoly);

        reci = MultiRecip(N-1, invPtr, cPtr, ts, tris, pPtr);

        if(reci==-1) {
            //printf("Not expecting the inversion can fail!\n"); 
            fflush(stdout);
            freePoly(invPtr);
            freePoly(cPtr);
            return -1;}

            if(reci>0){
                MultiNumbPolyMul_1( reci, inPoly, pPtr); 
            }

            if(reci==0){
                MultiCoefPolyMul_1(N, invPtr, inPoly, ts, tris, pPtr); 
            }

            freePoly(invPtr);
            freePoly(cPtr);
            return 1;
    }



    // return 1 means success.
    // return -1 means failed.
    /**
     * MonicizeTriSet_1:
     * @N: number of variables.
     * @ts: trinagular set.
     * @pPtr: the information of prime p.
     * 
     * The input zero-dimensional trinagular set 'ts' may be not a Lazard triangualr set.
     * This function make it monic and reduced, thus becomes a Lazard triangular set.
     * Return value: -1 means failed, 1 means succeeded.
     **/
    int MonicizeTriSet_1(sfixn N, TriSet *ts, MONTP_OPT2_AS_GENE *pPtr){
        TriRevInvSet *tris;
        int i, j, res;
        sfixn d1, inv;
        preFFTRep *t_i, *t_i2;
        sfixn tmp;
        sfixn newDg;
        sfixn *TriSetDgs;


        signal(SIGINT,catch_intr);

        if ( NMLZ(ts) == 1 ) { return 1; }

        d1=BUSZSI(ELEMI(ts,1),1);
        inv=inverseMod((DAT(ELEMI(ts,1)))[d1], pPtr->P);
        MultiNumbPolyMulMonicize_1(1, inv, ELEMI(ts,1), pPtr);

        TriSetDgs=(sfixn *) my_calloc(N+1, sizeof(sfixn));

        for(j=1; j<=N; j++){
            for(i=j;i<=N; i++){
                if( TriSetDgs[j] < BUSZSI(ELEMI(ts, i), j) )
                    TriSetDgs[j]=BUSZSI(ELEMI(ts, i), j) ;
                TriSetDgs[j]<<=1;
            }
        }



        for(i=2; i<=N; i++){

            if(Interrupted==1) {my_free (TriSetDgs); return -1; }

            // --> RevInv00.
            tris = EX_initTriRevInvSet(TriSetDgs, i-1, ts);
            getRevInvTiSet(TriSetDgs, i-1, tris, ts, pPtr);

            tmp = BDSI(ts, i);
            BDSI(ts, i) = BUSZSI(ELEMI(ts, i), i);
            t_i = EX_InitOnePoly(i, BDS(ts));
            BDSI(ts, i) = tmp;

            reduceCoeffs(i, t_i, ELEMI(ts, i), ts, tris, pPtr);

            EX_FreeOnePoly(ELEMI(ts, i));

            newDg=shrinkDeg(BUSZSI(t_i, i), DAT(t_i), CUMI(t_i, i));
            if (newDg == BUSZSI(t_i, i)){
                ELEMI(ts, i) = t_i;
            }
            else{
                BDSI(ts, i) = newDg;
                t_i2 = EX_InitOnePoly(i, BDS(ts));  
                fromtofftRepMultiD(i, CUM(t_i2), DAT(t_i2), CUM(t_i), BUSZS(t_i2), DAT(t_i));
                EX_FreeOnePoly(t_i);
                ELEMI(ts, i) = t_i2;
                BDSI(ts, i)--;
            }

            res=MonicizePoly_1(i, ELEMI(ts, i), ts, tris, pPtr);

            EX_freeTriRevInvSet(tris);
            if(res == -1) return -1; 

        }
        my_free (TriSetDgs);
        NMLZ(ts) = 1;
        return 1;
    }








    //=================================================================
    // Enclidean algorithm. GCD (FPtr, GPtr).
    //
    // Both FPtr, GPtr will be destroied after calling this function. 
    // return 1 means no gcd.
    // return 2 means there is a gcd.
    // return 3 means failed.
    //=================================================================
    sfixn
        MultiUniEuclidean_1(sfixn N, preFFTRep * FPtr, preFFTRep * GPtr, 
                TriSet * ts, TriRevInvSet * tris, MONTP_OPT2_AS_GENE * pPtr)
        {
            sfixn reci, deg, df,dg;
            preFFTRep co, tmp, R;

            df=shrinkDeg(BUSZSI(FPtr, N+1), DAT(FPtr), CUMI(FPtr, N+1));
            dg=shrinkDeg(BUSZSI(GPtr, N+1), DAT(GPtr), CUMI(GPtr, N+1));

            if(df<dg) return  MultiUniEuclidean_1(N, GPtr, FPtr, ts, tris, pPtr);   
            InitOneReducedPoly(&co, N, BDS(ts));
            InitOneReducedPoly(&tmp, N, BUSZS(ts->elems[N]));
            InitOneReducedPoly(&R, N+1, BUSZS(GPtr));

            while (dg>0){

                getLeadingCoefMulti(N+1, &co, GPtr);

                PolyCleanData(&tmp);

                BUSZSDI(tmp, N)=BUSZSI(ts->elems[N], N);
                reci=MultiRecip(N, &tmp, &co, ts, tris, pPtr);
                BUSZSDI(tmp, N)=BDSI(ts, N);

                if(reci==-1) {
                    freePoly(&tmp);
                    freePoly(&co);
                    freePoly(&R);
                    Throw 1;
                    return 3;}

                    if(reci>1){ MultiNumbPolyMulMonicize_1(N, reci, GPtr,  pPtr);}

                    if(reci==0){ MultiCoefPolyMulMonicize_1(N+1, &tmp, GPtr, ts, tris, pPtr);}

                    MultiUniPlainMod_1(N+1, df, FPtr, dg, GPtr, ts, tris, pPtr);

                    if(zeroPolyp(FPtr)){
                        freePoly(&tmp);
                        freePoly(&co);
                        freePoly(&R);
                        return 2;}
                        deg=shrinkDeg(BUSZSI(FPtr, N+1), DAT(FPtr), CUMI(FPtr, N+1));
                        CopyOnePoly_deg(&R, FPtr, deg);
                        CopyOnePoly_deg(FPtr, GPtr, BUSZSI(GPtr, N+1));
                        CopyOnePoly_deg(GPtr, &R, BUSZSDI(R, N+1));
                        dg=shrinkDeg(BUSZSI(GPtr, N+1), DAT(GPtr), CUMI(GPtr, N+1));

            }


            freePoly(&co);
            freePoly(&tmp);
            freePoly(&R);
            return 1;
        }




    //=================================================
    //  Enclidean algorithm. GCD (FPtr, GPtr)
    //
    // return 1 means no gcd.
    // return 2 means there is a gcd.
    // return 3 means failed.
    //===============================================



    /**
     * MultiUniEuclidean:
     * @N: number of variables.
     * @gcd: the gcd of 'FPtr' and 'GPtr'.
     * @FPtr: a C-Cube polynomial.
     * @GPtr: a C-Cube polynomial.
     * @ts: a triangular set 'ts'.
     * @tris: modular inverses of 'ts'.
     * @pPtr: the information of the prime number p.
     * 
     * Standard Enclidean algorithm to compute Greatest Common Divisor of FPtr, and GPtr.
     *
     * Return value: 1 means no GCD, 2 means there is a GCD, 3 means failed.
     **/
    sfixn
        MultiUniEuclidean(sfixn N, preFFTRep * gcd, preFFTRep * FPtr, preFFTRep * GPtr, 
                TriSet * ts, TriRevInvSet * tris, MONTP_OPT2_AS_GENE * pPtr)
        {
            sfixn tag;
            preFFTRep t1, t2;
            InitOnePoly(&t1, N+1, BUSZS(FPtr));
            InitOnePoly(&t2, N+1, BUSZS(GPtr));
            CopyOnePoly(&t1, FPtr);
            CopyOnePoly(&t2, GPtr);

            tag=MultiUniEuclidean_1(N, &t1, &t2, ts, tris, pPtr);

            fromtofftRepMultiD(N+1,  CUM(gcd), DAT(gcd), CUMD(t2), BUSZS(gcd), DATD(t2));
            freePoly(&t1);
            freePoly(&t2);
            if (tag==-1) Throw 2;
            return tag;
        }






    void InitKroTFTRep(KroTFTRep * kPtr, sfixn * resDgs, sfixn N, sfixn M,   
            MONTP_OPT2_AS_GENE * pPtr)
    {
        register int j;
        N(kPtr)=N;     
        M(kPtr)=M;
        ES(kPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn)); 
        DIMS(kPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
        LS(kPtr)=(sfixn *)my_calloc(N+1,sizeof(sfixn));
        CUM(kPtr)=(sfixn * )my_calloc(N+1,sizeof(sfixn));
        CUMI(kPtr, 1)= 1; SIZ(kPtr)=1;
        for(j=1;j<=N;j++){
            ESI(kPtr, j)=logceiling(resDgs[j]+1);
            DIMSI(kPtr, j)=1<<(ESI(kPtr, j));
            LSI(kPtr, j)=resDgs[j]+1;
            SIZ(kPtr)*=LSI(kPtr, j); 
            if(j>=2){CUMI(kPtr, j)=CUMI(kPtr, j-1)*LSI(kPtr, j-1);}
        }

        DATS(kPtr)=(sfixn **)my_calloc(M,sizeof(sfixn *));
        for(j=0;j<M;j++){
            DATSI(kPtr, j)=(sfixn * )my_calloc(SIZ(kPtr),sizeof(sfixn));
        }
        KN(kPtr)=1; KE(kPtr)=0;
        for(j=1;j<=N;j++) {KN(kPtr)*=DIMSI(kPtr, j); KE(kPtr)+=ESI(kPtr, j);}
        KROOTS(kPtr)=NULL;
        kPtr->Defsize=SIZ(kPtr);
    }




    void freeKroTFTRep(KroTFTRep * x){
        int i;
        if(KROOTS(x) != NULL) {my_free(KROOTS(x)); KROOTS(x)=NULL;}
        if(CUM(x) !=NULL) {my_free(CUM(x)); CUM(x)=NULL;}
        if(LS(x) !=NULL) {my_free(LS(x)); LS(x)=NULL;}
        if(ES(x) !=NULL) {my_free(ES(x)); ES(x)=NULL;}
        if(DIMS(x) !=NULL) {my_free(DIMS(x)); DIMS(x)=NULL;}
        if(DATS(x) != NULL){
            for(i=0; i<M(x); i++) 
                if(DATSI(x, i) !=NULL) {my_free(DATSI(x, i)); DATSI(x, i)=NULL;}
            my_free(DATS(x));
            DATS(x)=NULL;}
    }


    void polyMul_TFT(sfixn N, KroTFTRep * kPtr, preFFTRep * rPtr, preFFTRep * p1Ptr, 
            preFFTRep * p2Ptr,  MONTP_OPT2_AS_GENE * pPtr)
    {
        fromtofftRepMultiD(N,  CUM(kPtr), DATSI(kPtr, 0), CUM(p1Ptr), BUSZS(p1Ptr), DAT(p1Ptr));
        fromtofftRepMultiD(N,  CUM(kPtr), DATSI(kPtr, 1), CUM(p2Ptr), BUSZS(p2Ptr), DAT(p2Ptr));
        tftMultiD_test(DATSI(kPtr, 0), DATSI(kPtr, 1), N, ES(kPtr), DIMS(kPtr), LS(kPtr), pPtr);
        fromtofftRepMultiD(N,  CUM(rPtr), DAT(rPtr), CUM(kPtr), BUSZS(rPtr), DATSI(kPtr, 0));
    }

    /**
     * EX_mulPoly_TFT:
     * @N: number of variables.
     * @resPtr: (output) the product of 'f1' and 'f2'.
     * @f1: a C-Cube polynomial.
     * @f2: a C-Cube polynomial.
     * @pPtr: the information of the prime number p.
     * 
     * To compute the product of 'f1' and 'f2' by TFT.
     *
     * Return value: 
     **/

    void EX_mulPoly_TFT(sfixn N, preFFTRep * resPtr, preFFTRep * f1, preFFTRep * f2,  
            MONTP_OPT2_AS_GENE * pPtr)
    {
        KroTFTRep kro;
        InitKroTFTRep(&kro, BUSZS(resPtr), N, 2, pPtr); 
        polyMul_TFT(N, &kro, resPtr, f1, f2,  pPtr);
        freeKroTFTRep(&kro);
        return;
    }

    // return 0, indicating using Kronecker-FFT.
    // return 1, indicating using multiD-TFT.

    int
        forcutoff_Multi_TFT_FFT(sfixn N, sfixn *dgs1, sfixn *dgs2, int cutoff){
            int i;
            for(i=1; i<=N; i++){
                if(dgs1[i]<cutoff) return 0;
                if(dgs2[i]<cutoff) return 0;
            }
            return 1;
        }



    /**
     * EX_mulPoly_TFTFFT:
     * @N: the number of variables.
     * @resPtr: (output) the product of 'f1' and 'f2', the C-Cube polynomial .
     * @f1: a C-Cube polynomial.
     * @f2: a C-Cube polynomial.
     * @pPtr: the information of the prime p.
     * 
     * The product f1 and f2.
     *
     * Return value: 
     **/
    void EX_mulPoly_TFTFFT(sfixn N, preFFTRep * resPtr, preFFTRep * f1, preFFTRep * f2,
            MONTP_OPT2_AS_GENE * pPtr)
    {
        KroTFTRep kro;
        if(forcutoff_Multi_TFT_FFT(N, BUSZS(f1), BUSZS(f2), 128) ==0 ) {
            EX_mulPoly_FFT(N, resPtr, f1, f2, pPtr);
        } else {
            InitKroTFTRep(&kro, BUSZS(resPtr), N, 2, pPtr); 
            polyMul_TFT(N, &kro, resPtr, f1, f2,  pPtr);
            freeKroTFTRep(&kro);
        }
        return;
    }

    void EX_mulPoly_TFTFFT_Bench(sfixn N, preFFTRep * resPtr, preFFTRep * f1, 
            preFFTRep * f2,  MONTP_OPT2_AS_GENE * pPtr, int fftNOTtft)
    {
        KroTFTRep kro;
        if(fftNOTtft){
            EX_mulPoly_FFT_select(N, resPtr, f1, f2, pPtr, 0);
        } else {
            InitKroTFTRep(&kro, BUSZS(resPtr), N, 2, pPtr); 
            polyMul_TFT(N, &kro, resPtr, f1, f2,  pPtr);
            freeKroTFTRep(&kro);
        }
        return;
    }


    //=============================================================
    // Modulo poly by T_n, then by [T_{n-1}, T_{n-2}, ...,T_1] 
    /**
     * MultiUniFastMod_1_TFT:
     * @N: number of variables.
     * @tmpIn: a temp buffer.
     * @n: The next power of 2 of (deg(inPtr, X_N)-deg(T_N, X_N)+1).
     * @inPtr:(output) a C-Cube polynomial.
     * @ts: a triangular set.
     * @tris: modular inverses of 'ts'.
     * @pPtr: the information of the prime p.
     * 
     * Return value: 
     **/
    static void MultiUniFastMod_1_TFT(sfixn N, preFFTRep * tmpIn,  sfixn n, 
            preFFTRep * inPtr, TriSet * ts, TriRevInvSet * tris,  
            MONTP_OPT2_AS_GENE * pPtr)
    {
        sfixn m,e,d1,d2;
        sfixn cutDg;
        //double t=0;
        preFFTRep * resPoly, * resPoly2, * out;
        KroTFTRep * krotftrep, * krotftrep2;

        m=BUSZSI((ELEMI(ts, N)), N);

        if(n<m){ if(DEBUG2) printf("Leaving MultiUniFastMod_1.\n"); 
            return;}

            resPoly=(preFFTRep *)my_calloc(1,sizeof(preFFTRep));
            resPoly2=(preFFTRep *)my_calloc(1,sizeof(preFFTRep));
            out=(preFFTRep *)my_calloc(1,sizeof(preFFTRep));

            krotftrep=(KroTFTRep *)my_calloc(1,sizeof(KroTFTRep));
            krotftrep2=(KroTFTRep *)my_calloc(1,sizeof(KroTFTRep));
            e=logceiling(n-m+1);
            if(! (EXSTI(tris, N))){  
                initCopyOneRevInv(out, ELEMI(tris, N));
                NewtonRevInverse(N, out, ELEMI(tris, N), ELEMI(ts, N), ts, tris, pPtr );
                freePoly(out);}
                cutDg=n-m;
                d1=BUSZSI(tmpIn, N);
                d2=BUSZSI(ELEMI(tris, N), N);
                BUSZSI(tmpIn, N)=cutDg;
                BUSZSI(ELEMI(tris, N), N)=cutDg;
                reverseMulti_1(n, CUMI(tmpIn, N), DAT(tmpIn));
                InitResPoly(resPoly, N, BUSZS(tmpIn), BUSZS(ELEMI(tris, N)));

                InitKroTFTRep(krotftrep, BUSZS(resPoly), N, 2, pPtr);
                polyMul_TFT(N, krotftrep, resPoly, tmpIn, ELEMI(tris, N), pPtr);
                freeKroTFTRep(krotftrep);

                reverseMulti_1(cutDg, CUMI(resPoly, N), DAT(resPoly));

                BUSZSI(tmpIn, N)=d1;
                BUSZSI(ELEMI(tris, N), N)=d2;

                PolyCleanData(tmpIn);

                BUSZSI(tmpIn, N)=cutDg;

                reduceCoeffs(N, tmpIn, resPoly, ts, tris, pPtr);

                freePoly(resPoly); 

                InitResPoly(resPoly2, N,  BUSZS(ELEMI(ts, N)), BUSZS(tmpIn));
                InitKroTFTRep(krotftrep2, BUSZS(resPoly2), N, 2, pPtr); 

                polyMul_TFT(N, krotftrep2, resPoly2, ELEMI(ts, N), tmpIn,  pPtr);

                freeKroTFTRep(krotftrep2);
                BUSZSI(tmpIn, N)=d1;

                PolyCleanData(tmpIn);

                reduceCoeffs(N, tmpIn, resPoly2, ts, tris, pPtr);

                freePoly(resPoly2);

                subPoly_1(N, inPtr, tmpIn, pPtr->P);

                my_free(resPoly); 
                my_free(resPoly2);
                my_free(out);
                my_free(krotftrep); 
                my_free(krotftrep2);
    }


    /**
     * MultiMod_BULL:
     * @N: number of variables.
     * @outPtr: (output) the image of 'inPtr' modulo 'ts'.
     * @inPtr: a C-Cube polynomial.
     * @ts: a triangular set.
     * @tris: modular inverse of 'ts'.
     * @pPtr: the information of prime number p.
     * 
     * Reduce 'inPtr' w.r.t 'ts' and save the result in 'outPtr'.
     *
     * Return value: 
     **/
    void MultiMod_1_BULL(sfixn N, preFFTRep * inPtr, TriSet * ts, 
            TriRevInvSet * tris,  MONTP_OPT2_AS_GENE * pPtr)
    {
        sfixn *lastNonZeroPtr, *datPtr, *tmpVec;
        sfixn siz, deg, i, j;
        preFFTRep * tmpInPtr, *cpShellInPtr;

        signal(SIGINT,catch_intr);

        if(N<1) Throw 15;
        siz=BUSZSI(inPtr, 1)+1;
        lastNonZeroPtr=DAT(inPtr)+shrinkDeg(SIZ(inPtr)-1, DAT(inPtr), 1);  
        for(datPtr=DAT(inPtr); datPtr<=lastNonZeroPtr; datPtr+=siz) {
            deg=shrinkDeg(BUSZSI(inPtr, 1), datPtr, 1);
            if (((BDSI(ts, 1))>DivCut1)) {
                UniFastMod_1(deg, datPtr, BUSZSI((ELEMI(ts, 1)), 1), DAT(ELEMI(ts, 1)), 
                        (NLB(tris))[1], DAT(ELEMI(tris, 1)) ,pPtr); 
            } else {
                UniPlainMod_1(deg, datPtr, BUSZSI((ELEMI(ts, 1)), 1), 
                        DAT(ELEMI(ts, 1)), pPtr); 
            }
        }

        if(N==1) return;
        if(Interrupted==1) { return; }
        tmpInPtr=(preFFTRep *)my_malloc(sizeof(preFFTRep));
        tmpVec=(sfixn *)my_calloc(N+1, sizeof(sfixn));
        for(i=2; i<N-1; i++){  
            for(datPtr=DAT(inPtr); datPtr<=lastNonZeroPtr; datPtr+=CUMI(inPtr, i+1)){
                deg=shrinkDeg(CUTSI(inPtr, i), datPtr, CUMI(inPtr, i));      
                for(j=0; j<i;j++) tmpVec[j]=BDSI(ts, j);
                tmpVec[i]=deg;
                InitOneReducedPoly(tmpInPtr, i, tmpVec);
                fromtofftRep(i, tmpInPtr->accum, tmpInPtr->data, CUM(inPtr), 
                        BUSZS(tmpInPtr), datPtr);
                cpShellInPtr=(preFFTRep *)my_malloc(sizeof(preFFTRep));
                copyPolyPointers(cpShellInPtr, inPtr);
                N(cpShellInPtr)=i;
                SIZ(cpShellInPtr)=CUMI(inPtr, i+1);
                DAT(cpShellInPtr)=datPtr;

                if((((i==2)&&(BDSI(ts, i)>DivCut2)) || 
                            ((i==3)&&(BDSI(ts, i)>DivCut3)) || 
                            ((i>3)&&(BDSI(ts, i)>DivCutN))))
                {
                    MultiUniFastMod_1_TFT(i, tmpInPtr, deg, cpShellInPtr, ts, tris, pPtr);
                } else {
                    MultiUniPlainMod_1(i, deg, tmpInPtr, BDSI(ts, i)+1,  ELEMI(ts, i), ts, tris, pPtr);

                    fromtofftRep(i, CUM(cpShellInPtr), DAT(cpShellInPtr), CUM(tmpInPtr),
                            BUSZS(tmpInPtr), DAT(tmpInPtr));
                }
                freePoly(tmpInPtr);
                my_free(cpShellInPtr);
            }
        }

        my_free(tmpVec);
        my_free(tmpInPtr);    
        if(Interrupted==1) { return; }
        if(N>=3){
            tmpInPtr=(preFFTRep *)my_malloc(sizeof(preFFTRep));
            tmpVec=(sfixn *)my_calloc(N, sizeof(sfixn));
            for(datPtr=DAT(inPtr); datPtr<=lastNonZeroPtr; datPtr+=CUMI(inPtr, N)){
                deg=shrinkDeg(CUTSI(inPtr, N-1), datPtr, CUMI(inPtr, N-1));      
                for(j=0; j<N-1;j++) tmpVec[j]=BDSI(ts, j);
                tmpVec[N-1]=deg;
                InitOneReducedPoly(tmpInPtr, N-1, tmpVec);
                fromtofftRep(N-1, tmpInPtr->accum, tmpInPtr->data, CUM(inPtr), 
                        BUSZS(tmpInPtr), datPtr);
                cpShellInPtr=(preFFTRep *)my_malloc(sizeof(preFFTRep));
                copyPolyPointers(cpShellInPtr, inPtr);
                N(cpShellInPtr)=N-1;
                SIZ(cpShellInPtr)=CUMI(inPtr, N);
                DAT(cpShellInPtr)=datPtr;

                if((((N-1==2)&&(BDSI(ts, N-1)>DivCut2)) 
                            || ((N-1==3)&&(BDSI(ts, N-1)>DivCut3)) 
                            || ((N-1>3)&&(BDSI(ts, N-1)>DivCutN))))
                {
                    MultiUniFastMod_1_TFT(N-1, tmpInPtr, deg, cpShellInPtr, ts, tris, pPtr);
                } else{ 
                    MultiUniPlainMod_1(N-1, deg, tmpInPtr, BDSI(ts, N-1)+1,  ELEMI(ts, N-1), ts, tris, pPtr);
                    fromtofftRep(N-1, CUM(cpShellInPtr), DAT(cpShellInPtr), CUM(tmpInPtr),
                            BUSZS(tmpInPtr), DAT(tmpInPtr));
                }
                freePoly(tmpInPtr);
                my_free(cpShellInPtr);
            }
            my_free(tmpVec);
            my_free(tmpInPtr);
        }
        if(Interrupted==1) { return; }

        tmpInPtr=(preFFTRep *)my_malloc(sizeof(preFFTRep));
        tmpVec=(sfixn *)my_calloc(N+1, sizeof(sfixn));
        deg=shrinkDeg(BUSZSI(inPtr, N), DAT(inPtr), CUMI(inPtr, N));
        for(j=0; j<N; j++) tmpVec[j]=BDSI(ts, j);
        tmpVec[N]=deg;
        InitOneReducedPoly(tmpInPtr, N, tmpVec);
        fromtofftRep(N, tmpInPtr->accum, tmpInPtr->data, CUM(inPtr), 
                BUSZS(tmpInPtr), DAT(inPtr));
        MultiUniFastMod_1_TFT(N, tmpInPtr, deg, inPtr, ts, tris, pPtr);

        freePoly(tmpInPtr);
        my_free(tmpVec);
        my_free(tmpInPtr);
    }






    //=========================================================================
    // Bottom-up level-by-level reduction.
    // using TFTs.
    //=========================================================================
    void MultiMod_BULL(sfixn N, preFFTRep * outPtr, preFFTRep * inPtr, TriSet * ts, 
            TriRevInvSet * tris,  MONTP_OPT2_AS_GENE * pPtr)
    { 
        if (zeroCoefp(DAT(inPtr), SIZ(inPtr)) ) return;
        MultiMod_1_BULL(N, inPtr, ts, tris, pPtr);
        fromtofftRep(N, CUM(outPtr), DAT(outPtr), CUM(inPtr),  BUSZS(outPtr), DAT(inPtr));
    }


    //================================================================
    // Serail Multimod.
    //  either using MultiMod_DF().
    //  or using MultiMod_BULL().
    //================================================================

    /**
     * MultiMod:
     * @N: number of variables.
     * @outPtr: (output) the image of 'inPtr' modulo 'ts'.
     * @inPtr: a C-Cube polynomial.
     * @ts: a triangular set.
     * @tris: modular inverse of 'ts'.
     * @pPtr: the information of prime number p.
     * 
     * Reduce 'inPtr' w.r.t 'ts' and save the result in 'outPtr'.
     *
     * Return value: 
     **/
    void MultiMod(sfixn N, preFFTRep * outPtr, preFFTRep * inPtr, TriSet * ts, 
            TriRevInvSet * tris,  MONTP_OPT2_AS_GENE * pPtr)
    { 
        if(zeroPolyp(inPtr)) return;
        if(1){
            MultiMod_DF(N, outPtr, inPtr, ts, tris, pPtr);
        }else {
            MultiMod_BULL(N, outPtr, inPtr, ts, tris, pPtr);
        }
    }


    // 1 -> DF
    // 0 -> BULL
    /**
     * MultiMod_OPT:
     * @N: number of variables.
     * @outPtr: (output) the image of 'inPtr' modulo 'ts'.
     * @inPtr: a C-Cube polynomial.
     * @ts: a triangular set.
     * @tris: modular inverse of 'ts'.
     * @pPtr: the information of prime number p.
     * @opt: switching to 1 using Depth-first traversal.
     *       switching to 0 using Level-order traversal.
     *
     * Reduce 'inPtr' w.r.t 'ts' and save the result in 'outPtr'.
     *
     * Return value: 
     **/
    void
        MultiMod_OPT(sfixn N, preFFTRep * outPtr, preFFTRep * inPtr, TriSet * ts, 
                TriRevInvSet * tris,  MONTP_OPT2_AS_GENE * pPtr, int opt)
        {

            if(zeroPolyp(inPtr)) return;

            if(opt){
                MultiMod_DF(N, outPtr, inPtr, ts, tris, pPtr);
            }else {
                MultiMod_BULL(N, outPtr, inPtr, ts, tris, pPtr);
            }
        }







    //===================================================
    // MulMod
    // (1) resPtr = f1 * f2.
    // (2) out = resPtr modulo TS.
    //  * resPtr is allocated within the function.
    //===================================================
    preFFTRep * EX_mul_Reduced_ForLifting(sfixn N, preFFTRep * out, preFFTRep * f1, 
            preFFTRep * f2, TriSet * ts, TriRevInvSet * tris, MONTP_OPT2_AS_GENE * pPtr)
    {
        preFFTRep res;

        sfixn d1,d2,d3,d4,d5,sz1, sz2;
        d1=shrinkDeg(BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
        d2=shrinkDeg(BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
        sz1=getDenseSiz(N, f1, BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
        sz2=getDenseSiz(N, f2, BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
        d3=BUSZSI(f1, N);
        d4=BUSZSI(f2, N);
        InitResPoly(&res, N,  BUSZS(f1), BUSZS(f2));


        if((sz1>=MulCut)||(sz2>=MulCut)){

            d5=BUSZSDI(res, N);
            BUSZSI(f1, N)=d1;
            BUSZSI(f2, N)=d2;
            BUSZSDI(res, N)=d1+d2;

            EX_mulPoly_TFTFFT(N, &res, f1, f2, pPtr);

            BUSZSI(f1, N)=d3;
            BUSZSI(f2, N)=d4;
            BUSZSDI(res, N)=d5;
        } else{

            if(N==1){
                EX_Mont_PlainMul_OPT2_AS_GENE(BUSZSDI(res, 1), res.data, d1, DAT(f1), d2, DAT(f2), pPtr); 
            } else{

                plainMultiDMul(N, res.accum, res.data, CUM(f1), BUSZS(f1), CUM(f2), BUSZS(f2), DAT(f1), DAT(f2), pPtr); }

        }

        MultiMod_ForLifting(N, out, &res, ts, tris, pPtr, SIZ(ELEMI(ts, N)));

        freePoly(&res);
        return out;
    }


    void MultiMod_ForLifting(sfixn N, preFFTRep * outPtr, preFFTRep * inPtr, 
            TriSet * ts, TriRevInvSet * tris,  MONTP_OPT2_AS_GENE * pPtr, sfixn size)
    {
        if(size<2){
            MultiMod_DF_CLS(N, outPtr, inPtr, ts, tris, pPtr);
        }else{
            MultiMod_BULL(N, outPtr, inPtr, ts, tris, pPtr);
        }
    }


    /**
     * EX_getInitial:
     * @poly: C-Cube polynomial.
     * 
     * Return the initial of 'poly'.
     *
     * Return value: the initial of 'poly'.
     */
    preFFTRep * EX_getInitial(preFFTRep *poly){
        preFFTRep *lc;
        lc = EX_InitOnePoly(N(poly)-1, BUSZS(poly));
        getLeadingCoefMulti(N(poly), lc, poly);
        return lc;
    }





    // inner function.
    preFFTRep *shrinkOneDim(preFFTRep *inPoly){
        sfixn N, *dgs;
        sfixn *dat1, *dat2;
        int i;
        preFFTRep *newpoly;

        N = N(inPoly);
        dgs=(sfixn *)my_calloc(N, sizeof(sfixn));

        for(i=1; i<N; i++){
            dgs[i] = BUSZSI(inPoly, i); 
        }
        newpoly = EX_InitOnePoly(N-1, dgs);

        dat1 = DAT(newpoly);
        dat2 = DAT(inPoly);   

        for(i=0; i<SIZ(newpoly); i++){
            DAT(newpoly)[i] = DAT(inPoly)[i];
        }

        my_free(dgs);
        return newpoly;

    }



    preFFTRep *shrinkDegPoly_1(preFFTRep * inPoly){
        sfixn i, dg, N, *data, size;
        N=N(inPoly);
        dg=shrinkDeg(BUSZSI(inPoly, N), DAT(inPoly), CUMI(inPoly, N));
        if(dg<BUSZSI(inPoly, N)){
            size=(dg+1)*(CUMI(inPoly, N));
            data=(sfixn *)my_calloc(size, sizeof(sfixn));
            for(i=0; i<size; i++){
                data[i]=DAT(inPoly)[i];
            }
            SIZ(inPoly)=size;
            BUSZSI(inPoly, N)=dg;
            DFSIZ(inPoly)=SIZ(inPoly);
            my_free(DAT(inPoly));
            DAT(inPoly)=data;
        }
        return inPoly;
    }




    preFFTRep *EX_shrinkDegPoly(preFFTRep * inPoly){
        preFFTRep *outPoly;
        outPoly = EX_CopyOnePoly(inPoly);
        return shrinkDegPoly_1(outPoly);
    }

    preFFTRep *EX_GetPolyTail(preFFTRep * inPoly){
        preFFTRep *outPoly;
        sfixn N=N(inPoly);
        if(BUSZSI(inPoly, N) ==0) return CreateZeroPoly();
        BUSZSI(inPoly, N)--;
        outPoly = EX_CopyOnePoly(inPoly);
        BUSZSI(inPoly, N)++;
        return shrinkDegPoly_1(outPoly);
    }





    /**
     * EX_NormalizePoly:
     * @ininPoly: a C-Cube polynomial.
     * 
     * Make the data buffer of the input polynomial "tight".
     * Namely, remove leading zeros and shrink empty dimensions.
     *
     * Return value: The data-normalized version of the inpout polynomial.
     **/
    preFFTRep *EX_NormalizePoly(preFFTRep * ininPoly){
        sfixn N, dg;
        preFFTRep *inPoly, *newPoly1, *newPoly2;

        N = N(ininPoly);

        inPoly = EX_shrinkDegPoly(ininPoly);

        dg=shrinkDeg(BUSZSI(inPoly, N), DAT(inPoly), CUMI(inPoly, N));

        if ((dg == 0) && (N >1 )){
            newPoly1=shrinkOneDim(inPoly);
            EX_FreeOnePoly(inPoly);
            newPoly2=EX_NormalizePoly(newPoly1);
            EX_FreeOnePoly(newPoly1);
            return newPoly2;
        } else {
            newPoly2 = EX_CopyOnePoly(inPoly);
            EX_FreeOnePoly(inPoly);
            return newPoly2;
        }

    }



    /**
     * EX_NormalForm:
     * @N: the number of variables.
     * @poly: the C-Cube polynomial.
     * @ts: a trinagular set.
     * @tris: the modular inverses of 'ts'.
     * @pPtr: the information of the prime number p.
     *
     * Compute the normal form of 'poly' w.r.t. 'ts'. 
     *
     * Return value: 
     **/
    preFFTRep *
        EX_NormalForm(sfixn N, preFFTRep *poly, TriSet *ts, TriRevInvSet *tris, MONTP_OPT2_AS_GENE *pPtr){

            preFFTRep *out, *in, *res;
            int i, makecopy;
            sfixn *dgs;

            out = EX_InitOnePoly(N, BDS(ts));


            makecopy=0;
            dgs=(sfixn *)my_calloc(N+1, sizeof(sfixn));
            for(i=1; i<=N; i++){     
                if( BUSZSI(poly,i) <= BDSI(ts,i) ){
                    dgs[i] = BDSI(ts,i)+1;
                    makecopy=1;
                }else{
                    dgs[i]=BUSZSI(poly,i);
                }
            }

            if(makecopy==1){
                in = EX_InitOnePoly(N, dgs);
                my_free(dgs);
                fromtofftRep(N, CUM(in), DAT(in), CUM(poly),  BUSZS(poly), DAT(poly));
                MultiMod_OPT(N, out, in, ts, tris, pPtr, 0);
                EX_FreeOnePoly(in);
            }else{
                my_free(dgs);
                MultiMod_OPT(N, out, poly, ts, tris, pPtr, 0);
            }

            if ( BUSZSI(out, N) == 0 ) {
                res = EX_NormalizePoly(out);
                EX_FreeOnePoly(out);
            }
            else
            { res = out;}

            return res;
        }




    // <<checked>>
    sfixn *
        EX_getDgsForNormalForm(preFFTRep *poly, TriSet *ts, sfixn e){
            sfixn *dgs, M=N(poly);
            int i;

            dgs = (sfixn *)my_calloc(M+1, sizeof(sfixn));

            for(i=1; i<=M; i++){
                dgs[i]=BDSI(ts,i)+1;
                if (dgs[i]<BUSZSI(poly, i)) dgs[i]=BUSZSI(poly, i);
                dgs[i]<<=e;
            }
            return dgs;
        }





    // <<checked>>
    /**
     * EX_EY_NormalForm:
     * @poly: the C-Cube polynomial.
     * @ts: a trinagular set.
     * @pPtr: the information of the prime number p.
     *
     * Compute the normal form of 'poly' w.r.t. 'ts'. 
     *
     * Return value: normal form of input 'poly'
     **/
    preFFTRep *
        EX_EY_ForNormalForm(preFFTRep *poly, TriSet *ts, MONTP_OPT2_AS_GENE *pPtr){
            sfixn *dgs, M, N;
            TriRevInvSet * tris;
            preFFTRep *newpoly, *res;
            M = N(poly);
            N = N(ts);

            assert(M<=N);
            if(constantPolyp(poly)) return EX_CopyOnePoly(poly);
            dgs = EX_getDgsForNormalForm(poly, ts, 0);
            tris = EX_initTriRevInvSet(dgs, M, ts);
            getRevInvTiSet(dgs, M, tris, ts, pPtr);
            my_free(dgs);
            newpoly = EX_NormalForm(M, poly, ts, tris, pPtr);
            res = EX_NormalizePoly(newpoly);

            EX_FreeOnePoly(newpoly);
            EX_freeTriRevInvSet(tris);
            return res;
        }




    // <<checked>>

    /**
     * EX_EY_Normalize:
     * @poly: the input polynomial.
     * @ts: the zero dimensional triangular set.
     * @pPtr: the information of the prime number.
     * 
     * make the input polynomial monic and reduced w.r.t the triangular set 'ts'.
     *
     * Return value: the normalized 'poly'. 
     **/
    preFFTRep *
        EX_EY_Normalize(preFFTRep *poly, TriSet *ts, MONTP_OPT2_AS_GENE *pPtr){
            sfixn *dgs, N, M;
            TriRevInvSet * tris;
            preFFTRep *newpoly, *out=NULL;
            N = N(poly);
            M = N - 1;
            assert(M==N(ts));

            dgs = EX_getDgsForNormalForm(poly, ts, 0);
            tris = EX_initTriRevInvSet(dgs, M, ts);
            getRevInvTiSet(dgs, M, tris, ts, pPtr);
            my_free(dgs);
            newpoly = EX_ReduceCoeffs(N, poly, ts, tris, pPtr);
            MonicizePoly_1(N, newpoly, ts, tris, pPtr);
            EX_freeTriRevInvSet(tris);
            if ( BUSZSI(newpoly, N) == 0 ) {
                out = EX_NormalizePoly(newpoly);
                EX_FreeOnePoly(newpoly);
            }
            else
            { out = newpoly;}

            return out;
        }


    // <<checked>>
    // get real degree(poly, M)
    sfixn realDegPoly(preFFTRep *poly, sfixn M){
        sfixn N=N(poly), offset=0, chunk, dg=0, dg1;
        assert(M<=N);
        if(M==N){
            chunk = SIZ(poly);
        }else{
            chunk = CUMI(poly, M+1);
        }

        for(offset=0; offset<SIZ(poly); offset+=chunk){
            dg1=shrinkDeg(BUSZSI(poly, M), DAT(poly)+offset, CUMI(poly, M));
            if (dg1>dg) dg = dg1;
        }
        return dg;
    }






    preFFTRep *CreateZeroPoly(){
        sfixn dgs[2]={0, 0};
        preFFTRep *newPoly;
        newPoly = EX_InitOnePoly(1, dgs);
        return newPoly;
    }


    preFFTRep *CreateConsPoly(sfixn cons){
        sfixn dgs[2]={0, 0};
        preFFTRep *newPoly;
        newPoly = EX_InitOnePoly(1, dgs);
        DAT(newPoly)[0] = cons;
        return newPoly;
    }




    /**
     * CreateUniPoly:
     * @dg: the degree of the new polynomial.
     * @vec: coefficients of the new polynomial.
     *
     * Create a univariate C-Cube polynomial.
     * 
     * Return value: a C-Cube polynomial.
     **/
    preFFTRep *CreateUniPoly(sfixn dg, sfixn *vec){
        sfixn dgs[2]={0, 0};
        int i;
        preFFTRep *newPoly;
        dgs[1]=dg;
        newPoly = EX_InitOnePoly(1, dgs);
        for(i=0; i<=dgs[1]; i++){DAT(newPoly)[i]=vec[i]; }
        return newPoly;
    }







    // <<checked>>
    TriSet * EX_ExchangeOnePoly(preFFTRep *poly, TriSet *ints, MONTP_OPT2_AS_GENE *pPtr){
        TriSet *outts;
        int invertibility;
        sfixn N, M;
        N = N(ints);
        M = N(poly);
        assert(M<=N(ints));
        outts = EX_CopyOneTriSet(ints);
        EX_FreeOnePoly(ELEMI(outts, M));
        assert(shrinkDeg(BUSZSI(poly, M), DAT(poly), CUMI(poly, M)) == BUSZSI(poly, M));
        BDSI(outts, M)=BUSZSI(poly, M) - 1;
        ELEMI(outts, M) = poly;
        invertibility=MonicizeTriSet_1(N, outts, pPtr);
        if(invertibility==-1){
            Throw 102;
            return NULL;}
            return outts;
    }





    // Merge {start .. index+1} of ts1, with poly, then with {index-1, 1} of ts2.
    // 
    TriSet * EX_MergeTriSet(sfixn start, sfixn index, preFFTRep *inPoly, 
            TriSet *ts_top, TriSet *ts_under, MONTP_OPT2_AS_GENE *pPtr)
    {
        TriSet *outts;
        sfixn *dgs, i, N;
        int invertibility;
        preFFTRep *poly;

        //printf("++++++++++++++++++++++++++++++++++++++++++++++++++\n");
        //printf("start=%d\n", start);
        //printf("index=%d\n", index);
        //fflush(stdout);
        //printf("inPoly=");
        //printPoly(inPoly);
        //fflush(stdout);
        //printf("ts_top");
        //printTriSet(ts_top);
        //fflush(stdout);
        //printf("ts_under");
        //printTriSet(ts_under);
        //printf("++++++++++++++++++++++++++++++++++++++++++++++++++\n");
        //fflush(stdout);

        assert((start <= N(ts_top)) && (start>=1));
        assert((index <= start) && (index >= 1));

        dgs = (sfixn *) my_calloc(start, sizeof(sfixn));
        poly = EX_NormalizePoly(inPoly);

        for(i=1; i<index; i++){ dgs[i-1]=BDSI(ts_under, i)+1; }
        dgs[index-1]=BUSZSI(poly, index);
        for(i=index+1; i<=start; i++){ dgs[i-1]=BDSI(ts_top, i)+1; }

        N=start;
        outts = createWrapperDenseTriSet(N, dgs);
        my_free(dgs);
        for(i=1; i<index; i++){
            ELEMI(outts, i) =  EX_CopyOnePoly(ELEMI(ts_under, i));
        }

        ELEMI(outts, index)= poly;

        for(i=index+1; i<=start; i++){
            ELEMI(outts, i)=  EX_CopyOnePoly(ELEMI(ts_top, i));
        }

        invertibility = MonicizeTriSet_1(N, outts, pPtr);

        if(invertibility==-1){
            printf("Can't be monicized!\n");
            fflush(stdout);
            Throw 102;
            return NULL;
        }

        return outts;
    }




    // Get partial degree vector.
    // int _inner returns 0 if the coefficient polynomial is a zero polynomial.
    // otherwise, returns 1.

    int EX_getPartialDegVec_inner(sfixn *coefVecSizAddr, sfixn N, sfixn *dgs, 
            sfixn *accum, sfixn *data, sfixn *locAddr, sfixn *pdegVec)
    {
        int i, offset=0, r=0;
        sfixn stLoc=*locAddr;
        sfixn d=-1; // tester for zero-coefficient.
        //printf("N=%ld.\n", N);
        if(N==1){
            //printf("dgs[%ld]=%ld\n", N, dgs[N]);
            d=shrinkDegUni(dgs[N], data);
            //printf("d=%ld\n", d);

            if((d==0) && (data[0]==0)) {
                pdegVec[(*locAddr)++]=-1;
                return 0;      
            }
            else{
                pdegVec[(*locAddr)++]=d;
                (*coefVecSizAddr)+=d+1;
                return 1;      
            }
        }

        else{

            for(i=0; i<=dgs[N]; i++){
                r=EX_getPartialDegVec_inner(coefVecSizAddr, N-1, dgs, accum, data+offset, locAddr, pdegVec);
                // increase the d for degree.
                if(r) d = i;
                offset+=accum[N];
            }

            if(d==-1) {
                // if d==-1, we have recorded dgs[N] "-1" 
                //  in adjecentp last slots.
                //  We only want to record 1 single -1.
                (*locAddr)=stLoc;  
                pdegVec[(*locAddr)++]=-1; 
                // above 2 statement corresponds to the stack.
                // pop up all the "-1"s for all zero coefficients,
                // and push back a zero for encoding the zero polynomial.
                return 0;} 
            else { pdegVec[(*locAddr)++]=d; 
                return 1;}

        }

    }



    sfixn *RemoveMinusTwo(sfixn *vec){
        int siz=vec[0];
        int slider=0, i;
        for(i=1; i<=siz; i++){
            vec[i-slider]=vec[i];
            if(vec[i]==-2) ++slider;
        }
        vec[0]-=slider;
        return vec;
    }


    // at least # of variblabe >=2.
    // suppose NO leading -1's at the very beginning.
    // -1 is useful zeros.
    // -2 is useless zeros.
    void RemoveLCMinusOneAndFillCoefs(sfixn base, sfixn *coefSizeFinalAddr, 
            preFFTRep *Ptr, sfixn *coefVec, sfixn *locAddr, sfixn level,sfixn *vec)
    {
        int d, d2, i, diff;
        //FILE *file;
        if((*locAddr)==0) return;

        if(level==1){
            d=vec[(*locAddr)--];
            if(d >= 0) {
                copyVec_0_to_d(d, coefVec+(*coefSizeFinalAddr), DAT(Ptr)+base-(BUSZSI(Ptr,level)+1));
                (*coefSizeFinalAddr)+=d+1;
            }

            return;
        }else{

            d=vec[(*locAddr)--];

            if(level == N(Ptr) && d == -1 ) return; 


            if(d==-1){
                // do nothing.
            }
            else{
                d2=BUSZSI(Ptr, level) - d;
                while ((vec[(*locAddr)]==-1)&&((*locAddr)>0) && ((d2--)>0)) {vec[(*locAddr)--]=-2;}
                diff=BUSZSI(Ptr,level)-d;
                base-=diff*CUMI(Ptr, level);    
                for(i=0; i<=d; i++){
                    RemoveLCMinusOneAndFillCoefs(base, coefSizeFinalAddr, Ptr, coefVec, locAddr, level-1, vec);
                    base-=CUMI(Ptr, level);
                }
            }
        }
    }


    /* // Input: a given polynomial. */
    /* // Output: the partial-degree vector. */
    /* //         The partial-degree vector encodes a tree structure. */
    /* //         E.g. [1,2,1] encode a bivariate polynomial tree shape. Say x>y.  */
    /* //                        x[1] */
    /* //                        /  \ */
    /* //                      y[2] y[1] */
    /* //         Inside the Vec, if Vec[i]=-1, implies the tree node is zero!!! */
    /* //            where var[numb] represents the partial degree in "var" is "numb". */
    /* //         The parital degree vector vec[0] is the trueSize. */
    /* //  */
    /* //         The coefVec keeps the coefficents in X_{n-1}  */
    /* //         ( X_{n-1} is the second smallest variable ),  */
    /* //         in a post-order layout. Namely, Post-order saving the coefficients */
    /* //         except for the bottom variable. For bottom varible X_n still keeping */
    /* //         the coefficients from low degree to high degree.  */
    /* //        */      
    sfixn *EX_getPartialDegVec(preFFTRep *Ptr){
        /*   // the size of the PartialDegVec should be  */
        /*   //    E.g. f in  [x,y,z], x>y>z, deg(f,x)=3, deg(f,y)=2, deg (f,z)=4. */
        /*   //                              => size of the vec is 2+(4+1)*(2+1).       */
        /*   // the partial degree vector's maximum possible size. */
        /*   // Only encoding the degrees. plus 2 is for the top level var,  */
        /*   // and for keeping the size info of the vector. */
        /*   //  first "1" is for vec[0], second "1" is for the root node in the pDeg */
        sfixn szPdegVec=(SIZ(Ptr) / CUMI(Ptr, 2))+ (SIZ(Ptr) / CUMI(Ptr, 3)) + 1 + 1;
        sfixn *pdegVec=(sfixn *) my_calloc(szPdegVec, sizeof(sfixn));
        sfixn loc=1, *locAddr=&loc;
        sfixn coefVecSiz=0, *coefVecSizAddr=&coefVecSiz;
        sfixn coefSizeFinal=0, *coefSizeFinalAddr=&coefSizeFinal;
        sfixn *coefVec;
        if (N(Ptr)==0)  {
            printf("in EX_getPartialDegVec N(Ptr)==0, which is invalid input."); 
            Throw 1001;}

            //printf("approx. size of degVec is %ld.\n", szPdegVec);

            //printf("Ptr=");
            //printPoly(Ptr);
            //fflush(stdout);

            EX_getPartialDegVec_inner(coefVecSizAddr, N(Ptr), BUSZS(Ptr), CUM(Ptr), DAT(Ptr), locAddr, pdegVec);

            //printf("loc=%ld\n", loc);
            loc--;
            pdegVec[0]=loc;

            coefVec=(sfixn *)my_calloc(coefVecSiz, sizeof(sfixn));

            RemoveLCMinusOneAndFillCoefs(SIZ(Ptr), coefSizeFinalAddr, Ptr, coefVec, locAddr, N(Ptr),  pdegVec);

            RemoveMinusTwo(pdegVec);

            my_free(coefVec);
            return pdegVec;
    }





    // X_N is the main variable of FPtr,GPtr.
    // X_{N-1} the main variable of TriSet.
    /**
     * EX_MonicMultiPlainDivide:
     * @N: the number of variables.
     * @FPtr: a C-Cube polynomial.
     * @GPtr: a C-Cube polynomial.
     * @ts: a triangular set.
     * @pPtr: the information of the prime number p.
     * 
     * In the formula FPtr = GPtr*QPtr + RPtr mod ts
     * FPtr, GPtr and ts are inpout.
     * QPtr is the output
     * Return value: return the quotient of 'FPtr' divided by 'GPtr' modulo 'ts'.
     **/
    preFFTRep *
        EX_MonicMultiPlainDivide(sfixn N, preFFTRep *FPtr, preFFTRep *GPtr, TriSet *ts, MONTP_OPT2_AS_GENE *pPtr){
            sfixn i, *dgs, dF, dG, dQ;
            preFFTRep *QPtr;
            TriRevInvSet *tris;

            dF=shrinkDeg(BUSZSI(FPtr, N), DAT(FPtr), CUMI(FPtr, N));
            dG=shrinkDeg(BUSZSI(GPtr, N), DAT(GPtr), CUMI(GPtr, N));
            dQ=dF-dG;
            if(dQ<0) {printf("dF<dG! in EX_MultiUniPlainDivide()."); Throw 123;}
            dgs=(sfixn *)my_calloc(N+1, sizeof(sfixn));
            for(i=1; i<N; i++){
                dgs[i]=BDSI(ts, i);
            }
            dgs[N] = dF;
            QPtr = EX_InitOnePoly(N, dgs);
            my_free(dgs);

            //// to check !

            dgs=(sfixn *)my_calloc(N+1, sizeof(sfixn));
            for(i=1; i<N; i++){
                dgs[i] = BDSI(ts, i) + 1;
                dgs[i]<<=1;
            }


            // --> RevInv01.
            tris = EX_initTriRevInvSet(dgs, N-1, ts);
            getRevInvTiSet(dgs, N-1, tris, ts, pPtr);

            my_free(dgs);

            MultiUniPlainDivide_1(N, QPtr, FPtr, GPtr, ts, tris, pPtr);

            EX_freeTriRevInvSet(tris);


            return QPtr;
        }



    /**
     * EX_getRevInvTriSet:
     * @N: the number of variables.
     * @ts: a triangualr set.
     * @pPtr: the information the prime.
     * 
     * To compute the modular inverses of 'ts'.
     *
     * Return value: the modular inverses of 'ts'.
     **/
    TriRevInvSet *
        EX_getRevInvTriSet(sfixn N,  TriSet * ts,  MONTP_OPT2_AS_GENE * pPtr){
            int i;
            TriRevInvSet *tris;
            sfixn *dgs;
            dgs=(sfixn *)my_calloc(N+1, sizeof(sfixn));
            for(i=1; i<=N; i++){
                dgs[i] = BDSI(ts, i) + 1;
            }
            tris=(TriRevInvSet *)my_malloc(sizeof(TriRevInvSet));
            initTriRevInvSet( dgs, N, tris, ts);
            getRevInvTiSet(dgs, N, tris, ts, pPtr);
            my_free(dgs);
            return tris;

        }

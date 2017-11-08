/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
///////////////////////////////////////////////////////////////////////////////// 
// Log:
// 
// WP, 2009/08/26/
//
// (0) Align all the functions
//
// (1) For all TFT based functions, the number of points computed is changed to
//        lsPtr[i] = (((l-1)>>(power-i))+1)<<(power-i);
//     
// (2) "(l/2)<<1" is replaced with "(l>>1)<<1".
//
// (3) Mont_dft_OPT2_AS_GENE has been optimized and comments added.
//
///////////////////////////////////////////////////////////////////////////////// 

#include "FMUL.h"

// variables to test the interruption
extern sfixn Interrupted;
extern sfixn PrimeError;

// cuda flag
extern sfixn CUDA_TAG;

// type: exported.
// input: p = k N + 1 (a Fourier Prime), 
//        e = log_2{N}, and want to compute n-th root.
// ouput: the n-th primitive root of unity.
/**
 * EX_GetPrimitiveNthRoot:
 * @e: 
 * @n: equals 2^e and divides p-1
 * @p:  Fourier prime number
 * 
 * Return value: a primtive n-th root of unity
 **/
static sfixn EX_GetPrimitiveNthRoot (sfixn e, sfixn n, sfixn p){
    sfixn k, n2, root=0;
    k = (p-1)>>e;
    n2 = n>>1;
    srand(getSeed());
    while (p>2) {
       root = 1;
       while (root<2) root = rand()%p; 
       root = PowerMod(root, k, p);
       if (PowerMod(root,n2,p) != 1) break;
    }
    return root;	
}

//======================================================================
// Plain univariate polynomial multiplication over Z/pZ.
//======================================================================
// type: local
// note: good for all machine word Fourier Prime.
/**
 * Mont_PlainMul_OPT2_AS_GENE:
 * @degRes: degree of result
 * @resPtr: coefficient vector of result 
 * @degA:  degree of arg 1
 * @APtr:  coefficient vector of  arg 1
 * @degB: degree of arg 2
 * @BPtr: coefficient vector of  arg 2
 * @pPtr: info about the prime number 'p'
 * 
 * Claasical univariate multiplication 
 * in prime characterisitc 'p' (machine word)
 * using the improved Montgommery trick.
 * 
 * Return value: 
 **/
static void Mont_PlainMul_OPT2_AS_GENE (sfixn degRes, sfixn *resPtr, sfixn degA,
            sfixn *APtr, sfixn degB, sfixn *BPtr, MONTP_OPT2_AS_GENE *pPtr){

    sfixn i,j;
    sfixn p=pPtr->P, R=(1L<<pPtr->Rpow)%p, BRsft=pPtr->Base_Rpow, R2=(MulMod(R,R,p))<<BRsft, tmp;

    //printf("before res, A, B\n");
    //printPolyUni(degRes, resPtr);
    //printPolyUni(degA, APtr);
    //printPolyUni(degB, BPtr);
    //fflush(stdout);

    for(i=0; i<=degA; i++) {
        if (!APtr[i]) continue;
        if (APtr[i]==1) {
            for(j=0; j<=degB; j++) resPtr[i+j]=AddMod(BPtr[j], resPtr[i+j], p);
        } else {
            tmp = (MontMulMod_OPT2_AS_GENE(APtr[i],R2,pPtr))<<BRsft;
            for(j=0; j<=degB; j++) {
                //printf("resPtr[%ld]=%ld\n", i+j, resPtr[i+j]);
                resPtr[i+j]=AddMod( MontMulMod_OPT2_AS_GENE(BPtr[j],tmp,pPtr), resPtr[i+j], p);
	        }
        }
    }
}


// type: local
// note: for machine word Fourier Prime in special shape.
/**
 * Mont_PlainMul_OPT2_AS_GENE_SPE:
 * @degRes: degree of result
 * @resPtr: coefficient vector of result 
 * @degA:  degree of arg 1
 * @APtr:  coefficient vector of  arg 1
 * @degB: degree of arg 2
 * @BPtr: coefficient vector of  arg 2
 * @pPtr: info about the prime number 'p'
 * 
 * Claasical univariate multiplication 
 * in prime characterisitc 'p' (machine word)
 * using the improved Montgommery trick.
 * Assume p = c*N+1. Both c,N, need to be power of 2. 
 * 
 * Return value: 
 **/
static void Mont_PlainMul_OPT2_AS_GENE_SPE(sfixn degRes, sfixn *resPtr,
            sfixn degA, sfixn *APtr, sfixn degB, sfixn *BPtr, MONTP_OPT2_AS_GENE *pPtr ){
    sfixn i,j;
    sfixn p=pPtr->P, R=(1L<<pPtr->Rpow)%p, BRsft= pPtr->Base_Rpow, R2=(MulMod(R,R,p))<<BRsft, tmp;

    for(i=0; i<=degA; i++){
        if (!APtr[i]) continue;
        if (APtr[i]==1) {
            for (j=0; j<=degB; j++) resPtr[i+j] = AddMod(BPtr[j], resPtr[i+j], p);
        } else {
            tmp = (MontMulMod_OPT2_AS_GENE_SPE(APtr[i], R2, pPtr))<<BRsft;
	        for (j=0; j<=degB; j++) {
	            resPtr[i+j] = AddMod(MontMulMod_OPT2_AS_GENE_SPE(BPtr[j],tmp,pPtr), resPtr[i+j], p);
	        }
        }
     }
}

// type: exported
// input: A = (degA, APtr), B = (degB, BPtr) and p.
// ouput: (degRes, resPtr) = A*B mod p.
// use: classical univariate polynomial multiplication. 
// note: good for all machine word Fourier Prime.
/**
 * EX_Mont_PlainMul_OPT2_AS_GENE:
 * @degRes: degree of result
 * @resPtr: coefficient vector of result 
 * @degA:  degree of arg 1
 * @APtr:  coefficient vector of  arg 1
 * @degB: degree of arg 2
 * @BPtr: coefficient vector of  arg 2
 * @pPtr: info about the prime number 'p'
 * 
 * Claasical univariate multiplication 
 * in prime characterisitc 'p' (machine word)
 * using the improved Montgommery trick.
 * Will run special if p-1 is a power of 2.
 * 
 * Return value: 
 **/
void EX_Mont_PlainMul_OPT2_AS_GENE (sfixn degRes, sfixn * resPtr,
     sfixn degA, sfixn *APtr, sfixn degB, sfixn *BPtr, MONTP_OPT2_AS_GENE *pPtr) {

    sfixn d1 = degA, d2 = degB;
    d1 = shrinkDegUni(d1, APtr);
    d2 = shrinkDegUni(d2, BPtr);
    if ((pPtr->c_pow==0) || ((pPtr->c_pow+pPtr->Rpow)>=BASE)){
        Mont_PlainMul_OPT2_AS_GENE(degRes, resPtr, d1, APtr, d2, BPtr, pPtr );
    } else {
        Mont_PlainMul_OPT2_AS_GENE_SPE(degRes, resPtr, d1, APtr, d2, BPtr, pPtr );   
    }
}

//======================================================================
// Initialized the prime struct. 
//======================================================================

// type: exported.
// input: p.
// ouput: pPtr.
// use: initialize the prime structure. 
// note: 
/**
 * EX_MontP_Init_OPT2_AS_GENE:
 * @pPtr: pointer to prime number structure for the improved Montgommery trick.
 * @p: prime number.
 * 
 * 
 * Return value: the filled-in prime number structure
 **/
void EX_MontP_Init_OPT2_AS_GENE(MONTP_OPT2_AS_GENE * pPtr, sfixn p){
    sfixn p_1=p-1, RR;
    pPtr->P=p;
    pPtr->Npow=0;
    while (!(p_1%(1L<<pPtr->Npow))){ pPtr->Npow++; }
    pPtr->Npow--;
    pPtr->c=p_1>>pPtr->Npow;
    pPtr->Rpow=logceiling(p);
    pPtr->R_Npow=pPtr->Rpow-pPtr->Npow;
    pPtr->Base_Rpow=BASE-pPtr->Rpow;
    // for Assembly.
    pPtr->Base_Npow=BASE-pPtr->Npow;
    pPtr->c_sft=(pPtr->c)<<(pPtr->Npow);
    pPtr->N2_Rpow=(pPtr->Npow)*2-pPtr->Rpow;
    pPtr->Rmask=(1L<<pPtr->Rpow)-1;
    pPtr->R_Nmask=(1L<<pPtr->R_Npow)-1; 
    pPtr->c_pow=(logceiling(pPtr->c));
    pPtr->c_pow--;
    if(((1L<<pPtr->c_pow)+1)!=pPtr->c) pPtr->c_pow=0;
    pPtr->Max_Root=EX_GetPrimitiveNthRoot(pPtr->Npow, 1L<<(pPtr->Npow), p);
    RR=(1L<<pPtr->Rpow)%p;
    pPtr->R2BRsft=(MulMod(RR,RR,p))<<(pPtr->Base_Rpow);
}

//======================================================================
// Print the prime struct. 
//======================================================================
 
// type: exported.
// input: N/A.
// ouput: PRINTING the prime struct.
// use: 
// note:
/**
 * EX_MontP_Print_OPT2_AS_GENE:
 * @pPtr: a prime number structire
 * 
 * Prints `pPtr`.
 * 
 * Return value: 
 **/
void EX_MontP_Print_OPT2_AS_GENE(MONTP_OPT2_AS_GENE *pPtr){
    printf( "pPtr->P=%ld\n",pPtr->P);
    printf( "pPtr->Npow=%ld\n",pPtr->Npow);  
    printf( "pPtr->c=%ld\n",pPtr->c);
    printf( "pPtr->Rpow=%ld\n",pPtr->Rpow);
    printf( "pPtr->R_Npow=%ld\n",pPtr->R_Npow);
    printf( "pPtr->c_sft=%ld\n",pPtr->c_sft);
    printf( "pPtr->c_pow=%ld\n",pPtr->c_pow);
    printf( "pPtr->Base_Rpow=%ld\n",pPtr->Base_Rpow);// 32-Row on 32-bit machine.
    printf( "pPtr->Base_Npow=%ld\n",pPtr->Base_Npow);// 32-Row on 32-bit machine.
    printf( "pPtr->N2_Rpow=%ld\n",pPtr->N2_Rpow);
    printf( "pPtr->Rmask=%ld\n",pPtr->Rmask);
    printf( "pPtr->R_Nmask=%ld\n",pPtr->R_Nmask);
}


//======================================================================
// get the n-th root of unity. 
//======================================================================
// type: local.
// note: good for all machine word Fourier Prime.
/**
 * Mont_GetNthRoots_OPT2_AS_GENE:
 * @e: 
 * @n: equals 2^e and divides p-1
 * @rootsPtr: (output) an array of size n which contains the powers of the primitive root
 * @pPtr: prime number structire for the prime number p
 * 
 * Return value: returns `rootsPtr`
 **/
static void Mont_GetNthRoots_OPT2_AS_GENE(sfixn e, sfixn n, sfixn * rootsPtr, MONTP_OPT2_AS_GENE * pPtr){
    register sfixn j;
    sfixn root, rootR, R = (1L<<pPtr->Rpow)%pPtr->P, R_2 = MulMod(R, R, pPtr->P), BRsft = pPtr->Base_Rpow;

    if(e > (pPtr->Npow)){
        Interrupted = 1;
        PrimeError = 1;
        return;
    }

    //printf("The input FFT-N:%ld, this p:%ld can handle at most: %ld. !!!\n\n\n\n",n,pPtr->P, (1L<<pPtr->Npow));
    //root=EX_GetPrimitiveNthRoot(e, n, pPtr->P);
    root = PowerMod(pPtr->Max_Root, 1L<<((pPtr->Npow)-e), pPtr->P);
    //printf("%d-th primitive root = %d\n", n, root);
    rootR = MontMulMod_OPT2_AS_GENE(root, R_2<<BRsft, pPtr);
    rootsPtr[0] = R<<BRsft;
    rootsPtr[1] = rootR;
    for(j=2; j<n; j++) {
        rootsPtr[j] = MontMulMod_OPT2_AS_GENE(rootsPtr[j-1], rootR<<BRsft, pPtr);
        rootsPtr[j-1] <<= BRsft;
    }
    rootsPtr[n-1] <<= BRsft;
}

/**
 * Mont_GetNthRoots_OPT2_AS_GENE:
 * @e: 
 * @n: equals 2^e and divides p-1
 * @rootsPtr: (output) an array of size n which contains the powers of the primitive root
 * @pPtr: prime number structire for the prime number p
 * 
 *  Assume p is p=N-1, where N is a power of 2.
 * 
 * Return value: returns `rootsPtr`
 **/
static void Mont_GetNthRoots_OPT2_AS_GENE_SPE(sfixn e, sfixn n, sfixn * rootsPtr, MONTP_OPT2_AS_GENE * pPtr){
    register sfixn j;
    sfixn root, rootR, R=(1L<<pPtr->Rpow)%pPtr->P, R_2=MulMod(R, R, pPtr->P), BRsft=pPtr->Base_Rpow;
    
    if (e > (pPtr->Npow)){
      Interrupted=1; PrimeError=1;
      return;
    }
    
    root=PowerMod(pPtr->Max_Root, 1L<<((pPtr->Npow)-e), pPtr->P);
    rootR=MontMulMod_OPT2_AS_GENE_SPE(root, R_2<<BRsft, pPtr);
    rootsPtr[0]=R<<BRsft;
    rootsPtr[1]=rootR;
    for(j=2; j<n; j++) {
      rootsPtr[j] = MontMulMod_OPT2_AS_GENE_SPE(rootsPtr[j-1], rootR<<BRsft, pPtr);
      rootsPtr[j-1]<<=BRsft;
    }
    rootsPtr[n-1]<<=BRsft;
}


/**
 * EX_Mont_GetNthRoots_OPT2_AS_GENE:
 * @e: 
 * @n: equals 2^e and divides p-1
 * @rootsPtr: (output) an array of size n which contains the powers of the primitive root
 * @pPtr: prime number structire for the prime number p
 * 
 *  Use the special code (for p-1 is a power of 2) if applicable
 *  otherwise use the generic code.
 * 
 * Return value: returns `rootsPtr`
 **/
void EX_Mont_GetNthRoots_OPT2_AS_GENE(sfixn e, sfixn n, sfixn * rootsPtr, MONTP_OPT2_AS_GENE * pPtr){

    if(e > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }
    
    if((pPtr->c_pow==0) || ((pPtr->c_pow+pPtr->Rpow)>=BASE) ){
        Mont_GetNthRoots_OPT2_AS_GENE(e, n, rootsPtr, pPtr);
    }else{
        Mont_GetNthRoots_OPT2_AS_GENE_SPE(e, n, rootsPtr, pPtr); 
    }
}

/**
 * Mont_GetNthRoots_OPT2_AS_GENE_RAND:
 * @e: 
 * @n: equals 2^e and divides p-1
 * @rootsPtr: (output) an array of size n which contains the powers of the primitive root
 * @pPtr: prime number structire for the prime number p
 * 
 *  Probably obselete function
 * 
 * Return value: returns `rootsPtr`
 **/
static void
Mont_GetNthRoots_OPT2_AS_GENE_RAND(sfixn e, sfixn n, sfixn * rootsPtr, MONTP_OPT2_AS_GENE * pPtr){
    register sfixn j;
    sfixn root, rootR, R=(1L<<pPtr->Rpow)%pPtr->P, R_2=MulMod(R, R, pPtr->P), BRsft=pPtr->Base_Rpow;
    
    if(e > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }
    
    root=EX_GetPrimitiveNthRoot(e, n, pPtr->P);
    rootR=MontMulMod_OPT2_AS_GENE(root, R_2<<BRsft, pPtr);
    rootsPtr[0]=R<<BRsft;
    rootsPtr[1]=rootR;
    for(j=2; j<n; j++) {
        rootsPtr[j]= MontMulMod_OPT2_AS_GENE(rootsPtr[j-1], rootR<<BRsft, pPtr);
        rootsPtr[j-1]<<=BRsft;
    }
    rootsPtr[n-1]<<=BRsft;
}

/**
 * Mont_GetNthRoots_OPT2_AS_GENE_SPE_RAND:
 * 
 * 
 * Probably obselete function
 * 
 * Return value: 
 **/
static void
Mont_GetNthRoots_OPT2_AS_GENE_SPE_RAND(sfixn e, sfixn n, sfixn * rootsPtr, MONTP_OPT2_AS_GENE * pPtr){
    register sfixn j;
    sfixn root, rootR, R=(1L<<pPtr->Rpow)%pPtr->P, R_2=MulMod(R, R, pPtr->P), BRsft=pPtr->Base_Rpow;
    if(e>pPtr->Npow)
        printf("The input FFT-N:%ld, this p:%ld can handle at most: %ld. \n",n,pPtr->P, (1L<<pPtr->Npow));
    root=EX_GetPrimitiveNthRoot(e, n, pPtr->P);
    rootR=MontMulMod_OPT2_AS_GENE_SPE(root, R_2<<BRsft, pPtr);
    rootsPtr[0]=R<<BRsft;
    rootsPtr[1]=rootR;
    for(j=2; j<n; j++) {
        rootsPtr[j]= MontMulMod_OPT2_AS_GENE_SPE(rootsPtr[j-1], rootR<<BRsft, pPtr);
        rootsPtr[j-1]<<=BRsft;
    }
    rootsPtr[n-1]<<=BRsft;
}

/**
 * Mont_GetNthRoots_OPT2_AS_GENE_SPE_RAND:
 * 
 * Probably obselete function
 * 
 * Return value: 
 **/
void EX_Mont_GetNthRoots_OPT2_AS_GENE_RAND(sfixn e, sfixn n, sfixn * rootsPtr, MONTP_OPT2_AS_GENE * pPtr){
  if((pPtr->c_pow==0) || ((pPtr->c_pow+pPtr->Rpow)>=BASE)){
    Mont_GetNthRoots_OPT2_AS_GENE_RAND(e, n, rootsPtr, pPtr);
  } else {
    Mont_GetNthRoots_OPT2_AS_GENE_SPE_RAND(e, n, rootsPtr, pPtr); 
  }
}

//==============================================================================
// DFT
//==============================================================================

// type: local.
// note: good for all machine word Fourier Prime.
/**
 * Mont_dft_OPT2_AS_GENE:
 * @n: the FFT size
 * @power: log2 of n
 * @rootsPtr: powers of the n-th primitive root
 * @tmpVecPtr: (output) work space (array of size n)
 * @degA: degree of input polynomial
 * @APtr: coefficient vector of input polynomial
 * @pPtr: prime number structure
 * 
 * NOTE : The algorithm plainly comes from the TFT papers.
 *        One optimization is that it conducts two butterflies in the same team,
 *        where all butterflies share the same primitive root.
 *
 *        A team is a set of butterflies which share the same index i. All
 *        butterflies can be written as
 *
 *        X_{s, im+j}   = X_{s-1, im+j} + w^{[i]_sm} X_{s-1, im+m+j}
 *        X_{s, im+m+j} = X_{s-1, im+j} - w^{[i]_sm} X_{s-1, im+m+j}
 *
 *        where m = n/2^s and s is the level index.
 *
 *        This function is not in-place. APtr[i] remains unchanged, for all i.
 *
 * Return value: DFT of the input polynomial
 **/
static sfixn *Mont_dft_OPT2_AS_GENE (sfixn n, sfixn power, sfixn *rootsPtr, sfixn *tmpVecPtr,
                                     sfixn degA, sfixn *APtr, MONTP_OPT2_AS_GENE *pPtr) {
    sfixn i;
    register sfixn p=pPtr->P;
    register sfixn tmpw;
    sfixn t1,t2;
    sfixn m, halfn=n>>1, s, t, Tno,  start=0, tmp1, tmp2;
    sfixn *lPtr1, *rPtr1, *endPtr;

    if(power > (pPtr->Npow)){
       Interrupted=1; PrimeError=1;
       return tmpVecPtr;
    }
    if(n<2){
       Interrupted=1; PrimeError=1;
       return tmpVecPtr;
    }
    if(power<0) return tmpVecPtr;

    // The top level, s = 1
    t1 = degA - halfn + 1;
    if (t1>=0) {
        for (i=0; i<t1; i++) {
            tmpVecPtr[i] = AddMod(APtr[i], APtr[i+halfn], p);
            tmpVecPtr[i+halfn] = SubMod(APtr[i], APtr[i+halfn], p);
        }
        for (i=t1; i<halfn; i++) {
            tmpVecPtr[i] = tmpVecPtr[i+halfn] = APtr[i];
        }
    } else {
        for (i=0; i<=degA; i++) {
            tmpVecPtr[i] = tmpVecPtr[i+halfn] = APtr[i];
        }
        for (i=degA+1; i<halfn; i++) {
            tmpVecPtr[i] = tmpVecPtr[i+halfn] = 0 ;
        }
    }
    /////////////////////////////////////////////////////////////////
    // s     : the level index
    // tno   : the cardinality of {0, 2, ..., 2^{s}-2} equals 2^{s-1}
    // t     : the index of each team, 0, ..., tno-1
    // m     : 2^(power-s)
    /////////////////////////////////////////////////////////////////
    
    // compute the rest levels.
    m = halfn>>1;
    for (s=2; s<power; s++) {
        // # of teams with "full butterflies";
        Tno = (sfixn)1<<(s-1);
        start=0;
        // middle-loop begin
        for (t=0; t<Tno; t++){
            i=t<<1;
            i=partialBitRev(i,s);
            i*=m;
            tmpw=rootsPtr[i];
            lPtr1=tmpVecPtr+start; rPtr1=lPtr1+m;
            t1=MontMulMod_OPT2_AS_Double_GENE(&t2, rPtr1[0], tmpw, rPtr1[1], tmpw, pPtr);
            endPtr=lPtr1+m-2;

            for(; lPtr1<endPtr; lPtr1+=2, rPtr1+=2){
                rPtr1[0]=SubMod(lPtr1[0],t1,p);
                lPtr1[0]=AddMod(lPtr1[0],t1,p);
                rPtr1[1]=SubMod(lPtr1[1],t2,p);
                lPtr1[1]=AddMod(lPtr1[1],t2,p);
                t1=MontMulMod_OPT2_AS_Double_GENE(&t2, rPtr1[2], tmpw, rPtr1[3], tmpw, pPtr);
            } // inner-loop end.
            tmp1=lPtr1[0];
            tmp2=lPtr1[1];
            rPtr1[0]=SubMod(tmp1,t1,p);
            rPtr1[1]=SubMod(tmp2,t2,p);
            lPtr1[0]=AddMod(tmp1,t1,p);
            lPtr1[1]=AddMod(tmp2,t2,p);
            start+=(m<<1);
        } // middle-loop end.
        m>>=1;
    }

    if(power>1){
        for (i=0; i<n; i+=2) {
            tmpw=rootsPtr[partialBitRev(i,power)];
            t1=MontMulMod_OPT2_AS_GENE(tmpVecPtr[i+1],tmpw, pPtr);     
            tmp1=tmpVecPtr[i];
            tmpVecPtr[i]=AddMod(tmp1,t1, p);
            tmpVecPtr[i+1]=SubMod(tmp1,t1,p);
        }
    }
    return tmpVecPtr; 
}


// type: local.
// note: good for machine word Fourier Prime in special shape.
/**
 * Mont_dft_OPT2_AS_GENE_SPE:
 * @n: the FFT size
 * @power: log2 of n
 * @rootsPtr: powers of the n-th primitive root
 * @tmpVecPtr: (output) work space (int array of size n)
 * @degA: degree of input polynomial
 * @APtr: coefficient vector of input polynomial
 * @pPtr: prime number structure
 * 
 * Assume p-1 is a power of 2
 * 
 * Return value: DFT of the input polynomial
 **/
static sfixn *Mont_dft_OPT2_AS_GENE_SPE ( sfixn n, sfixn power, sfixn *rootsPtr,
             sfixn *tmpVecPtr,  sfixn degA, sfixn *APtr, MONTP_OPT2_AS_GENE *pPtr) {

    sfixn i;
    register sfixn  p=pPtr->P;
    sfixn t1, t2;
    sfixn m, halfn=n>>1, s, t, Tno,  start=0, tmp1, tmp2, tmpw;
    sfixn *lPtr1, *rPtr1, *endPtr;
    
    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return tmpVecPtr;
    }
    if(n<2){
        Interrupted=1; PrimeError=1;
        return tmpVecPtr;
    }
    if(power<0) return tmpVecPtr;

    cleanVec(n-1, tmpVecPtr);
    // DFTs begin=========================================>
    // compute the top level. (level-1 to level-2).
    if((degA+1)>halfn){
        tmp1=degA-halfn+1;
        for (i=0; i<tmp1; i++) {
            tmpVecPtr[i]=AddMod(APtr[i],APtr[i+halfn], p);
            tmpVecPtr[i+halfn]=SubMod(APtr[i],APtr[i+halfn], p);
        }
        for (i=tmp1;i<halfn;i++) tmpVecPtr[i]=tmpVecPtr[i+halfn]=APtr[i];
    } else {
        for(i=0; i<=degA;i++) tmpVecPtr[i]=tmpVecPtr[i+halfn]=APtr[i];
    }

    // compute the rest levels.
    m=halfn>>1;
    for (s=2; s<power; s++){
        // # of teams with "full butterflies";
        Tno = (sfixn)1<<(s-1);
        start=0;

        // middle-loop begin
        for (t=0; t<Tno; t++){

            tmpw=rootsPtr[partialBitRev(t<<1,s)*m];
            lPtr1=tmpVecPtr+start; rPtr1=lPtr1+m;
            t1=MontMulMod_OPT2_AS_Double_GENE_SPE(&t2, rPtr1[0], tmpw, rPtr1[1], tmpw, pPtr);
            // inner-loop begin
            endPtr=lPtr1+m-2;
            for(; lPtr1<endPtr; lPtr1+=2, rPtr1+=2){
                rPtr1[0]=SubMod(lPtr1[0],t1,p);
                lPtr1[0]=AddMod(lPtr1[0],t1,p);
                rPtr1[1]=SubMod(lPtr1[1],t2,p);
                lPtr1[1]=AddMod(lPtr1[1],t2,p);
                t1=MontMulMod_OPT2_AS_Double_GENE_SPE(&t2, rPtr1[2],tmpw, rPtr1[3], tmpw, pPtr);
            } // inner-loop end.

            tmp1=lPtr1[0];
            tmp2=lPtr1[1];
            rPtr1[0]=SubMod(tmp1,t1,p);
            rPtr1[1]=SubMod(tmp2,t2,p);
            lPtr1[0]=AddMod(tmp1,t1,p);
            lPtr1[1]=AddMod(tmp2,t2,p);
            start+=(m<<1);
        } // middle-loop end.
        m>>=1;
    }

    if(power>1){
        for (i=0; i<n; i+=2) {
            tmpw=rootsPtr[partialBitRev(i,power)];
            t1=MontMulMod_OPT2_AS_GENE_SPE(tmpVecPtr[i+1],tmpw, pPtr);     
            tmp1=tmpVecPtr[i];
            tmpVecPtr[i]=AddMod(tmp1,t1, p);
            tmpVecPtr[i+1]=SubMod(tmp1,t1,p);
        }
    }
    return tmpVecPtr; 
}



// type: exported.
// note: pairup with EX_Mont_INVDFT_OPT2_AS_GENE_R.
/**
 * EX_Mont_DFT_OPT2_AS_GENE:
 * @n: the FFT size
 * @power: log2 of n
 * @rootsPtr: powers of the n-th primitive root
 * @tmpVecPtr: (output) work space (int array of size n)
 * @degA: degree of input polynomial
 * @APtr: coefficient vector of input polynomial
 * @pPtr: prime number structure
 * 
 * Exported version which can use the special case
 * (p-1 is a power of 2) if applicable.
 * 
 * Return value: DFT of the input polynomial
 **/
void EX_Mont_DFT_OPT2_AS_GENE ( sfixn n, sfixn power, sfixn * rootsPtr,
     sfixn * tmpVecPtr, sfixn degA, sfixn * APtr, MONTP_OPT2_AS_GENE * pPtr){

    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }
    if((pPtr->c_pow==0) || ((pPtr->c_pow+pPtr->Rpow)>=BASE) ){
        Mont_dft_OPT2_AS_GENE ( n, power, rootsPtr, tmpVecPtr, degA, APtr, pPtr);
    } else {
        Mont_dft_OPT2_AS_GENE_SPE ( n, power, rootsPtr, tmpVecPtr,  degA, APtr, pPtr);    
    }
}

// type: local, in-place.
// note: good for all machine word Fourier Prime.
/**
 * Mont_dft_OPT2_AS_GENE_1:
 * @n: the FFT size
 * @power: log2 of n
 * @rootsPtr: powers of the n-th primitive root
 * @tmpVecPtr: (output) work space (int array of size n)
 * @pPtr: prime number structure
 * 
 * DFT in place. The input coefficient array is  `tmpVecPtr`.
 * 
 * 
 * Return value: DFT of the input polynomial
 **/
static void Mont_DFT_OPT2_AS_GENE_1 ( sfixn n, sfixn power, sfixn *rootsPtr,
            sfixn *tmpVecPtr, MONTP_OPT2_AS_GENE *pPtr) {

    sfixn i;
    register sfixn p=pPtr->P;
    sfixn t1,t2;
    sfixn m, halfn=n>>1, s, t, Tno,  start=0, tmp1, tmp2, tmpw;
    sfixn  *lPtr1, *rPtr1, *endPtr;

    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }
    if(n<2){
        Interrupted=1; PrimeError=1;
        return;
    }
    if(power<0) return;
    // DFTs begin=========================================>
    // compute the top level. (level-1 to level-2).
    for (i=0; i<halfn; i++) {
        tmp1=tmpVecPtr[i];
        tmpVecPtr[i]=AddMod(tmpVecPtr[i],tmpVecPtr[i+halfn], pPtr->P);
        tmpVecPtr[i+halfn]=SubMod(tmp1,tmpVecPtr[i+halfn],pPtr->P);
    }

    // compute the rest levels.
    m=halfn>>1;
    for (s=2; s<power; s++){
        // # of teams with "full butterflies";
        Tno=n/(m<<1);
        start=0;
        // middle-loop begin
        for (t=0; t<Tno; t++){
            tmpw=rootsPtr[partialBitRev(t<<1,s)*m];
            lPtr1=tmpVecPtr+start; rPtr1=lPtr1+m;
            t1=MontMulMod_OPT2_AS_Double_GENE(&t2, rPtr1[0], tmpw, rPtr1[1],tmpw, pPtr);
            // inner-loop begin
            endPtr=lPtr1+m-2;

            for(; lPtr1<endPtr; lPtr1+=2, rPtr1+=2){
                rPtr1[0]=SubMod(lPtr1[0],t1,p);
                lPtr1[0]=AddMod(lPtr1[0],t1,p);

                rPtr1[1]=SubMod(lPtr1[1],t2,p);
                lPtr1[1]=AddMod(lPtr1[1],t2,p);

                t1=MontMulMod_OPT2_AS_Double_GENE(&t2, rPtr1[2],tmpw, rPtr1[3], tmpw, pPtr);
            } // inner-loop end.
            tmp1=lPtr1[0];
            tmp2=lPtr1[1];
            rPtr1[0]=SubMod(tmp1,t1,p);
            rPtr1[1]=SubMod(tmp2,t2,p);
            lPtr1[0]=AddMod(tmp1,t1,p);
            lPtr1[1]=AddMod(tmp2,t2,p);
            start+=(m<<1);
        } // middle-loop end.
        m>>=1;
    }
    if(power>1){
        for (i=0; i<n; i+=2) {
            tmpw=rootsPtr[partialBitRev(i,power)];
            t1=MontMulMod_OPT2_AS_GENE(tmpVecPtr[i+1],tmpw, pPtr);     
            tmp1=tmpVecPtr[i];
            tmpVecPtr[i]=AddMod(tmp1,t1, p);
            tmpVecPtr[i+1]=SubMod(tmp1,t1,p);
        }
    }
}

// type: local, in-place.
// note: good for machine word Fourier Prime in special shape.
/**
 * Mont_dft_OPT2_AS_GENE_SPE_1:
 * @n: the FFT size
 * @power: log2 of n
 * @rootsPtr: powers of the n-th primitive root
 * @tmpVecPtr: (output) work space (int array of size n)
 * @pPtr: prime number structure
 * 
 * DFT in place. The input coefficient array is  `tmpVecPtr`.
 *  Assume p-1 is a power of 2
 * 
 * Return value: DFT of the input polynomial
 **/

static void Mont_DFT_OPT2_AS_GENE_SPE_1 ( sfixn n, sfixn power, sfixn * rootsPtr,
            sfixn * tmpVecPtr, MONTP_OPT2_AS_GENE * pPtr) {

    sfixn i;
    register sfixn p=pPtr->P;
    sfixn t1,t2;
    sfixn m, halfn=n>>1, s, t, Tno,  start=0, tmp1, tmp2, tmpw;
    sfixn  *lPtr1, *rPtr1, *endPtr;

    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }
    if(n<2){
        Interrupted=1; PrimeError=1;
        return;
    } 
    if(power<0) return ;
  
    // DFTs begin=========================================>
    // compute the top level. (level-1 to level-2).
    for (i=0; i<halfn; i++) {
        tmp1=tmpVecPtr[i];
        tmpVecPtr[i]=AddMod(tmpVecPtr[i],tmpVecPtr[i+halfn], pPtr->P);
        tmpVecPtr[i+halfn]=SubMod(tmp1,tmpVecPtr[i+halfn],pPtr->P);
    }

    // compute the rest levels.
    m=halfn>>1;
    for (s=2; s<power; s++){
        // # of teams with "full butterflies";
        Tno=n/(m<<1);
        start=0;
        // middle-loop begin
        for (t=0; t<Tno; t++){
            tmpw=rootsPtr[partialBitRev(t<<1,s)*m];
            lPtr1=tmpVecPtr+start; rPtr1=lPtr1+m;
            t1=MontMulMod_OPT2_AS_Double_GENE_SPE(&t2, rPtr1[0], tmpw, rPtr1[1],tmpw, pPtr);
            // inner-loop begin
            endPtr=lPtr1+m-2;
            for(; lPtr1<endPtr; lPtr1+=2, rPtr1+=2){
              rPtr1[0]=SubMod(lPtr1[0],t1,p);
              lPtr1[0]=AddMod(lPtr1[0],t1,p);
              rPtr1[1]=SubMod(lPtr1[1],t2,p);
              lPtr1[1]=AddMod(lPtr1[1],t2,p);
              t1=MontMulMod_OPT2_AS_Double_GENE_SPE(&t2, rPtr1[2],tmpw, rPtr1[3], tmpw, pPtr);
            } // inner-loop end.

            tmp1=lPtr1[0];
            tmp2=lPtr1[1];
            rPtr1[0]=SubMod(tmp1,t1,p);
            rPtr1[1]=SubMod(tmp2,t2,p);
            lPtr1[0]=AddMod(tmp1,t1,p);
            lPtr1[1]=AddMod(tmp2,t2,p);
            start+=(m<<1);
        } // middle-loop end.
        m>>=1;
    }

    if(power>1){
        for (i=0; i<n; i+=2) {
         tmpw=rootsPtr[partialBitRev(i,power)];
         t1=MontMulMod_OPT2_AS_GENE_SPE(tmpVecPtr[i+1],tmpw, pPtr);     
         tmp1=tmpVecPtr[i];
         tmpVecPtr[i]=AddMod(tmp1,t1, p);
         tmpVecPtr[i+1]=SubMod(tmp1,t1,p);
        }
    }
}

// type: exported
/**
 * EX_Mont_DFT_OPT2_AS_GENE_1:
 * @n: the FFT size
 * @power: log2 of n
 * @rootsPtr: powers of the n-th primitive root
 * @tmpVecPtr: (output) work space (int array of size n)
 * @pPtr: prime number structure
 * 
 * DFT in place. Exported version. 
 *  Use the special code (for p-1 is a power of 2) if applicable
 *  otherwise use the generic code.
 * 
 * Return value: DFT of the input polynomial
 **/

void EX_Mont_DFT_OPT2_AS_GENE_1 ( sfixn n, sfixn power, sfixn * rootsPtr,
     sfixn * tmpVecPtr, MONTP_OPT2_AS_GENE * pPtr ){

    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }

    if((pPtr->c_pow==0) || ((pPtr->c_pow+pPtr->Rpow)>=BASE)){
        Mont_DFT_OPT2_AS_GENE_1 ( n, power, rootsPtr, tmpVecPtr, pPtr);
    }
    else{
        Mont_DFT_OPT2_AS_GENE_SPE_1 (n, power, rootsPtr, tmpVecPtr, pPtr);
    }
}

//==============================================================================
// INV DFT
//==============================================================================
// type: local.
// note: good for all machine word Fourier Prime.
/**
 * Mont_invdft_OPT2_AS_GENE:
 * @n: the FFT size
 * @power: log2 of n
 * @rootsPtr: powers of the n-th primitive root
 * @tmpVecPtr:  work space (int array of size n)
 * @degRes: degree of result
 * @ResPtr: (output) coefficient vector of result
 * @pPtr: prime number structure
 * 
 * Inverse DFT. 
 * 
 * Return value: R multiply the inverse DFT  of the input polynomial,
 * where R = next power of 2 of the prime number.
 **/
static sfixn * Mont_invdft_OPT2_AS_GENE (sfixn n, sfixn power, sfixn * rootsPtr, sfixn *tmpVecPtr,
               sfixn degRes, sfixn *ResPtr, MONTP_OPT2_AS_GENE *pPtr) {

    register sfixn i, p=pPtr->P;
    // l is the size of result poly. 
    sfixn m, halfn=n>>1, s, u, t, invn, start=0, tmp1,tmp2, tmp, BRsft=pPtr->Base_Rpow;
    sfixn * wPtr, * lPtr, * rPtr;
    sfixn R=(1L<<(pPtr->Rpow))%p;
    
    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return tmpVecPtr;
    }

    if(n<2){
       //printf("n<2 won't work for FFT\n");
       //fflush(stdout);
       Interrupted=1; PrimeError=1;
       return tmpVecPtr;
    }

    if (power==0) return tmpVecPtr;
    wPtr=(sfixn *)my_calloc(halfn, sizeof(sfixn));
    // lsPtr[0] not in use.
    // lsPtr[i] keeps the number of points needed to compute at level i for poly1/2 DFT tree.
    // the bottom level has one extra to compute, but leave it, won't hurt.
    m=halfn;
    for(i=0; i<n; i+=2){
      tmp=tmpVecPtr[i];
      tmpVecPtr[i]=AddMod(tmpVecPtr[i], tmpVecPtr[i+1], p);
      tmpVecPtr[i+1]=SubMod(tmp, tmpVecPtr[i+1], p);
    }
    m=2;
    wPtr[0]=R<<pPtr->Base_Rpow;

    for(s=power-1; s>1; s--){
        for(i=1; i<m; i++) wPtr[i]=(rootsPtr[n-(i<<(s-1))]); 
        start=0;   
        for(t=0; t<(n/(m<<1)); t++){

            lPtr=tmpVecPtr+start;
            rPtr=lPtr+m;
            for(u=0; u<m; u+=2){
                tmp1=MontMulMod_OPT2_AS_Double_GENE(&tmp2,rPtr[u],wPtr[u],rPtr[u+1],wPtr[u+1], pPtr);
                rPtr[u]=SubMod(lPtr[u],tmp1,p);
                lPtr[u]=AddMod(lPtr[u],tmp1,p);
                rPtr[u+1]=SubMod(lPtr[u+1],tmp2,p);
                lPtr[u+1]=AddMod(lPtr[u+1],tmp2,p);
            }
            start+=m<<1;
        }
        m<<=1;
    }

    if(power>1){
        lPtr=tmpVecPtr;
        rPtr=lPtr+halfn;
        tmp1=MontMulMod_OPT2_AS_Double_GENE(&tmp2,rPtr[0],rootsPtr[0], rPtr[1], rootsPtr[n-1], pPtr);
        rPtr[0]=SubMod(lPtr[0],tmp1,p);
        lPtr[0]=AddMod(lPtr[0],tmp1,p);
        rPtr[1]=SubMod(lPtr[1],tmp2,p);
        lPtr[1]=AddMod(lPtr[1],tmp2,p);

        for(u=2; u<halfn; u+=2){
            tmp1=MontMulMod_OPT2_AS_Double_GENE(&tmp2,rPtr[u],rootsPtr[n-u], rPtr[u+1],rootsPtr[n-u-1], pPtr);
            rPtr[u]=SubMod(lPtr[u],tmp1,p);
            lPtr[u]=AddMod(lPtr[u],tmp1,p);
            rPtr[u+1]=SubMod(lPtr[u+1],tmp2,p);
            lPtr[u+1]=AddMod(lPtr[u+1],tmp2,p);
        }
    }

    invn=inverseMod(n,p);
    invn=MulMod(R,invn,p);
    invn<<=BRsft;
    for(i=0; i<=degRes; i++) ResPtr[i]=MontMulMod_OPT2_AS_GENE(tmpVecPtr[i], invn, pPtr);
    my_free(wPtr);
    return(tmpVecPtr);
}

// type: local.
// note: good for machine word Fourier Prime in special shape.
/**
 * Mont_invdft_OPT2_AS_SPE:
 * @n: the FFT size
 * @power: log2 of n
 * @rootsPtr: powers of the n-th primitive root
 * @tmpVecPtr:  work space (int array of size n)
 * @degRes: degree of result
 * @ResPtr: (output) coefficient vector of result
 * @pPtr: prime number structure
 * 
 * Inverse DFT.  Case where p-1 is a power of 2.
 * 
 * Return value:  R multiply the  Inverse DFT  of the input polynomial, where R = next power of 2 of the prime number.
 **/
static sfixn * Mont_invdft_OPT2_AS_GENE_SPE ( sfixn n, sfixn power, sfixn *rootsPtr, sfixn *tmpVecPtr,
                                              sfixn degRes, sfixn *ResPtr, MONTP_OPT2_AS_GENE *pPtr) {

    register sfixn i, p=pPtr->P;
    // l is the size of result poly. 
    sfixn m, halfn=n>>1, s, u, t, invn, start=0, tmp1,tmp2, tmp, BRsft=pPtr->Base_Rpow;
    sfixn R=(1L<<(pPtr->Rpow))%p;
    sfixn *wPtr, *lPtr, *rPtr;

    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return tmpVecPtr;
    }
    if(n<2){
        Interrupted=1; PrimeError=1;
        return tmpVecPtr;
    }
    
    if (power==0) return tmpVecPtr;
    wPtr=(sfixn *)my_calloc(halfn, sizeof(sfixn));
    // lsPtr[0] not in use.
    
    for(i=0; i<n; i+=2){
        tmp=tmpVecPtr[i];
        tmpVecPtr[i]=AddMod(tmpVecPtr[i], tmpVecPtr[i+1], p);
        tmpVecPtr[i+1]=SubMod(tmp, tmpVecPtr[i+1], p);
    }
    m=2;
    wPtr[0]=R<<pPtr->Base_Rpow;
    for(s=power-1; s>1; s--){
        for(i=1; i<m; i++) wPtr[i]=(rootsPtr[n-(i<<(s-1))]); 
        start=0;   
        for(t=0; t<(n/(m<<1)); t++) {
            lPtr=tmpVecPtr+start;
            rPtr=lPtr+m;
            for(u=0; u<m; u+=2){
                tmp1=MontMulMod_OPT2_AS_Double_GENE_SPE(&tmp2,rPtr[u],wPtr[u],rPtr[u+1],wPtr[u+1], pPtr);
                rPtr[u]=SubMod(lPtr[u],tmp1,p);
                lPtr[u]=AddMod(lPtr[u],tmp1,p);
                rPtr[u+1]=SubMod(lPtr[u+1],tmp2,p);
                lPtr[u+1]=AddMod(lPtr[u+1],tmp2,p);
            }
            start+=m<<1;
        }
        m<<=1;
    }

    if(power>1){
        lPtr=tmpVecPtr;
        rPtr=lPtr+halfn;   
        tmp1=MontMulMod_OPT2_AS_Double_GENE_SPE(&tmp2,rPtr[0],rootsPtr[0],rPtr[1],rootsPtr[n-1], pPtr);
        rPtr[0]=SubMod(lPtr[0],tmp1,p);
        lPtr[0]=AddMod(lPtr[0],tmp1,p);
        rPtr[1]=SubMod(lPtr[1],tmp2,p);
        lPtr[1]=AddMod(lPtr[1],tmp2,p);
        for(u=2; u<halfn; u+=2){
            tmp1=MontMulMod_OPT2_AS_Double_GENE_SPE(&tmp2,rPtr[u],rootsPtr[n-u],rPtr[u+1],rootsPtr[n-u-1], pPtr);
            rPtr[u]=SubMod(lPtr[u],tmp1,p);
            lPtr[u]=AddMod(lPtr[u],tmp1,p);
            rPtr[u+1]=SubMod(lPtr[u+1],tmp2,p);
            lPtr[u+1]=AddMod(lPtr[u+1],tmp2,p);
        }
    }
    invn=inverseMod(n,p);
    invn=MulMod(R,invn,p);
    invn<<=BRsft;
    for(i=0; i<=degRes; i++) ResPtr[i]=MontMulMod_OPT2_AS_GENE_SPE(tmpVecPtr[i], invn, pPtr);
    my_free(wPtr);

    return(tmpVecPtr);
}



// type: exported.
// note: pair-up with EX_Mont_INVDFT_OPT2_AS_GENE_R.
/**
 * EX_Mont_INVDFT_OPT2_AS_GENE:
 * @n: the FFT size
 * @power: log2 of n
 * @rootsPtr: powers of the n-th primitive root
 * @tmpVecPtr:  work space (int array of size n)
 * @degRes: degree of result
 * @ResPtr: (output) coefficient vector of result
 * @pPtr: prime number structure
 * 
 * Inverse DFT. Exported function. Hanfles p-1 is a power of 2.
 * 
 * 
 * Return value: Inverse DFT of the input polynomial. Big R issue not handled here.
 **/
void EX_Mont_INVDFT_OPT2_AS_GENE (sfixn n, sfixn power, sfixn *rootsPtr, sfixn *tmpVecPtr,
                                  sfixn degRes, sfixn *ResPtr, MONTP_OPT2_AS_GENE *pPtr){

    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }

    if((pPtr->c_pow==0) || ((pPtr->c_pow+pPtr->Rpow)>=BASE) ){
        Mont_invdft_OPT2_AS_GENE (n, power, rootsPtr, tmpVecPtr, degRes, ResPtr, pPtr);
    } else {
        Mont_invdft_OPT2_AS_GENE_SPE (n, power, rootsPtr, tmpVecPtr, degRes, ResPtr, pPtr);
    }
}

// type: local.
// note: good for all machine word Fourier Prime.
//       remove the extra R from DFT at the end.
/**
 * Mont_invdft_OPT2_AS_GENE_R:
 * @n: the FFT size
 * @power: log2 of n
 * @rootsPtr: powers of the n-th primitive root
 * @tmpVecPtr:  work space (int array of size n)
 * @degRes: degree of result
 * @ResPtr: (output) coefficient vector of result
 * @pPtr: prime number structure
 * 
 * Inverse DFT. 
 * 
 * 
 * Return value: Inverse DFT  of the input polynomial.
 **/
static sfixn *Mont_invdft_OPT2_AS_GENE_R (sfixn n, sfixn power, sfixn *rootsPtr, sfixn *tmpVecPtr,
                                          sfixn degRes, sfixn *ResPtr, MONTP_OPT2_AS_GENE *pPtr) {

    register sfixn i, p=pPtr->P;
    // l is the size of result poly. 
    sfixn m, halfn=n>>1, s, u, t, invn, start=0, tmp1,tmp2, tmp, BRsft=pPtr->Base_Rpow;
    sfixn R=(1L<<(pPtr->Rpow))%p; 
    sfixn *wPtr, *lPtr, *rPtr;
    
    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return tmpVecPtr;
    }
    if(n<2){
        Interrupted=1; PrimeError=1;
        return tmpVecPtr;
    }
    if (power==0) return tmpVecPtr;
    wPtr=(sfixn *)my_calloc(halfn, sizeof(sfixn));

    for(i=0; i<n; i+=2){
        tmp=tmpVecPtr[i];
        tmpVecPtr[i]=AddMod(tmpVecPtr[i], tmpVecPtr[i+1], p);
        tmpVecPtr[i+1]=SubMod(tmp, tmpVecPtr[i+1], p);
    }
    m=2;
    wPtr[0]=R<<pPtr->Base_Rpow;

    for(s=power-1; s>1; s--){
        for(i=1; i<m; i++) wPtr[i]=(rootsPtr[n-(i<<(s-1))]); 
        start=0;   
        for(t=0; t<(n/(m<<1)); t++){
            lPtr=tmpVecPtr+start;
            rPtr=lPtr+m;
            for(u=0; u<m; u+=2){
                tmp1=MontMulMod_OPT2_AS_Double_GENE(&tmp2,rPtr[u],wPtr[u],rPtr[u+1],wPtr[u+1], pPtr);
                rPtr[u]=SubMod(lPtr[u],tmp1,p);
                lPtr[u]=AddMod(lPtr[u],tmp1,p);
                rPtr[u+1]=SubMod(lPtr[u+1],tmp2,p);
                lPtr[u+1]=AddMod(lPtr[u+1],tmp2,p);
            }
            start+=m<<1;
        }
        m<<=1;
    }

    if(power>1){
        lPtr=tmpVecPtr;
        rPtr=lPtr+halfn;
        tmp1=MontMulMod_OPT2_AS_Double_GENE(&tmp2,rPtr[0],rootsPtr[0],rPtr[1],rootsPtr[n-1], pPtr);
        rPtr[0]=SubMod(lPtr[0],tmp1,p);
        lPtr[0]=AddMod(lPtr[0],tmp1,p);
        rPtr[1]=SubMod(lPtr[1],tmp2,p);
        lPtr[1]=AddMod(lPtr[1],tmp2,p);
        for(u=2; u<halfn; u+=2){
            tmp1=MontMulMod_OPT2_AS_Double_GENE(&tmp2,rPtr[u],rootsPtr[n-u],rPtr[u+1],rootsPtr[n-u-1], pPtr);
            rPtr[u]=SubMod(lPtr[u],tmp1,p);
            lPtr[u]=AddMod(lPtr[u],tmp1,p);
            rPtr[u+1]=SubMod(lPtr[u+1],tmp2,p);
            lPtr[u+1]=AddMod(lPtr[u+1],tmp2,p);
        }
    }
    invn=inverseMod(n,p);
    invn=MulMod(R,invn,p);
    invn=MulMod(R,invn,p);
    invn<<=BRsft;
    for(i=0; i<=degRes; i++) ResPtr[i]=MontMulMod_OPT2_AS_GENE(tmpVecPtr[i], invn, pPtr);
    my_free(wPtr);
    return(tmpVecPtr);
}

// type: local.
// note: good for machine word Fourier Prime in special shape.
//       remove the extra R from DFT at the end.
/**
 * Mont_invdft_OPT2_AS_SPE_R:
 * @n: the FFT size
 * @power: log2 of n
 * @rootsPtr: powers of the n-th primitive root
 * @tmpVecPtr:  work space (int array of size n)
 * @degRes: degree of result
 * @ResPtr: (output) coefficient vector of result
 * @pPtr: prime number structure
 * 
 * Inverse DFT.  Case where p-1 is a power of 2.
 * 
 * 
 * Return value: Inverse DFT.
 **/
static sfixn *Mont_invdft_OPT2_AS_GENE_SPE_R (sfixn n, sfixn power, sfixn *rootsPtr, sfixn *tmpVecPtr,
                                              sfixn degRes, sfixn *ResPtr, MONTP_OPT2_AS_GENE *pPtr)
{

    register sfixn i, p=pPtr->P;
    // l is the size of result poly. 
    sfixn m, halfn=n>>1, s, u, t, invn, start=0, tmp1,tmp2, tmp, BRsft=pPtr->Base_Rpow;
    sfixn R=(1L<<(pPtr->Rpow))%p; 
    sfixn *wPtr, *lPtr, *rPtr;
    
    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return tmpVecPtr;
    }
    if (power==0) return tmpVecPtr;

    wPtr=(sfixn *)my_calloc(halfn, sizeof(sfixn));
    for(i=0; i<n; i+=2){
         tmp=tmpVecPtr[i];
         tmpVecPtr[i]=AddMod(tmpVecPtr[i], tmpVecPtr[i+1], p);
         tmpVecPtr[i+1]=SubMod(tmp, tmpVecPtr[i+1], p);
    }
    m=2;
    wPtr[0]=R<<pPtr->Base_Rpow;
    for(s=power-1; s>1; s--){
        for(i=1; i<m; i++) wPtr[i]=(rootsPtr[n-(i<<(s-1))]); 
        start=0;   
        for(t=0; t<(n/(m<<1)); t++){
            lPtr=tmpVecPtr+start;
            rPtr=lPtr+m;
            for(u=0; u<m; u+=2){
                tmp1=MontMulMod_OPT2_AS_Double_GENE_SPE(&tmp2,rPtr[u],wPtr[u],rPtr[u+1],wPtr[u+1], pPtr);
                rPtr[u]=SubMod(lPtr[u],tmp1,p);
                lPtr[u]=AddMod(lPtr[u],tmp1,p);
                rPtr[u+1]=SubMod(lPtr[u+1],tmp2,p);
                lPtr[u+1]=AddMod(lPtr[u+1],tmp2,p);
            }
            start+=m<<1;
        }
        m<<=1;
    }

    if(power>1){
        lPtr=tmpVecPtr;
        rPtr=lPtr+halfn;
        tmp1=MontMulMod_OPT2_AS_Double_GENE_SPE(&tmp2,rPtr[0],rootsPtr[0],rPtr[1],rootsPtr[n-1], pPtr);
        rPtr[0]=SubMod(lPtr[0],tmp1,p);
        lPtr[0]=AddMod(lPtr[0],tmp1,p);
        rPtr[1]=SubMod(lPtr[1],tmp2,p);
        lPtr[1]=AddMod(lPtr[1],tmp2,p);
        for(u=2; u<halfn; u+=2){
            tmp1=MontMulMod_OPT2_AS_Double_GENE_SPE(&tmp2,rPtr[u],rootsPtr[n-u],rPtr[u+1],rootsPtr[n-u-1], pPtr);
            rPtr[u]=SubMod(lPtr[u],tmp1,p);
            lPtr[u]=AddMod(lPtr[u],tmp1,p);
            rPtr[u+1]=SubMod(lPtr[u+1],tmp2,p);
            lPtr[u+1]=AddMod(lPtr[u+1],tmp2,p);
        }
    }
    invn=inverseMod(n,p);
    invn=MulMod(R,invn,p);
    invn=MulMod(R,invn,p);
    invn<<=BRsft;
    for(i=0; i<=degRes; i++) ResPtr[i]=MontMulMod_OPT2_AS_GENE_SPE(tmpVecPtr[i], invn, pPtr);
    my_free(wPtr);
    return(tmpVecPtr);
}

// type: exported.
// note: pair-up with EX_Mont_DFT_OPT2_AS_GENE.
/**
 * EX_Mont_INVDFT_OPT2_AS_GENE_R:
 * @n: the FFT size
 * @power: log2 of n
 * @rootsPtr: powers of the n-th primitive root
 * @tmpVecPtr:  work space (int array of size n)
 * @degRes: degree of result
 * @ResPtr: (output) coefficient vector of result
 * @pPtr: prime number structure
 * 
 * Inverse DFT. Exported function. Hanfles p-1 is a power of 2.
 * 
 * 
 * Return value: Inverse DFT of the input polynomial.
 **/
void EX_Mont_INVDFT_OPT2_AS_GENE_R (sfixn n, sfixn power, sfixn * rootsPtr, sfixn * tmpVecPtr,
                                    sfixn degRes, sfixn * ResPtr, MONTP_OPT2_AS_GENE * pPtr ){

    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }

    if((pPtr->c_pow==0) || ((pPtr->c_pow+pPtr->Rpow)>=BASE) ){
        Mont_invdft_OPT2_AS_GENE_R ( n, power, rootsPtr, tmpVecPtr, degRes, ResPtr, pPtr);
    } else{
        Mont_invdft_OPT2_AS_GENE_SPE_R ( n, power, rootsPtr, tmpVecPtr, degRes, ResPtr, pPtr);
    }
}

// type: local, in-place.
// note: good for all machine word Fourier Prime.
static void Mont_INVDFT_OPT2_AS_GENE_1 (sfixn n, sfixn power, sfixn * rootsPtr,
            sfixn * tmpVecPtr, MONTP_OPT2_AS_GENE * pPtr) {

    register sfixn i, p=pPtr->P;
    // l is the size of result poly. 
    sfixn m, halfn=n>>1, s, u, t, invn, start=0, tmp1,tmp2, tmp, BRsft=pPtr->Base_Rpow;
    sfixn * wPtr, * lPtr, * rPtr;
    sfixn R=(1L<<(pPtr->Rpow))%p; 

    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }
    if(n<2){
        Interrupted=1; PrimeError=1;
        return;
    }
    if (power==0) return;
    wPtr=(sfixn *)my_calloc(halfn, sizeof(sfixn));

    for(i=0; i<n; i+=2){
        tmp=tmpVecPtr[i];
        tmpVecPtr[i]=AddMod(tmpVecPtr[i], tmpVecPtr[i+1], p);
        tmpVecPtr[i+1]=SubMod(tmp, tmpVecPtr[i+1], p);
    }
    m=2;
    wPtr[0]=R<<pPtr->Base_Rpow;

    for(s=power-1; s>1; s--){
        for(i=1; i<m; i++) wPtr[i]=(rootsPtr[n-(i<<(s-1))]); 
        start=0;   
        for(t=0; t<(n/(m<<1)); t++){
            lPtr=tmpVecPtr+start;
            rPtr=lPtr+m;
            for(u=0; u<m; u+=2){
              tmp1=MontMulMod_OPT2_AS_Double_GENE(&tmp2,rPtr[u],wPtr[u],rPtr[u+1],wPtr[u+1], pPtr);
              rPtr[u]=SubMod(lPtr[u],tmp1,p);
              lPtr[u]=AddMod(lPtr[u],tmp1,p);
              rPtr[u+1]=SubMod(lPtr[u+1],tmp2,p);
              lPtr[u+1]=AddMod(lPtr[u+1],tmp2,p);
            }
            start+=m<<1;
        }
        m<<=1;
    }
    if(power>1){

        lPtr=tmpVecPtr;
        rPtr=lPtr+halfn;
        tmp1=MontMulMod_OPT2_AS_Double_GENE(&tmp2,rPtr[0],rootsPtr[0],rPtr[1],rootsPtr[n-1], pPtr);
        rPtr[0]=SubMod(lPtr[0],tmp1,p);
        lPtr[0]=AddMod(lPtr[0],tmp1,p);
        rPtr[1]=SubMod(lPtr[1],tmp2,p);
        lPtr[1]=AddMod(lPtr[1],tmp2,p);
        for(u=2; u<halfn; u+=2){
            tmp1=MontMulMod_OPT2_AS_Double_GENE(&tmp2,rPtr[u],rootsPtr[n-u],rPtr[u+1],rootsPtr[n-u-1], pPtr);
            rPtr[u]=SubMod(lPtr[u],tmp1,p);
            lPtr[u]=AddMod(lPtr[u],tmp1,p);
            rPtr[u+1]=SubMod(lPtr[u+1],tmp2,p);
            lPtr[u+1]=AddMod(lPtr[u+1],tmp2,p);
        }
    }
   
    invn=inverseMod(n,p);
    invn=MulMod(R,invn,p);
    invn<<=BRsft;
    for(i=0; i<n; i++) tmpVecPtr[i]=MontMulMod_OPT2_AS_GENE(tmpVecPtr[i], invn, pPtr);
    my_free(wPtr);
}

// type: local, in-place.
// note: good for machine word Fourier Prime in speical shape.
static void Mont_INVDFT_OPT2_AS_GENE_SPE_1 ( sfixn n, sfixn power, sfixn * rootsPtr,
            sfixn * tmpVecPtr, MONTP_OPT2_AS_GENE * pPtr) {

    register sfixn i, p=pPtr->P;
    // l is the size of result poly. 
    sfixn m, halfn=n>>1, s, u, t, invn, start=0, tmp1,tmp2, tmp, BRsft=pPtr->Base_Rpow;
    sfixn * wPtr, * lPtr, * rPtr;
    sfixn R=(1L<<(pPtr->Rpow))%p; 
    
    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }
    if(n<2){
        Interrupted=1; PrimeError=1;
        return;
    }

    if (power==0) return ;
    wPtr=(sfixn *)my_calloc(halfn, sizeof(sfixn));
    m=halfn;
    for(i=0; i<n; i+=2){
        tmp=tmpVecPtr[i];
        tmpVecPtr[i]=AddMod(tmpVecPtr[i], tmpVecPtr[i+1], p);
        tmpVecPtr[i+1]=SubMod(tmp, tmpVecPtr[i+1], p);
    }
    m=2;
    wPtr[0]=R<<pPtr->Base_Rpow;

    for(s=power-1; s>1; s--){
        for(i=1; i<m; i++) wPtr[i]=(rootsPtr[n-(i<<(s-1))]); 
        start=0;   
        for(t=0; t<(n/(m<<1)); t++){
            lPtr=tmpVecPtr+start;
            rPtr=lPtr+m;
            for(u=0; u<m; u+=2){
              tmp1=MontMulMod_OPT2_AS_Double_GENE_SPE(&tmp2,rPtr[u],wPtr[u],rPtr[u+1],wPtr[u+1], pPtr);
              rPtr[u]=SubMod(lPtr[u],tmp1,p);
              lPtr[u]=AddMod(lPtr[u],tmp1,p);
              rPtr[u+1]=SubMod(lPtr[u+1],tmp2,p);
              lPtr[u+1]=AddMod(lPtr[u+1],tmp2,p);
            }
            start+=m<<1;
        }
        m<<=1;
    }

    if(power>1){
        lPtr=tmpVecPtr;
        rPtr=lPtr+halfn;
        tmp1=MontMulMod_OPT2_AS_Double_GENE_SPE(&tmp2,rPtr[0],rootsPtr[0],rPtr[1],rootsPtr[n-1], pPtr);
        rPtr[0]=SubMod(lPtr[0],tmp1,p);
        lPtr[0]=AddMod(lPtr[0],tmp1,p);
        rPtr[1]=SubMod(lPtr[1],tmp2,p);
        lPtr[1]=AddMod(lPtr[1],tmp2,p);
        for(u=2; u<halfn; u+=2){
            tmp1=MontMulMod_OPT2_AS_Double_GENE_SPE(&tmp2,rPtr[u],rootsPtr[n-u],rPtr[u+1],rootsPtr[n-u-1], pPtr);
            rPtr[u]=SubMod(lPtr[u],tmp1,p);
            lPtr[u]=AddMod(lPtr[u],tmp1,p);
            rPtr[u+1]=SubMod(lPtr[u+1],tmp2,p);
            lPtr[u+1]=AddMod(lPtr[u+1],tmp2,p);
        }
    }

    invn=inverseMod(n,p);
    invn=MulMod(R,invn,p);
    invn<<=BRsft;
    for(i=0; i<n; i++) tmpVecPtr[i]=MontMulMod_OPT2_AS_GENE_SPE(tmpVecPtr[i], invn, pPtr);
    my_free(wPtr);
}

// type: exported
void EX_Mont_INVDFT_OPT2_AS_GENE_1 ( sfixn n, sfixn power, sfixn * rootsPtr,
     sfixn * tmpVecPtr, MONTP_OPT2_AS_GENE * pPtr){

    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }
    if((pPtr->c_pow==0) || ((pPtr->c_pow+pPtr->Rpow)>=BASE) ){
        Mont_INVDFT_OPT2_AS_GENE_1(n, power, rootsPtr,tmpVecPtr, pPtr);   
    } else{
        Mont_INVDFT_OPT2_AS_GENE_SPE_1(n, power, rootsPtr,tmpVecPtr, pPtr);
    }
}

// type: local, in-place.
// note: good for all machine word Fourier Prime.
//       remove the extra R from DFT at the end.
 /**
 * Mont_INVDFT_OPT2_AS_GENE_R_1:
 * @n: the FFT size
 * @power: log2 of n
 * @rootsPtr: powers of the n-th primitive root
 * @tmpVecPtr:  work space (int array of size n)
 * @pPtr: prime number structure
 * 
 * Inverse DFT. In place
 * 
 * 
 * Return value: Inverse DFT  of the input polynomial.
 **/    
static void Mont_INVDFT_OPT2_AS_GENE_R_1 (sfixn n, sfixn power, sfixn * rootsPtr,
            sfixn * tmpVecPtr, MONTP_OPT2_AS_GENE * pPtr) {

    register sfixn i, p=pPtr->P;
    // l is the size of result poly. 
    sfixn m, halfn=n>>1, s, u, t, invn, start=0, tmp1,tmp2, tmp, BRsft=pPtr->Base_Rpow;
    sfixn * wPtr, * lPtr, * rPtr;
    sfixn R=(1L<<(pPtr->Rpow))%p; 

    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }
    if(n<2){
        Interrupted=1; PrimeError=1;
        return;
    }


    if (power==0) return;
    wPtr=(sfixn *)my_calloc(halfn, sizeof(sfixn));
    m=halfn;
    
    for(i=0; i<n; i+=2){
         tmp=tmpVecPtr[i];
         tmpVecPtr[i]=AddMod(tmpVecPtr[i], tmpVecPtr[i+1], p);
         tmpVecPtr[i+1]=SubMod(tmp, tmpVecPtr[i+1], p);
    }
    m=2;
    wPtr[0]=R<<pPtr->Base_Rpow;

    for(s=power-1; s>1; s--){
        for(i=1; i<m; i++) wPtr[i]=(rootsPtr[n-(i<<(s-1))]); 
        start=0;   
        for(t=0; t<(n/(m<<1)); t++){
            lPtr=tmpVecPtr+start;
            rPtr=lPtr+m;
            for(u=0; u<m; u+=2){
                tmp1=MontMulMod_OPT2_AS_Double_GENE(&tmp2,rPtr[u],wPtr[u],rPtr[u+1],wPtr[u+1], pPtr);
                rPtr[u]=SubMod(lPtr[u],tmp1,p);
                lPtr[u]=AddMod(lPtr[u],tmp1,p);
                rPtr[u+1]=SubMod(lPtr[u+1],tmp2,p);
                lPtr[u+1]=AddMod(lPtr[u+1],tmp2,p);
            }
            start+=m<<1;
        }
        m<<=1;
    }

    if(power>1){
        lPtr=tmpVecPtr;
        rPtr=lPtr+halfn;  
        tmp1=MontMulMod_OPT2_AS_Double_GENE(&tmp2,rPtr[0],rootsPtr[0],rPtr[1],rootsPtr[n-1],pPtr);
        rPtr[0]=SubMod(lPtr[0],tmp1,p);
        lPtr[0]=AddMod(lPtr[0],tmp1,p);
        rPtr[1]=SubMod(lPtr[1],tmp2,p);
        lPtr[1]=AddMod(lPtr[1],tmp2,p);
        for(u=2; u<halfn; u+=2){
            tmp1=MontMulMod_OPT2_AS_Double_GENE(&tmp2,rPtr[u],rootsPtr[n-u],rPtr[u+1],rootsPtr[n-u-1], pPtr);
            rPtr[u]=SubMod(lPtr[u],tmp1,p);
            lPtr[u]=AddMod(lPtr[u],tmp1,p);
            rPtr[u+1]=SubMod(lPtr[u+1],tmp2,p);
            lPtr[u+1]=AddMod(lPtr[u+1],tmp2,p);
        }
    }
    invn=inverseMod(n,p);
    invn=MulMod(R,invn,p);
    invn=MulMod(R,invn,p);
    invn<<=BRsft;
    for(i=0; i<n; i++) tmpVecPtr[i]=MontMulMod_OPT2_AS_GENE(tmpVecPtr[i], invn, pPtr);
    my_free(wPtr);
}

// type: local, in-place.
// note: good for machine word Fourier Prime in special shape.
//       remove the extra R from DFT at the end.
 /**
 * Mont_INVDFT_OPT2_AS_GENE_SPE_R_1:
 * @n: the FFT size
 * @power: log2 of n
 * @rootsPtr: powers of the n-th primitive root
 * @tmpVecPtr:  work space (int array of size n)
 * @pPtr: prime number structure
 * 
 * Inverse DFT. In place. Case where p-1 is a power of 2.
 * 
 * Return value: Inverse DFT  of the input polynomial.
 **/    
static void Mont_INVDFT_OPT2_AS_GENE_SPE_R_1 ( sfixn n, sfixn power, sfixn * rootsPtr,
            sfixn * tmpVecPtr, MONTP_OPT2_AS_GENE * pPtr) {

    register sfixn i, p=pPtr->P;
    // l is the size of result poly. 
    sfixn m, halfn=n>>1, s, u, t, invn, start=0, tmp1,tmp2, tmp, BRsft=pPtr->Base_Rpow;
    sfixn * wPtr, * lPtr, * rPtr;
    sfixn R=(1L<<(pPtr->Rpow))%p; 
    
    if(power > (pPtr->Npow)){
       Interrupted=1; PrimeError=1;
       return;
    }
    if(n<2){
        Interrupted=1; PrimeError=1;
        return;
    }

    if (power==0) return ;
    wPtr=(sfixn *)my_calloc(halfn, sizeof(sfixn));
    m=halfn;

    for(i=0; i<n; i+=2){
        tmp=tmpVecPtr[i];
        tmpVecPtr[i]=AddMod(tmpVecPtr[i], tmpVecPtr[i+1], p);
        tmpVecPtr[i+1]=SubMod(tmp, tmpVecPtr[i+1], p);
    }
    m=2;
    wPtr[0]=R<<pPtr->Base_Rpow;

    for(s=power-1; s>1; s--){
        for(i=1; i<m; i++) wPtr[i]=(rootsPtr[n-(i<<(s-1))]); 
        start=0;   
        for(t=0; t<(n/(m<<1)); t++){
            lPtr=tmpVecPtr+start;
            rPtr=lPtr+m;
            for(u=0; u<m; u+=2){
                tmp1=MontMulMod_OPT2_AS_Double_GENE_SPE(&tmp2,rPtr[u],wPtr[u],rPtr[u+1],wPtr[u+1], pPtr);
                rPtr[u]=SubMod(lPtr[u],tmp1,p);
                lPtr[u]=AddMod(lPtr[u],tmp1,p);
                rPtr[u+1]=SubMod(lPtr[u+1],tmp2,p);
                lPtr[u+1]=AddMod(lPtr[u+1],tmp2,p);
            }
            start+=m<<1;
        }
        m<<=1;
    }

    if(power>1){
        lPtr=tmpVecPtr;
        rPtr=lPtr+halfn;
        tmp1=MontMulMod_OPT2_AS_Double_GENE_SPE(&tmp2,rPtr[0],rootsPtr[0],rPtr[1],rootsPtr[n-1],pPtr);
        rPtr[0]=SubMod(lPtr[0],tmp1,p);
        lPtr[0]=AddMod(lPtr[0],tmp1,p);
        rPtr[1]=SubMod(lPtr[1],tmp2,p);
        lPtr[1]=AddMod(lPtr[1],tmp2,p);
        for(u=2; u<halfn; u+=2){
            tmp1=MontMulMod_OPT2_AS_Double_GENE_SPE(&tmp2,rPtr[u],rootsPtr[n-u],rPtr[u+1],rootsPtr[n-u-1],pPtr);
            rPtr[u]=SubMod(lPtr[u],tmp1,p);
            lPtr[u]=AddMod(lPtr[u],tmp1,p);
            rPtr[u+1]=SubMod(lPtr[u+1],tmp2,p);
            lPtr[u+1]=AddMod(lPtr[u+1],tmp2,p);
        }
    }

    invn=inverseMod(n,p);
    invn=MulMod(R,invn,p);
    invn=MulMod(R,invn,p);
    invn<<=BRsft;
    for(i=0; i<n; i++) tmpVecPtr[i]=MontMulMod_OPT2_AS_GENE_SPE(tmpVecPtr[i], invn, pPtr);
    my_free(wPtr);
}


// type: exported.
 /**
 * EX_Mont_INVDFT_OPT2_AS_GENE__R_1:
 * @n: the FFT size
 * @power: log2 of n
 * @rootsPtr: powers of the n-th primitive root
 * @tmpVecPtr:  work space (int array of size n)
 * @pPtr: prime number structure
 * 
 * Inverse DFT. In place. Exported.
 * 
 * Return value: Inverse DFT  of the input polynomial.
 **/    
void EX_Mont_INVDFT_OPT2_AS_GENE_R_1 ( sfixn n, sfixn power, sfixn * rootsPtr,
     sfixn * tmpVecPtr, MONTP_OPT2_AS_GENE * pPtr){

    if((pPtr->c_pow==0) || ((pPtr->c_pow+pPtr->Rpow)>=BASE) ){
        Mont_INVDFT_OPT2_AS_GENE_R_1(n, power, rootsPtr,tmpVecPtr, pPtr);   
    } else{
        Mont_INVDFT_OPT2_AS_GENE_SPE_R_1(n, power, rootsPtr,tmpVecPtr, pPtr);
    }
}

//==============================================================================
// pairwise mul
//==============================================================================
static void Mont_PairwiseMul_OPT2_AS_R(sfixn n, sfixn * APtr, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr){
    register sfixn i;
    sfixn BRsft=pPtr->Base_Rpow;
    for(i=0; i<n; i++) APtr[i]=MontMulMod_OPT2_AS_GENE(APtr[i], BPtr[i]<<BRsft, pPtr);
}


static void Mont_PairwiseMul_OPT2_AS_SPE_R(sfixn n, sfixn * APtr, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr){
    register sfixn i;
    sfixn BRsft=pPtr->Base_Rpow;
    for(i=0; i<n; i++) APtr[i]=MontMulMod_OPT2_AS_GENE_SPE(APtr[i], BPtr[i]<<BRsft, pPtr);
}

 /**
 * EX_Mont_PairwiseMul_OPT2_AS_R:
 * @n: the FFT size
 * @APtr: coefficient vector of polynomial A
 * @BPtr: coefficient vector of polynomial B
 * @pPtr: prime number structure
 * 
 * 
 * Pairwise multiplicaiton. Big R issue *NOT* handled. 
 * This function is for expert usage only.
 * 
 * Return value: 
 **/    
void EX_Mont_PairwiseMul_OPT2_AS_R(sfixn n, sfixn * APtr, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr){
 
    if((pPtr->c_pow==0) || ((pPtr->c_pow+pPtr->Rpow)>=BASE) ){
        Mont_PairwiseMul_OPT2_AS_R(n, APtr, BPtr, pPtr);
    } else{
        Mont_PairwiseMul_OPT2_AS_SPE_R(n, APtr, BPtr, pPtr);
    }
}

 /**
 * EX_Mont_PairwiseMul_OPT2_AS:
 * @n: the FFT size
 * @APtr: coefficient vector of polynomial A
 * @BPtr: coefficient vector of polynomial B
 * @p: prime number
 * 
 * Pairwise multiplicaiton. Exported. No worries about the big R.
 * 
 * Return value: 
 **/    
void EX_Mont_PairwiseMul_OPT2_AS(sfixn n, sfixn * APtr, sfixn * BPtr, sfixn p){
    register sfixn i;
    for(i=0; i<n; i++) APtr[i]=MulMod(APtr[i], BPtr[i], p);
}

//==============================================================================
// FFT
//==============================================================================
// type: exported.
 /**
 * EX_Mont_FFTMul_OPT2_AS_GENE:
 * @n: the FFT size
 * @e: log2 of n
 * @degRes: degree of result
 * @resPtr: coefficient vector of result
 * @degA: degree of A
 * @APtr: coefficient vector of A 
 * @degB: degree of B
 * @BPtr: coefficient vector of B
 * @pPtr: prime number structure
 * 
 * FFT-based multiplication of A by B. Result is in resPtr.
 * 
 * Return value: the product of two polynomials.
 **/   
void EX_Mont_FFTMul_OPT2_AS_GENE(sfixn n, sfixn e, sfixn degRes, sfixn * resPtr,
     sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr){

    register sfixn i;
    sfixn l_AB, e1, n1, BRsft=pPtr->Base_Rpow;
    sfixn * rootsPtr, * dftAPtr, * dftBPtr;

//#if defined(_cuda_modp_)
//    cumodp_err err;
//#endif

    l_AB=degRes+1; // compute l_AB for TFT(A,B).
    e1=logceiling(l_AB);
    n1=1L<<e1;
    // force this FFT use the n from parameter if n<n1.
    if ((n<n1) || (degRes==0)) {n1=n; e1=e;}
    if (e1 > (pPtr->Npow)){ Interrupted=1; PrimeError=1; return; }

//#if defined(_cuda_modp_)
//    if (CUDA_TAG && e >= 16 && can_call_fftmul_uni(degA, degB)) {
//        if (DEBUG) printf("## fftmul::degA = %d, degB = %d\n", degA, degB);
//        err = cumodp_fftmul_uni(degRes, resPtr, degA, APtr, degB, BPtr, pPtr->P);
//        if (err == CUMODP_SUCCESS) {
//            if (DEBUG) printf("## fftmul::gpu code succeeds and returns\n");
//            return; 
//        } else {
//            if (DEBUG) fprintf(stderr, "cuda code fails\n");
//        }
//    }
//#endif

    rootsPtr=(sfixn *)my_calloc(n1, sizeof(sfixn));
    dftAPtr=(sfixn *)my_calloc(n1, sizeof(sfixn));
    dftBPtr=(sfixn *)my_calloc(n1, sizeof(sfixn));
  
    if((pPtr->c_pow==0) || ((pPtr->c_pow+pPtr->Rpow)>=BASE) ){
        Mont_GetNthRoots_OPT2_AS_GENE(e1,n1,rootsPtr,pPtr);
        Mont_dft_OPT2_AS_GENE( n1, e1, rootsPtr, dftAPtr, degA, APtr, pPtr);
        Mont_dft_OPT2_AS_GENE( n1, e1, rootsPtr, dftBPtr, degB, BPtr, pPtr);
        for(i=0; i<n1; i++) dftAPtr[i]=MontMulMod_OPT2_AS_GENE(dftAPtr[i], dftBPtr[i]<<BRsft, pPtr);
        Mont_invdft_OPT2_AS_GENE_R(n1, e1, rootsPtr, dftAPtr, degRes, resPtr, pPtr);
    }else{
        Mont_GetNthRoots_OPT2_AS_GENE_SPE(e1,n1,rootsPtr,pPtr);
        Mont_dft_OPT2_AS_GENE_SPE( n1, e1, rootsPtr, dftAPtr, degA, APtr, pPtr);
        Mont_dft_OPT2_AS_GENE_SPE( n1, e1, rootsPtr, dftBPtr, degB, BPtr, pPtr);
        for(i=0; i<n1; i++) dftAPtr[i]=MontMulMod_OPT2_AS_GENE_SPE(dftAPtr[i], dftBPtr[i]<<BRsft, pPtr);
        Mont_invdft_OPT2_AS_GENE_SPE_R(n1, e1, rootsPtr, dftAPtr, degRes, resPtr, pPtr);
    }

    my_free(dftAPtr);
    my_free(dftBPtr);
    my_free(rootsPtr);
}

// type: exported, in-place.
 /**
 * EX_Mont_FFTMul_OPT2_AS_GENE_1:
 * @n: the FFT size
 * @e: log2 of n
 * @degRes: degree of result
 * @degA: (Input and Output) degree of A
 * @APtr: coefficient vector of A 
 * @degB: degree of B
 * @BPtr: coefficient vector of B
 * @pPtr: prime number structure
 * 
 * FFT-based multiplication of A by B. In place: result in A.
 * 
 * Return value: the product of two polynomials.
 **/   
void EX_Mont_FFTMul_OPT2_AS_GENE_1(sfixn n, sfixn e, sfixn degRes, sfixn degA, sfixn * APtr,
     sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr){

    register sfixn i;
    sfixn l_AB, e1, n1;
    sfixn * rootsPtr;

    l_AB=degRes+1; // compute l_AB for TFT(A,B).
    e1=logceiling(l_AB);
    n1=1L<<e1;
    // force this FFT use the n from parameter if n<n1.
    if ((n<n1)||(degRes==0)) {n1=n; e1=e;}

    if(e1 > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }
    rootsPtr=(sfixn *)my_calloc(n1, sizeof(sfixn));
    EX_Mont_GetNthRoots_OPT2_AS_GENE(e1, n1, rootsPtr, pPtr);
    for(i=degA+1;i<n1;i++) APtr[i]=0;
    for(i=degB+1;i<n1;i++) BPtr[i]=0;  
    EX_Mont_DFT_OPT2_AS_GENE_1( n1, e1, rootsPtr, APtr, pPtr);
    EX_Mont_DFT_OPT2_AS_GENE_1( n1, e1, rootsPtr, BPtr, pPtr);
    EX_Mont_PairwiseMul_OPT2_AS_R(n1, APtr, BPtr, pPtr);
    EX_Mont_INVDFT_OPT2_AS_GENE_R_1(n1, e1, rootsPtr, APtr, pPtr);
    my_free(rootsPtr);
}

// type: exported, in-place.
void EX_Mont_FFTMul_OPT2_AS_GENE_1_2(sfixn n, sfixn e, sfixn * APtr, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr){
    sfixn * rootsPtr;
    rootsPtr=(sfixn *)my_calloc(n, sizeof(sfixn));
    EX_Mont_GetNthRoots_OPT2_AS_GENE(e, n, rootsPtr, pPtr);
    EX_Mont_DFT_OPT2_AS_GENE_1( n, e, rootsPtr, APtr, pPtr);
    EX_Mont_DFT_OPT2_AS_GENE_1( n, e, rootsPtr, BPtr, pPtr);
    EX_Mont_PairwiseMul_OPT2_AS_R(n, APtr, BPtr, pPtr);
    EX_Mont_INVDFT_OPT2_AS_GENE_R_1(n, e, rootsPtr, APtr, pPtr);
    my_free(rootsPtr);
}

// type: exported, in-place.
 /**
 * EX_Mont_FFTMul_OPT2_AS_GENE_1:
 * @n: the FFT size
 * @e: log2 of n
 * @degRes: degree of result
 * @degA: (Input and Output) degree of A
 * @APtr: coefficient vector of A 
 * @pPtr: prime number structure
 * 
 * FFT-based square of A. In place: result in A.
 * Return value: the square of a polynomial.
 **/   
void EX_Mont_FFTSQUARE_OPT2_AS_GENE_1(sfixn n, sfixn e, sfixn degRes,
     sfixn degA, sfixn * APtr, MONTP_OPT2_AS_GENE * pPtr){

    register sfixn i;
    sfixn l_AB, e1, n1;
    sfixn * rootsPtr;
    
    l_AB=degRes+1; // compute l_AB for TFT(A,B).
    e1=logceiling(l_AB);
    n1=1L<<e1;
    // force this FFT use the n from parameter if n<n1.
    if((n<n1)||(degRes==0)) {n1=n; e1=e;}
    if(e > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }
    rootsPtr=(sfixn *)my_calloc(n1, sizeof(sfixn));
    EX_Mont_GetNthRoots_OPT2_AS_GENE(e1, n1, rootsPtr, pPtr);
    for(i=degA+1;i<n1;i++) APtr[i]=0;
    EX_Mont_DFT_OPT2_AS_GENE_1( n1, e1, rootsPtr, APtr, pPtr);
    EX_Mont_PairwiseMul_OPT2_AS_R(n1, APtr, APtr, pPtr);
    EX_Mont_INVDFT_OPT2_AS_GENE_R_1(n1, e1, rootsPtr, APtr, pPtr);
    my_free(rootsPtr);
}

// type: exported, in-place.
// note: for Kronecker.
 /**
 * EX_Mont_FFTMul_OPT2_AS_GENE_1:
 * @n: the FFT size
 * @e: log2 of n
 * @degRes: degree of result
 * @rootsPtr: powers of the primitive n-th root of unity
 * @degA: (Input and Output) degree of A
 * @APtr: coefficient vector of A 
 * @degB: degree of B
 * @BPtr: coefficient vector of B
 * @pPtr: prime number structure
 * 
 * FFT-based multiplication of A by B. In place: result in A.
 * 
 * Return value: the product of two polynomials.
 **/   
void EX_KN_Mont_FFTMul_OPT2_AS_GENE_1(sfixn n, sfixn e, sfixn degRes, 
    sfixn * rootsPtr, sfixn degA, sfixn *APtr, sfixn degB, sfixn *BPtr, 
    MONTP_OPT2_AS_GENE *pPtr)
{
    sfixn i, l_AB, e1, n1;

#if defined(_cuda_modp_)
    cumodp_err err;
#endif

    l_AB = degRes + 1; 
    e1 = logceiling(l_AB);
    n1 = 1L << e1;
    // force this FFT use the n from parameter if n < n1.
    if ((n < n1) || (degRes == 0)) { n1 = n; e1 = e; }
    if (e1 > (pPtr->Npow)) { Interrupted = 1; PrimeError = 1; return; }

    for (i = degA + 1; i < n1; i++) APtr[i] = 0;
    for (i = degB + 1; i < n1; i++) BPtr[i] = 0;  
    
#if defined(_cuda_modp_)
    // GPU code
    if (CUDA_TAG && e1 >= 16 && can_call_fftmul_uni(degA, degB)) {
        if (DEBUG) printf("## fftmul::calling GPU code with size 2^%d\n", e1);
        if (DEBUG) printf("## fftmul::degA = %d, degB = %d\n", degA, degB);
        // Gaurantee that APtr does not get modified if failed
        err = cumodp_fftmul_uni(n1 - 1, APtr, degA, APtr, degB, BPtr, pPtr->P);
        if (err == CUMODP_SUCCESS) {
            if (DEBUG) printf("## fftmul::gpu code succeeds and returns\n");
            return; 
        } else {
            if (DEBUG) fprintf(stderr, "cuda code fails\n");
        }
    }
#endif

    EX_Mont_DFT_OPT2_AS_GENE_1(n1, e1, rootsPtr, APtr, pPtr);
    EX_Mont_DFT_OPT2_AS_GENE_1(n1, e1, rootsPtr, BPtr, pPtr);
    EX_Mont_PairwiseMul_OPT2_AS_R(n1, APtr, BPtr, pPtr);
    EX_Mont_INVDFT_OPT2_AS_GENE_R_1(n1, e1, rootsPtr, APtr, pPtr);
}


// fastest FFT and good for FFT-primes.
 /**
 * EX_Mont_FFTMul_OPT2_AS_GENE_1:
 * @n: the FFT size
 * @e: log2 of n
 * @degRes: degree of result
 * @rootsPtr: powers of the primitive n-th root of unity
 * @degA: (Input and Output) degree of A
 * @APtr: coefficient vector of A 
 * @pPtr: prime number structure
 * 
 * FFT-based square of A. In place: result in A.
 * 
 * Return value: the square of a polynomial.
 **/   
void EX_KN_Mont_FFTSQUARE_OPT2_AS_GENE_1(sfixn n, sfixn e, sfixn degRes, sfixn * rootsPtr,
     sfixn degA, sfixn * APtr, MONTP_OPT2_AS_GENE * pPtr){

    register sfixn i;
    sfixn l_AB, e1, n1;
    l_AB=degRes+1; // compute l_AB for TFT(A,B).
    e1=logceiling(l_AB);
    n1=1L<<e1;
    // force this FFT use the n from parameter if n<n1.
    if ((n<n1)||(degRes==0)) {n1=n; e1=e;}
    EX_Mont_GetNthRoots_OPT2_AS_GENE(e1, n1, rootsPtr, pPtr);
    for(i=degA+1;i<n1;i++) APtr[i]=0;
    EX_Mont_DFT_OPT2_AS_GENE_1( n1, e1, rootsPtr, APtr, pPtr);
    EX_Mont_PairwiseMul_OPT2_AS_R(n1, APtr, APtr, pPtr);
    EX_Mont_INVDFT_OPT2_AS_GENE_R_1(n1, e1, rootsPtr, APtr, pPtr);
}

// type: exported.
// note: FFTMul with a cutoff with plainMul
 /**
 * EX_Mont_FFTMul:
 * @degRes: degree of result
 * @resPtr: coefficient vector of result
 * @degA: degree of A
 * @APtr: coefficient vector of A 
 * @degB: degree of B
 * @BPtr: coefficient vector of B
 * @pPtr: prime number structure
 * 
 * Multiplication of A by B. Result is in resPtr.
 * Uses classical multiplication for small degrees and for larger ones. 
 * 
 * Return value: the product of two polynomials.
 **/   
void EX_Mont_FFTMul(sfixn degRes, sfixn * resPtr, sfixn degA, sfixn * APtr,
     sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn e,n;
    if(degRes<CutOffPlainFFTMul){
        EX_Mont_PlainMul_OPT2_AS_GENE(degRes, resPtr, degA, APtr, degB, BPtr, pPtr );
    } else { 
        e=logceiling(degRes+1);
        n=1<<e;
        if(e > (pPtr->Npow)){
            Interrupted=1; PrimeError=1;
            return;
        }
        EX_Mont_FFTMul_OPT2_AS_GENE(n, e, degRes, resPtr, degA, APtr, degB, BPtr, pPtr);
   }
}

/**
 * EX_Mont_Mul:
 * @degResAddr: degree of result
 * @degA: degree of A
 * @APtr: coefficient vector of A 
 * @degB: degree of B
 * @BPtr: coefficient vector of B
 * @pPtr: prime number structure
 * 
 * Returns the coefficient of the product of A by B.
 * Uses classical multiplication for small degrees and for larger ones. 
 * 
 * Return value: the product of two polynomials.
 **/   
sfixn *EX_Mont_Mul(sfixn *degResAddr, sfixn degA, sfixn * APtr,
       sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn *resPtr;
    sfixn e,n;
      
    *degResAddr=degA+degB;
    resPtr=(sfixn *)my_calloc((*degResAddr)+1, sizeof(sfixn));

    if((*degResAddr)<CutOffPlainFFTMul){
        EX_Mont_PlainMul_OPT2_AS_GENE(*degResAddr, resPtr, degA, APtr, degB, BPtr, pPtr );
    } else { 
        e=logceiling((*degResAddr)+1);
        n=1<<e;
        if(e > (pPtr->Npow)){
            Interrupted=1; PrimeError=1;
            return resPtr;
        }
        EX_Mont_FFTMul_OPT2_AS_GENE(n, e, *degResAddr, resPtr, degA, APtr, degB, BPtr, pPtr);
    }
    return resPtr;
}

// TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT
// TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT
// TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT
// TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT
// TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT
// TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT
// TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT
// TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT
// TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT
// TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT
// TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT
// TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT TFT

//==============================================================================
// TFT  (WHOLE)
//==============================================================================

/*
a  b
 \/
 /\
c  d
   Input c,d. Output a,b.
*/
static void bfOpt_new_0(sfixn * lPtr, sfixn * rPtr, sfixn w, MONTP_OPT2_AS_GENE * pPtr) { 
    sfixn tmp1;
    tmp1=MontMulMod_OPT2_AS_GENE(rPtr[0],w, pPtr);
    rPtr[0]=SubMod(lPtr[0],tmp1,pPtr->P);
    lPtr[0]=AddMod(lPtr[0],tmp1,pPtr->P);
}

/*
a  b
 \/
 /\
c  d
Input c,d. Output a.
*/
static void lhalfbfOpt_new_0(sfixn * lPtr, sfixn * rPtr, sfixn  w, MONTP_OPT2_AS_GENE * pPtr ) { 
  lPtr[0]=AddMod(lPtr[0], MontMulMod_OPT2_AS_GENE(rPtr[0],w, pPtr), pPtr->P);
}

/* 
a  b
 \/
 /\
c  d
Input c,b. Output a,d. */
static void bfOpt_new_1(sfixn * lPtr, sfixn * rPtr, sfixn * wInvPtr, MONTP_OPT2_AS_GENE * pPtr ) {
    sfixn c;
    c=lPtr[0];
    lPtr[0]= SubMod(AddMod(lPtr[0],lPtr[0],pPtr->P), rPtr[0], pPtr->P);
    rPtr[0]= MontMulMod_OPT2_AS_GENE(SubMod(c, rPtr[0],pPtr->P), wInvPtr[0], pPtr);
}

/*
a  b
 \/
 /\
c  d
Input c,b. Output a.
*/

static void lhalfBfOpt_new_1(sfixn * lPtr, sfixn * rPtr, MONTP_OPT2_AS_GENE * pPtr) {
     lPtr[0]= SubMod(AddMod(lPtr[0],lPtr[0],pPtr->P),rPtr[0],pPtr->P);  
}

/*
a  b
 \/
 /\
c  d
 Input a,b. Output c.
*/

static void lhalfbfOpt_new_2(sfixn * lPtr, sfixn * rPtr, sfixn inv12RSFT, MONTP_OPT2_AS_GENE * pPtr ) { 
  sfixn l=AddMod(lPtr[0],rPtr[0],pPtr->P);
  lPtr[0]=MontMulMod_OPT2_AS_GENE(l,inv12RSFT,pPtr);
}

// fastest FFT and good for FFT-primes.
 /**
 * EX_Mont_TFTMul_OPT2_AS_GENE:
 * @resPtr: coefficient vector of result
 * @d1: degree of A
 * @v1Ptr: coefficient vector of A 
 * @d2: degree of B
 * @v2Ptr: coefficient vector of B
 * @pPtr: prime number structure
 * 
 * TFT-based multiplication of v1Ptr by v2Ptr. Result is in resPtr.
 * 
 * 
 * Return value: the product of two polynomials.
 **/  
void Mont_TFTMul_OPT2_AS_GENE(sfixn * resPtr, sfixn d1, sfixn * v1Ptr, sfixn d2, sfixn * v2Ptr, MONTP_OPT2_AS_GENE * pPtr){

    sfixn i;
    // l is the size of result poly. 
    sfixn p, BRsft, inv12RSFT, R, l=d1+d2+1, n=1, m, halfn, power=0;
    sfixn s, t, u, Tno, invn, Rno, Bno, start=0, tmp, tmp2, left=0, right=0, tmpw, end1, end2;
    sfixn *lsPtr, *newv1Ptr, *newv2Ptr, *rootsPtr, *wPtr, *lPtr1, *rPtr1, *lPtr2, *rPtr2, *tmpPtr;
    register sfixn t1, t2, tt1, tt2;

    // compute the n=2^power.
    for(i=l; i>1; i>>=1) {n<<=1; power++;}
    if (n<l) {n<<=1; power++;}  

    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }

    if (l==n) return EX_Mont_FFTMul_OPT2_AS_GENE( n, power, l, resPtr, d1, v1Ptr, d2, v2Ptr, pPtr);

    p=pPtr->P;
    BRsft=pPtr->Base_Rpow;
    R=(1L<<(pPtr->Rpow))%p; 
    inv12RSFT=inverseMod(2,p);
    inv12RSFT=MulMod(R,inv12RSFT,p);
    inv12RSFT=inv12RSFT<<BRsft;

    halfn=n>>1;
    newv1Ptr=(sfixn *)my_calloc(n, sizeof(sfixn));
    newv2Ptr=(sfixn *)my_calloc(n, sizeof(sfixn));
    rootsPtr=(sfixn *)my_calloc(n, sizeof(sfixn));
    EX_Mont_GetNthRoots_OPT2_AS_GENE(power, n, rootsPtr, pPtr);

    // lsPtr[0] not in use.
    lsPtr=(sfixn *)my_calloc(power+1,sizeof(sfixn));

    // lsPtr[i] keeps the number of points needed to compute at level i 
    m=halfn;
    for(i=1; i<=power; i++ ){
        // WAS lsPtr[i]=(l/m+1)*m;
        // WAS m>>=1;
        // The number of points has been adjusted according the TFT paper
        lsPtr[i] = (((l-1)>>(power-i))+1)<<(power-i);
    }

    for (i=0; i<=d1; i++)   newv1Ptr[i]=v1Ptr[i];
    for (i=0; i<=d2; i++)   newv2Ptr[i]=v2Ptr[i];

    // DFTs begin=========================================>
    // compute the top level. (level-1 to level-2).
    
    for (i=0; i<(l-halfn); i++) {
        tmp=newv1Ptr[i];
        newv1Ptr[i]=AddMod(newv1Ptr[i],newv1Ptr[i+halfn],p);
        newv1Ptr[i+halfn]=SubMod(tmp,newv1Ptr[i+halfn],p);
        tmp=newv2Ptr[i];
        newv2Ptr[i]=AddMod(newv2Ptr[i],newv2Ptr[i+halfn],p);
        newv2Ptr[i+halfn]=SubMod(tmp,newv2Ptr[i+halfn],p);
    }
    for(i=(halfn-(n-l)); i<halfn;i++){
        newv1Ptr[i+halfn]=newv1Ptr[i];
        newv2Ptr[i+halfn]=newv2Ptr[i];
    }

    // compute the rest levels.
    m=halfn>>1;
    for (s=2; s<=power; s++){
        // # of teams with "full butterflies";
        Tno=lsPtr[s]/(m<<1);
        // # of "half bufflies" in the last team;
        Rno=lsPtr[s]%(m<<1);
        start=0;

        // middle-loop begin
        for (t=0; t<Tno; t++){
            tmpw=rootsPtr[partialBitRev(t<<1,s)*m];
            lPtr1=newv1Ptr+start; rPtr1=lPtr1+m;
            t1=MontMulMod_OPT2_AS_GENE(rPtr1[0],tmpw, pPtr);
            lPtr2=newv2Ptr+start; rPtr2=lPtr2+m;
            t2=MontMulMod_OPT2_AS_GENE(rPtr2[0],tmpw, pPtr);
            tmp=m-1;
            // inner-loop begin
            for (u=0; u<tmp; u++ ) {
                rPtr1[u]=SubMod(lPtr1[u],t1,p);
                lPtr1[u]=AddMod(lPtr1[u],t1,p);
                tt1=MontMulMod_OPT2_AS_GENE(rPtr1[u+1],tmpw, pPtr);
                t1=tt1;
                rPtr2[u]=SubMod(lPtr2[u],t2,p);
                lPtr2[u]=AddMod(lPtr2[u],t2,p);
                tt2=MontMulMod_OPT2_AS_GENE(rPtr2[u+1],tmpw, pPtr);
                t2=tt2;
            } // inner-loop end.

            rPtr1[tmp]=SubMod(lPtr1[tmp],t1,p);
            lPtr1[tmp]=AddMod(lPtr1[tmp],t1,p);
 
            rPtr2[tmp]=SubMod(lPtr2[tmp],t2,p);
            lPtr2[tmp]=AddMod(lPtr2[tmp],t2,p);
            start+=(m<<1);
         } // middle-loop end.
        // the last group only need to compute half bufferflies.
        if(Rno){
            tmpw=rootsPtr[partialBitRev(Tno<<1,s)*m];
            for (u=start; u<start+m; u++ ){
	            lPtr1=newv1Ptr+u;
                lPtr1[0]=AddMod(lPtr1[0],MontMulMod_OPT2_AS_GENE(lPtr1[m],tmpw, pPtr),p);
                lPtr2=newv2Ptr+u;
                lPtr2[0]=AddMod(lPtr2[0],MontMulMod_OPT2_AS_GENE(lPtr2[m],tmpw, pPtr),p);
            }
        } // if(Rno) end
        m>>=1;
    }
    
    // DFTs end ==============================================>
    for(i=0; i<l; i++) newv1Ptr[i]=MontMulMod_OPT2_AS_GENE(newv1Ptr[i], newv2Ptr[i]<<BRsft, pPtr);

    // INV-DFT starts
    // step-1: left FFT-tree bottom uPpter->
    wPtr=(sfixn *)my_calloc(halfn, sizeof(sfixn));
    end1=(l>>1)<<1;   
    for(i=0; i<end1; i+=2){
        tmp=newv1Ptr[i];
        newv1Ptr[i]=AddMod(newv1Ptr[i],newv1Ptr[i+1],p);
        newv1Ptr[i+1]=SubMod(tmp,newv1Ptr[i+1],p);
    }
    m=2;
    wPtr[0]=rootsPtr[0];
    for(s=power-2; s>0; s--){
        for(i=1; i<m; i++) wPtr[i]=rootsPtr[n-(i<<s)];
        tmp=m<<1; 
        end2=(l/tmp)*tmp;
        for(start=0; start<end2; start+=tmp){
            tmpPtr=newv1Ptr+start;
            for(u=0; u<m; u++) bfOpt_new_0(tmpPtr+u, tmpPtr+u+m, wPtr[u], pPtr);
        }
        m<<=1;
    }

    //step-2: Compute the top level.
    Bno=halfn;
    for(i=l; i<n; i++){
        tmp=i-Bno;
        newv1Ptr[i]= MontMulMod_OPT2_AS_GENE(newv1Ptr[tmp], rootsPtr[tmp], pPtr);
        newv1Ptr[tmp]=AddMod(newv1Ptr[tmp],newv1Ptr[tmp],p); 
    }

    //step-3: right part of FFT-tree, top-down go  as deep as possible.
    Bno=halfn>>1;
    for (s=1; s<power; s++ ){
        tmp=lsPtr[s]-l;
        // later on we will use inverse of roots.
      
        if(tmp<Bno){
            tmp2=Bno-tmp;
            for(i=0; i<tmp; i++ ) wPtr[i]=rootsPtr[(i+tmp2)<<s];
            end1=lsPtr[s]-l;
            tmpPtr=newv1Ptr+l;
            for(t=0; t<end1; t++ ) bfOpt_new_1(tmpPtr+t-Bno, tmpPtr+t, wPtr+t, pPtr);
        }
        if(tmp>Bno){ 
            end1=lsPtr[s]-Bno;
            for(tmpPtr=newv1Ptr+l; tmpPtr<newv1Ptr+end1; tmpPtr++ ) lhalfbfOpt_new_2(tmpPtr, tmpPtr+Bno, inv12RSFT, pPtr);
        }
    
        if(tmp==Bno){
            lPtr1=newv1Ptr-Bno+l; tmpPtr=newv1Ptr+l;
            for(; lPtr1<tmpPtr; lPtr1++) lhalfBfOpt_new_1(lPtr1, lPtr1+Bno, pPtr);
            left=l-Bno; right=l;
            break;
        }
        Bno>>=1; 
    }
    
    //step-4: bottom up again.
    wPtr[0]=rootsPtr[0];
    for(; s>1; s--){
        Bno<<=1;
        if(lsPtr[s]==lsPtr[s-1]){
            for(i=1; i<Bno; i++ ) wPtr[i]=rootsPtr[n-(i<<(s-1))];
            for(t=left; t<right; t++) bfOpt_new_0(newv1Ptr+t-Bno, newv1Ptr+t, wPtr[t-left], pPtr);
            left=left-Bno;
        } else {
            for(t=left; t<right; t++) lhalfBfOpt_new_1(newv1Ptr+t, newv1Ptr+t+Bno, pPtr);
        }
    }
    
    //step-5: fill up the missing points at the top level.
    for(i=1; i<halfn; i++ ) wPtr[i]=rootsPtr[n-i];
    for(i=0; i<l-halfn; i++) bfOpt_new_0(newv1Ptr+i, newv1Ptr+i+halfn, wPtr[i],pPtr);  

    invn=inverseMod(n,p);
    invn=MulMod(R,invn,p);
    invn=MulMod(R,invn,p);
    invn<<=BRsft;
    for(i=0; i<l; i++) resPtr[i]=MontMulMod_OPT2_AS_GENE(newv1Ptr[i], invn, pPtr);
    
    my_free(lsPtr); 
    my_free(newv1Ptr); 
    my_free(newv2Ptr); 
    my_free(rootsPtr);
    my_free(wPtr);
}

// SPE =============>
// SPE =============>
// SPE =============>
// SPE =============>
// SPE =============>
// SPE =============>
// SPE =============>
// SPE =============>
// SPE =============>

/*
a  b
 \/
 /\
c  d
   Input c,d. Output a,b.
*/

static void bfOpt_new_0_SPE(sfixn * lPtr, sfixn * rPtr, sfixn w, MONTP_OPT2_AS_GENE * pPtr) { 
    sfixn tmp1;
    tmp1=MontMulMod_OPT2_AS_GENE_SPE(rPtr[0],w, pPtr);
    rPtr[0]=SubMod(lPtr[0],tmp1,pPtr->P);
    lPtr[0]=AddMod(lPtr[0],tmp1,pPtr->P);
}


/*
a  b
 \/
 /\
c  d
Input c,d. Output a.
*/
static void lhalfbfOpt_new_0_SPE(sfixn * lPtr, sfixn * rPtr, sfixn  w, MONTP_OPT2_AS_GENE * pPtr ) { 
    lPtr[0]=AddMod(lPtr[0], MontMulMod_OPT2_AS_GENE_SPE(rPtr[0],w, pPtr), pPtr->P);
}


/* 
a  b
 \/
 /\
c  d
Input c,b. Output a,d. */

static void bfOpt_new_SPE_1(sfixn * lPtr, sfixn * rPtr, sfixn * wInvPtr, MONTP_OPT2_AS_GENE * pPtr ) {
    sfixn c;
    c=lPtr[0];
    lPtr[0]= SubMod(AddMod(lPtr[0],lPtr[0],pPtr->P), rPtr[0], pPtr->P);
    rPtr[0]= MontMulMod_OPT2_AS_GENE_SPE(SubMod(c, rPtr[0],pPtr->P), wInvPtr[0], pPtr);
}

/*
a  b
 \/
 /\
c  d
Input c,b. Output a.
*/

static void lhalfBfOpt_new_SPE_1(sfixn * lPtr, sfixn * rPtr, MONTP_OPT2_AS_GENE * pPtr) {
     lPtr[0]= SubMod(AddMod(lPtr[0],lPtr[0],pPtr->P),rPtr[0],pPtr->P);  
}

/*
a  b
 \/
 /\
c  d
 Input a,b. Output c.
*/
static void lhalfbfOpt_new_2_SPE(sfixn * lPtr, sfixn * rPtr, sfixn inv12RSFT, MONTP_OPT2_AS_GENE * pPtr) { 
  sfixn l=AddMod(lPtr[0],rPtr[0],pPtr->P);
  lPtr[0]=MontMulMod_OPT2_AS_GENE_SPE(l,inv12RSFT,pPtr);
}

 /**
 * EX_Mont_TFTMul_OPT2_AS_SPE:
 * @resPtr: coefficient vector of result
 * @d1: degree of A
 * @v1Ptr: coefficient vector of A 
 * @d2: degree of B
 * @v2Ptr: coefficient vector of B
 * @pPtr: prime number structure
 * 
 * TFT-based multiplication of v1Ptr by v2Ptr. Result is in resPtr.
 * assume prime is N+1, where N is a power of 2.
 * 
 * Return value: the product of two polynomials.
 **/  
void Mont_TFTMul_OPT2_AS_GENE_SPE(sfixn * resPtr, sfixn d1, sfixn * v1Ptr,
     sfixn d2, sfixn * v2Ptr, MONTP_OPT2_AS_GENE * pPtr){

    sfixn i;
    // l is the size of result poly. 
    sfixn p, BRsft, inv12RSFT, R, l=d1+d2+1, n=1, m, halfn, power=0;
    sfixn s, t, u, Tno, invn, Rno, Bno, start=0, tmp, tmp2, left=0, right=0, tmpw, end1, end2;
    sfixn * lsPtr, * newv1Ptr, * newv2Ptr, * rootsPtr, * wPtr, * lPtr1, * rPtr1, * lPtr2, *rPtr2, * tmpPtr;
    register sfixn t1,t2,tt1,tt2;
    
    // compute the n=2^power.
    for(i=l; i>1; i>>=1) {n<<=1; power++;}
    if (n<l) {n<<=1; power++;}  
    
    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }
    if (l==n) return EX_Mont_FFTMul_OPT2_AS_GENE( n, power, l, resPtr, d1, v1Ptr, d2, v2Ptr, pPtr);
 

    p=pPtr->P;
    BRsft=pPtr->Base_Rpow;
    R=(1L<<(pPtr->Rpow))%p; 
    inv12RSFT=inverseMod(2,p);
    inv12RSFT=MulMod(R,inv12RSFT,p);
    inv12RSFT=inv12RSFT<<BRsft;
    
    halfn=n>>1;
    newv1Ptr=(sfixn *)my_calloc(n, sizeof(sfixn));
    newv2Ptr=(sfixn *)my_calloc(n, sizeof(sfixn));
    rootsPtr=(sfixn *)my_calloc(n, sizeof(sfixn));
    EX_Mont_GetNthRoots_OPT2_AS_GENE(power, n, rootsPtr, pPtr);

    // lsPtr[0] not in use.
    lsPtr=(sfixn *)my_calloc(power+1,sizeof(sfixn));

    // lsPtr[i] keeps the number of points needed to compute at level i 
    m=halfn;
    for(i=1; i<=power; i++ ){
        //WAS lsPtr[i]=(l/m+1)*m;
        //WAS  m>>=1;
        lsPtr[i] = (((l-1)>>(power-i))+1)<<(power-i);
    }

    for (i=0; i<=d1; i++)   newv1Ptr[i]=v1Ptr[i];
    for (i=0; i<=d2; i++)   newv2Ptr[i]=v2Ptr[i];

    // DFTs begin=========================================>
    // compute the top level. (level-1 to level-2).

    for (i=0; i<(l-halfn); i++) {
        tmp=newv1Ptr[i];
        newv1Ptr[i]=AddMod(newv1Ptr[i],newv1Ptr[i+halfn],p);
        newv1Ptr[i+halfn]=SubMod(tmp,newv1Ptr[i+halfn],p);
        tmp=newv2Ptr[i];
        newv2Ptr[i]=AddMod(newv2Ptr[i],newv2Ptr[i+halfn],p);
        newv2Ptr[i+halfn]=SubMod(tmp,newv2Ptr[i+halfn],p);
    }
    for(i=(halfn-(n-l)); i<halfn;i++){
        newv1Ptr[i+halfn]=newv1Ptr[i];
        newv2Ptr[i+halfn]=newv2Ptr[i];
    }

    // compute the rest levels.
    m=halfn>>1;
    for (s=2; s<=power; s++){
        // # of teams with "full butterflies";
        Tno=lsPtr[s]/(m<<1);
        // # of "half bufflies" in the last team;
        Rno=lsPtr[s]%(m<<1);
        start=0;

        // middle-loop begin
        for (t=0; t<Tno; t++){
            tmpw=rootsPtr[partialBitRev(t<<1,s)*m];
            lPtr1=newv1Ptr+start; rPtr1=lPtr1+m;
            t1=MontMulMod_OPT2_AS_GENE_SPE(rPtr1[0],tmpw, pPtr);
            lPtr2=newv2Ptr+start; rPtr2=lPtr2+m;
            t2=MontMulMod_OPT2_AS_GENE_SPE(rPtr2[0],tmpw, pPtr);

            tmp=m-1;
            // inner-loop begin
            for (u=0; u<tmp; u++ ) {
                rPtr1[u]=SubMod(lPtr1[u],t1,p);
                lPtr1[u]=AddMod(lPtr1[u],t1,p);
                tt1=MontMulMod_OPT2_AS_GENE_SPE(rPtr1[u+1],tmpw, pPtr);
                t1=tt1;
                rPtr2[u]=SubMod(lPtr2[u],t2,p);
                lPtr2[u]=AddMod(lPtr2[u],t2,p);
                tt2=MontMulMod_OPT2_AS_GENE_SPE(rPtr2[u+1],tmpw, pPtr);
                t2=tt2;
            } // inner-loop end.

            rPtr1[tmp]=SubMod(lPtr1[tmp],t1,p);
            lPtr1[tmp]=AddMod(lPtr1[tmp],t1,p);
 
            rPtr2[tmp]=SubMod(lPtr2[tmp],t2,p);
            lPtr2[tmp]=AddMod(lPtr2[tmp],t2,p);
            start+=(m<<1);
        } // middle-loop end.
        // the last group only need to compute half bufferflies.
        if(Rno){
            tmpw=rootsPtr[partialBitRev(Tno<<1,s)*m];
            for (u=start; u<start+m; u++ ){
	            lPtr1=newv1Ptr+u;
                lPtr1[0]=AddMod(lPtr1[0],MontMulMod_OPT2_AS_GENE_SPE(lPtr1[m],tmpw, pPtr),p);
                lPtr2=newv2Ptr+u;
                lPtr2[0]=AddMod(lPtr2[0],MontMulMod_OPT2_AS_GENE_SPE(lPtr2[m],tmpw, pPtr),p);
            }
        } // if(Rno) end
        m>>=1;
    } // DFTs end ==============================================>

    for(i=0; i<l; i++) newv1Ptr[i]=MontMulMod_OPT2_AS_GENE_SPE(newv1Ptr[i], newv2Ptr[i]<<BRsft, pPtr);

    // INV-DFT starts
    // step-1: left FFT-tree bottom uPpter->

    wPtr=(sfixn *)my_calloc(halfn, sizeof(sfixn));
    end1=(l>>1)<<1;   
    for(i=0; i<end1; i+=2){
        tmp=newv1Ptr[i];
        newv1Ptr[i]=AddMod(newv1Ptr[i],newv1Ptr[i+1],p);
        newv1Ptr[i+1]=SubMod(tmp,newv1Ptr[i+1],p);
    }
    m=2;
    wPtr[0]=rootsPtr[0];
    for(s=power-2; s>0; s--){
        for(i=1; i<m; i++) wPtr[i]=rootsPtr[n-(i<<s)];
        //start=0;
        tmp=m<<1; 
        end2=(l/tmp)*tmp;
        for(start=0; start<end2; start+=tmp){
	        tmpPtr=newv1Ptr+start;
            for(u=0; u<m; u++) bfOpt_new_0_SPE(tmpPtr+u, tmpPtr+u+m, wPtr[u], pPtr);
        }
        m<<=1;
   }

    //step-2: Compute the top level.
    Bno=halfn;
    for(i=l; i<n; i++){
        tmp=i-Bno;
        newv1Ptr[i]= MontMulMod_OPT2_AS_GENE_SPE(newv1Ptr[tmp], rootsPtr[tmp], pPtr);
        newv1Ptr[tmp]=AddMod(newv1Ptr[tmp],newv1Ptr[tmp],p); 
    }

    //step-3: right part of FFT-tree, top-down go  as deep as possible.
    Bno=halfn>>1;
    for (s=1; s<power; s++ ){
        tmp=lsPtr[s]-l;
        // later on we will use inverse of roots.
        if(tmp<Bno){
            tmp2=Bno-tmp;
            for(i=0; i<tmp; i++ ) wPtr[i]=rootsPtr[(i+tmp2)<<s];
            end1=lsPtr[s]-l;
            tmpPtr=newv1Ptr+l;
            for(t=0; t<end1; t++ ) bfOpt_new_SPE_1(tmpPtr+t-Bno, tmpPtr+t, wPtr+t, pPtr);
        }
        if(tmp>Bno){ 
            end1=lsPtr[s]-Bno;
            for(tmpPtr=newv1Ptr+l; tmpPtr<newv1Ptr+end1; tmpPtr++ )
                lhalfbfOpt_new_2_SPE(tmpPtr, tmpPtr+Bno, inv12RSFT, pPtr);
        }
 
        if(tmp==Bno){
            lPtr1=newv1Ptr-Bno+l; tmpPtr=newv1Ptr+l;
            for(; lPtr1<tmpPtr; lPtr1++) lhalfBfOpt_new_SPE_1(lPtr1, lPtr1+Bno, pPtr);
            left=l-Bno; right=l;
            break;
        }
        Bno>>=1; 
    }

    //step-4: bottom up again.
    wPtr[0]=rootsPtr[0];
    for(; s>1; s--){
        Bno<<=1;
        if(lsPtr[s]==lsPtr[s-1]){
            for(i=1; i<Bno; i++ ) wPtr[i]=rootsPtr[n-(i<<(s-1))];
            for(t=left; t<right; t++) bfOpt_new_0_SPE(newv1Ptr+t-Bno, newv1Ptr+t, wPtr[t-left], pPtr);
                 left=left-Bno;
        } else {
        for(t=left; t<right; t++) lhalfBfOpt_new_SPE_1(newv1Ptr+t, newv1Ptr+t+Bno, pPtr);}
    }

    //step-5: fill up the missing points at the top level.
    for(i=1; i<halfn; i++ ) wPtr[i]=rootsPtr[n-i];
    for(i=0; i<l-halfn; i++) bfOpt_new_0(newv1Ptr+i, newv1Ptr+i+halfn, wPtr[i],pPtr);  

    invn=inverseMod(n,p);
    invn=MulMod(R,invn,p);
    invn=MulMod(R,invn,p);
    invn<<=BRsft;
    for(i=0; i<l; i++) resPtr[i]=MontMulMod_OPT2_AS_GENE_SPE(newv1Ptr[i], invn, pPtr);
    
    my_free(lsPtr); 
    my_free(newv1Ptr);
    my_free(newv2Ptr); 
    my_free(rootsPtr);
    my_free(wPtr);
}



// note: WHOLE means the TFT algorithm is not seperated into TDFT and TINVDFT.
/**
 * EX_Mont_TFTMul_OPT2_AS_GENE_WHOLE:
 * @resPtr: coefficient vector of result
 * @d1: degree of A
 * @v1Ptr: coefficient vector of A 
 * @d2: degree of B
 * @v2Ptr: coefficient vector of B
 * @pPtr: prime number structure
 * 
 * Exported version.
 * 
 * Return value: the product of two polynomials.
 **/  
void EX_Mont_TFTMul_OPT2_AS_GENE_WHOLE(sfixn * resPtr, sfixn d1, sfixn * v1Ptr,
     sfixn d2, sfixn * v2Ptr, MONTP_OPT2_AS_GENE * pPtr){
    if((pPtr->c_pow==0) || ((pPtr->c_pow+pPtr->Rpow)>=BASE) ){
        //printf("Using TFT for general FFT prime.");
        Mont_TFTMul_OPT2_AS_GENE(resPtr, d1, v1Ptr, d2, v2Ptr, pPtr);
    }else{
        //printf("Using TFT for special FFT prime.");
        Mont_TFTMul_OPT2_AS_GENE_SPE(resPtr, d1, v1Ptr, d2, v2Ptr, pPtr);
    }
}

//==============================================================================
// TDFT
//==============================================================================

void Mont_TDFT_OPT2_AS_GENE_1(sfixn l, sfixn * rootsPtr, sfixn * newv1Ptr, MONTP_OPT2_AS_GENE * pPtr){

    sfixn i;
    sfixn p, n=1, m, halfn, power=0, s, t, u, Tno, Rno, start=0, tmp, tmpw;
    sfixn *lsPtr, *lPtr1, *rPtr1;
    register sfixn t1,tt1;
    
    // compute the n=2^power.
    for(i=l; i>1; i>>=1) {n<<=1; power++;}
    if (n<l) {n<<=1; power++;}  
    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }
    if (l==n) return EX_Mont_DFT_OPT2_AS_GENE_1( n, power, rootsPtr, newv1Ptr, pPtr);

    p=pPtr->P;
    halfn=n>>1;
    // lsPtr[0] not in use.
    lsPtr=(sfixn *)my_calloc(power+1,sizeof(sfixn));

    // lsPtr[i] keeps the number of points needed to compute at level i 
    m=halfn;
    for(i=1; i<=power; i++) {
        //WAS lsPtr[i]=(l/m+1)*m;
        //WAS m>>=1;
        lsPtr[i] = (((l-1)>>(power-i))+1)<<(power-i);
    }
    // DFTs begin=========================================>
    // compute the top level. (level-1 to level-2).
    for (i=0; i<(l-halfn); i++) {
        tmp=newv1Ptr[i];
        newv1Ptr[i]=AddMod(newv1Ptr[i],newv1Ptr[i+halfn],p);
        newv1Ptr[i+halfn]=SubMod(tmp,newv1Ptr[i+halfn],p);
    }
    for(i=(halfn-(n-l)); i<halfn;i++){
        newv1Ptr[i+halfn]=newv1Ptr[i];
    }

    // compute the rest levels.
    m=halfn>>1;
    for (s=2; s<=power; s++){
        // # of teams with "full butterflies";
        Tno=lsPtr[s]/(m<<1);
        // # of "half bufflies" in the last team;
        Rno=lsPtr[s]%(m<<1);
        start=0;

        // middle-loop begin
        for (t=0; t<Tno; t++){
            tmpw=rootsPtr[partialBitRev(t<<1,s)*m];
            lPtr1=newv1Ptr+start; rPtr1=lPtr1+m;
            t1=MontMulMod_OPT2_AS_GENE(rPtr1[0],tmpw, pPtr);
            tmp=m-1;
             // inner-loop begin
            for (u=0; u<tmp; u++ ) {
                rPtr1[u]=SubMod(lPtr1[u],t1,p);
                lPtr1[u]=AddMod(lPtr1[u],t1,p);
                tt1=MontMulMod_OPT2_AS_GENE(rPtr1[u+1],tmpw, pPtr);
                t1=tt1;
            } // inner-loop end.
            rPtr1[tmp]=SubMod(lPtr1[tmp],t1,p);
            lPtr1[tmp]=AddMod(lPtr1[tmp],t1,p);
            start+=(m<<1);
        } // middle-loop end.
        // the last group only need to compute half bufferflies.
        if(Rno){
            tmpw=rootsPtr[partialBitRev(Tno<<1,s)*m];
            for (u=start; u<start+m; u++ ){
	            lPtr1=newv1Ptr+u;
                lPtr1[0]=AddMod(lPtr1[0],MontMulMod_OPT2_AS_GENE(lPtr1[m],tmpw, pPtr),p);
            }
        } // if(Rno) end
        m>>=1;
    } // DFTs end ==============================================>
    my_free(lsPtr);
}

void Mont_TDFT_OPT2_AS_GENE_SPE_1(sfixn l, sfixn * rootsPtr, sfixn * newv1Ptr, MONTP_OPT2_AS_GENE * pPtr){

    sfixn i;
    sfixn p, n=1, m, halfn, power=0, s, t, u, Tno, Rno, start=0, tmp, tmpw;
    sfixn * lsPtr, * lPtr1, * rPtr1;
    register sfixn t1,tt1;

    // compute the n=2^power.
    for(i=l; i>1; i>>=1) {n<<=1; power++;}
    if (n<l) {n<<=1; power++;}  
    
    if(power > (pPtr->Npow)){
        Interrupted=1; PrimeError=1;
        return;
    }

    if (l==n) return EX_Mont_DFT_OPT2_AS_GENE_1( n, power, rootsPtr, newv1Ptr, pPtr);

    p=pPtr->P;
    halfn=n>>1;

    // lsPtr[0] not in use.
    lsPtr=(sfixn *)my_calloc(power+1,sizeof(sfixn));
    // lsPtr[i] keeps the number of points needed to compute at level i for poly1/2 DFT tree
    m=halfn;
    for(i=1; i<=power; i++ ){
        // WAS lsPtr[i]=(l/m+1)*m;
        // WAS m>>=1;
        // The number of points has been adjusted according the TFT paper WP
        lsPtr[i] = (((l-1)>>(power-i))+1)<<(power-i);
    }
    // DFTs begin
    // compute the top level. (level-1 to level-2).
    for (i=0; i<(l-halfn); i++) {
        tmp=newv1Ptr[i];
        newv1Ptr[i]=AddMod(newv1Ptr[i],newv1Ptr[i+halfn],p);
        newv1Ptr[i+halfn]=SubMod(tmp,newv1Ptr[i+halfn],p);
    }
    for(i=(halfn-(n-l)); i<halfn;i++){
        newv1Ptr[i+halfn]=newv1Ptr[i];
    }

    // compute the rest levels.
    m=halfn>>1;
    for (s=2; s<=power; s++){
        // # of teams with "full butterflies";
        Tno=lsPtr[s]/(m<<1);
        // # of "half bufflies" in the last team;
        Rno=lsPtr[s]%(m<<1);
        start=0;
        // middle-loop begin
        for (t=0; t<Tno; t++){
            tmpw=rootsPtr[partialBitRev(t<<1,s)*m];
            lPtr1=newv1Ptr+start; rPtr1=lPtr1+m;
            t1=MontMulMod_OPT2_AS_GENE_SPE(rPtr1[0],tmpw, pPtr);
            tmp=m-1;
            // inner-loop begin
            for (u=0; u<tmp; u++ ) {
                rPtr1[u]=SubMod(lPtr1[u],t1,p);
                lPtr1[u]=AddMod(lPtr1[u],t1,p);
                tt1=MontMulMod_OPT2_AS_GENE_SPE(rPtr1[u+1],tmpw, pPtr);
                t1=tt1;
            }// inner-loop end.

            rPtr1[tmp]=SubMod(lPtr1[tmp],t1,p);
            lPtr1[tmp]=AddMod(lPtr1[tmp],t1,p);
            start+=(m<<1);
        }// middle-loop end.
        
        // the last group only need to compute half bufferflies.
        if(Rno){
            tmpw=rootsPtr[partialBitRev(Tno<<1,s)*m];
            for (u=start; u<start+m; u++ ){
	            lPtr1=newv1Ptr+u;
                lPtr1[0]=AddMod(lPtr1[0],MontMulMod_OPT2_AS_GENE_SPE(lPtr1[m],tmpw, pPtr),p);
            }
        }// if(Rno) end
        m>>=1;
    } // DFTs end 
    my_free(lsPtr);
}

// type: exported.
 /**
 * EX_Mont_TDFT_OPT2_AS_GENE_1:
 * @l: the TFT size
 * @rootsPtr: powers of the n-th primitive root, n is the next power of 2 of l.
 * @tmpVecPtr:  work space (int array of size n)
 * @pPtr: prime number structure
 * 
 * Exported verson
 * 
 * Return value: DFT  of the input polynomial.
 **/   

void EX_Mont_TDFT_OPT2_AS_GENE_1 ( sfixn l, sfixn * rootsPtr, sfixn * tmpVecPtr, MONTP_OPT2_AS_GENE * pPtr){

    if((pPtr->c_pow==0) || ((pPtr->c_pow+pPtr->Rpow)>=BASE) ){
        Mont_TDFT_OPT2_AS_GENE_1(l, rootsPtr,tmpVecPtr, pPtr);   
    }else{
        Mont_TDFT_OPT2_AS_GENE_SPE_1(l, rootsPtr,tmpVecPtr, pPtr);
    }
}

//==============================================================================
// TINVDFT
//==============================================================================

// type: local, in-place

void Mont_INVTDFT_OPT2_AS_GENE_1(sfixn l, sfixn * rootsPtr, sfixn * newv1Ptr, MONTP_OPT2_AS_GENE * pPtr){
    register sfixn i;
    // l is the size of result poly. 
    sfixn p, BRsft, inv12RSFT, R, n=1, m, halfn, power=0;
    sfixn s, t, u, invn, Bno, start=0, tmp, tmp2, left=0, right=0,  end1, end2;
    sfixn *lsPtr,  *wPtr, *lPtr1,  *tmpPtr;

    // compute the n=2^power.
    for(i=l; i>1; i>>=1) {n<<=1; power++;}
    if (n<l) {n<<=1; power++;}  

    if (l==n) return EX_Mont_INVDFT_OPT2_AS_GENE_1( n, power, rootsPtr, newv1Ptr, pPtr);

    p=pPtr->P;
    BRsft=pPtr->Base_Rpow;
    R=(1L<<(pPtr->Rpow))%p; 
    inv12RSFT=inverseMod(2,p);
    inv12RSFT=MulMod(R,inv12RSFT,p);
    inv12RSFT=inv12RSFT<<BRsft;
    halfn=n>>1;

    lsPtr=(sfixn *)my_calloc(power+1,sizeof(sfixn));
    m=halfn;
    for(i=1; i<=power; i++ ){
        // WAS lsPtr[i]=(l/m+1)*m;
        // WAS m>>=1;
        lsPtr[i] = (((l-1)>>(power-i))+1)<<(power-i);
    }
    wPtr=(sfixn *)my_calloc(halfn, sizeof(sfixn));

    end1=(l>>1)<<1;   
    for(i=0; i<end1; i+=2){
        tmp=newv1Ptr[i];
        newv1Ptr[i]=AddMod(newv1Ptr[i],newv1Ptr[i+1],p);
        newv1Ptr[i+1]=SubMod(tmp,newv1Ptr[i+1],p);
    }
    m=2;
    wPtr[0]=rootsPtr[0];
    for(s=power-2; s>0; s--){
        for(i=1; i<m; i++) wPtr[i]=rootsPtr[n-(i<<s)];
        tmp=m<<1; 
        end2=(l/tmp)*tmp;
        for(start=0; start<end2; start+=tmp){
	        tmpPtr=newv1Ptr+start;
            for(u=0; u<m; u++) bfOpt_new_0(tmpPtr+u, tmpPtr+u+m, wPtr[u], pPtr);
        }
        m<<=1;
    }

    //step-2: Compute the top level.
    
    Bno=halfn;
    for(i=l; i<n; i++){
        tmp=i-Bno;
        newv1Ptr[i]= MontMulMod_OPT2_AS_GENE(newv1Ptr[tmp], rootsPtr[tmp], pPtr);
        newv1Ptr[tmp]=AddMod(newv1Ptr[tmp],newv1Ptr[tmp],p); 
    }

    //step-3: right part of FFT-tree, top-down go  as deep as possible.
    Bno=halfn>>1;
    for (s=1; s<power; s++ ){
        tmp=lsPtr[s]-l;
        // later on we will use inverse of roots.
        if(tmp<Bno){
            tmp2=Bno-tmp;
            for(i=0; i<tmp; i++ ) wPtr[i]=rootsPtr[(i+tmp2)<<s];
            end1=lsPtr[s]-l;
            tmpPtr=newv1Ptr+l;
            for(t=0; t<end1; t++ ) bfOpt_new_1(tmpPtr+t-Bno, tmpPtr+t, wPtr+t, pPtr);
        }
        if(tmp>Bno){ 
            end1=lsPtr[s]-Bno;
            for(tmpPtr=newv1Ptr+l; tmpPtr<newv1Ptr+end1; tmpPtr++ )
                lhalfbfOpt_new_2(tmpPtr, tmpPtr+Bno, inv12RSFT, pPtr);
        }
        if(tmp==Bno){
            lPtr1=newv1Ptr-Bno+l; tmpPtr=newv1Ptr+l;
            for(; lPtr1<tmpPtr; lPtr1++) lhalfBfOpt_new_1(lPtr1, lPtr1+Bno, pPtr);
            left=l-Bno; right=l;
            break;
        }
        Bno>>=1; 
    }

    //step-4: bottom up again.
    wPtr[0]=rootsPtr[0];
    for(; s>1; s--){
        Bno<<=1;
        if(lsPtr[s]==lsPtr[s-1]){
            for(i=1; i<Bno; i++ ) wPtr[i]=rootsPtr[n-(i<<(s-1))];
            for(t=left; t<right; t++) bfOpt_new_0(newv1Ptr+t-Bno, newv1Ptr+t, wPtr[t-left], pPtr);
            left=left-Bno;
        }else{
           for(t=left; t<right; t++) lhalfBfOpt_new_1(newv1Ptr+t, newv1Ptr+t+Bno, pPtr);
        }
    }

    //step-5: fill up the missing points at the top level.
    for(i=1; i<halfn; i++ ) wPtr[i]=rootsPtr[n-i];
    for(i=0; i<l-halfn; i++) bfOpt_new_0(newv1Ptr+i, newv1Ptr+i+halfn, wPtr[i],pPtr);  

    invn=inverseMod(n,p);
    invn=MulMod(R,invn,p);
    invn<<=BRsft;
    for(i=0; i<l; i++) newv1Ptr[i]=MontMulMod_OPT2_AS_GENE(newv1Ptr[i], invn, pPtr);
    my_free(wPtr);
    my_free(lsPtr);
}

void Mont_INVTDFT_OPT2_AS_GENE_SPE_1(sfixn l, sfixn * rootsPtr, sfixn * newv1Ptr, MONTP_OPT2_AS_GENE * pPtr){
    register sfixn i;
    // l is the size of result poly. 
    sfixn p, BRsft, inv12RSFT, R,  n=1, m, halfn, power=0;
    sfixn s, t, u, invn, Bno, start=0, tmp, tmp2, left=0, right=0,  end1, end2;
    sfixn *lsPtr,  *wPtr, *lPtr1,  *tmpPtr;

    // compute the n=2^power.
    for(i=l; i>1; i>>=1) {n<<=1; power++;}
    if (n<l) {n<<=1; power++;}  

    if (l==n) return EX_Mont_INVDFT_OPT2_AS_GENE_1( n, power, rootsPtr, newv1Ptr, pPtr);

    p=pPtr->P;
    BRsft=pPtr->Base_Rpow;
    R=(1L<<(pPtr->Rpow))%p; 
    inv12RSFT=inverseMod(2,p);
    inv12RSFT=MulMod(R,inv12RSFT,p);
    inv12RSFT=inv12RSFT<<BRsft;
    halfn=n>>1;

    lsPtr=(sfixn *)my_calloc(power+1,sizeof(sfixn));
    m=halfn;
    for(i=1; i<=power; i++ ){
        // WAS lsPtr[i]=(l/m+1)*m;
        // WAS m>>=1;
        lsPtr[i] = (((l-1)>>(power-i))+1)<<(power-i);
    }

    wPtr=(sfixn *)my_calloc(halfn, sizeof(sfixn));

    end1=(l>>1)<<1;   
    for(i=0; i<end1; i+=2){
        tmp=newv1Ptr[i];
        newv1Ptr[i]=AddMod(newv1Ptr[i],newv1Ptr[i+1],p);
        newv1Ptr[i+1]=SubMod(tmp,newv1Ptr[i+1],p);
    }
    m=2;
    wPtr[0]=rootsPtr[0];
    for(s=power-2; s>0; s--){
        for(i=1; i<m; i++) wPtr[i]=rootsPtr[n-(i<<s)];
        tmp=m<<1; 
        end2=(l/tmp)*tmp;
        for(start=0; start<end2; start+=tmp){
	        tmpPtr=newv1Ptr+start;
            for(u=0; u<m; u++) bfOpt_new_0_SPE(tmpPtr+u, tmpPtr+u+m, wPtr[u], pPtr);
        }
        m<<=1;
    }

    //step-2: Compute the top level.
    Bno=halfn;
    for(i=l; i<n; i++){
        tmp=i-Bno;
        newv1Ptr[i]= MontMulMod_OPT2_AS_GENE_SPE(newv1Ptr[tmp], rootsPtr[tmp], pPtr);
        newv1Ptr[tmp]=AddMod(newv1Ptr[tmp],newv1Ptr[tmp],p); 
    }   

    //step-3: right part of FFT-tree, top-down go  as deep as possible.
    Bno=halfn>>1;
    for (s=1; s<power; s++ ){
        tmp=lsPtr[s]-l;
        // later on we will use inverse of roots.
   
        if(tmp<Bno){
            tmp2=Bno-tmp;
            for(i=0; i<tmp; i++ ) wPtr[i]=rootsPtr[(i+tmp2)<<s];
            end1=lsPtr[s]-l;
            tmpPtr=newv1Ptr+l;
            for(t=0; t<end1; t++ ) bfOpt_new_SPE_1(tmpPtr+t-Bno, tmpPtr+t, wPtr+t, pPtr);
        }
        if(tmp>Bno){ 
            end1=lsPtr[s]-Bno;
            for(tmpPtr=newv1Ptr+l; tmpPtr<newv1Ptr+end1; tmpPtr++ )
                lhalfbfOpt_new_2_SPE(tmpPtr, tmpPtr+Bno, inv12RSFT, pPtr);
        }
        if(tmp==Bno){
            lPtr1=newv1Ptr-Bno+l; tmpPtr=newv1Ptr+l;
            for(; lPtr1<tmpPtr; lPtr1++) lhalfBfOpt_new_SPE_1(lPtr1, lPtr1+Bno, pPtr);
            left=l-Bno; right=l;
            break;
        }
        Bno>>=1; 
    }

    //step-4: bottom up again.
    wPtr[0]=rootsPtr[0];
 
    for(; s>1; s--){
        Bno<<=1;
        if(lsPtr[s]==lsPtr[s-1]){
            for(i=1; i<Bno; i++ ) wPtr[i]=rootsPtr[n-(i<<(s-1))];
            for(t=left; t<right; t++) bfOpt_new_0_SPE(newv1Ptr+t-Bno, newv1Ptr+t, wPtr[t-left], pPtr);
            left=left-Bno;
        }else{
            for(t=left; t<right; t++) lhalfBfOpt_new_SPE_1(newv1Ptr+t, newv1Ptr+t+Bno, pPtr);
        }
    }

    //step-5: fill up the missing points at the top level.
    for(i=1; i<halfn; i++ ) wPtr[i]=rootsPtr[n-i];
    for(i=0; i<l-halfn; i++) bfOpt_new_0(newv1Ptr+i, newv1Ptr+i+halfn, wPtr[i],pPtr);  

    invn=inverseMod(n,p);
    invn=MulMod(R,invn,p);
    invn<<=BRsft;
    for(i=0; i<l; i++) newv1Ptr[i]=MontMulMod_OPT2_AS_GENE_SPE(newv1Ptr[i], invn, pPtr);
    my_free(wPtr);
    my_free(lsPtr);
}

 // type: exported.
 /**
 * EX_Mont_INVTDFT_OPT2_AS_GENE_1:
 * @l: the TFT size
 * @rootsPtr: powers of the n-th primitive root, n is the next power of 2 of l.
 * @tmpVecPtr:  work space (int array of size n)
 * @pPtr: prime number structure
 * 
 * Exported verson
 * 
 * Return value: Inverse DFT  of the input polynomial.
 **/   

void EX_Mont_INVTDFT_OPT2_AS_GENE_1 ( sfixn l, sfixn * rootsPtr, sfixn * tmpVecPtr, MONTP_OPT2_AS_GENE * pPtr){

    if((pPtr->c_pow==0) || ((pPtr->c_pow+pPtr->Rpow)>=BASE) ){
        Mont_INVTDFT_OPT2_AS_GENE_1(l, rootsPtr, tmpVecPtr, pPtr);   
    }else{
        Mont_INVTDFT_OPT2_AS_GENE_SPE_1(l, rootsPtr, tmpVecPtr, pPtr);
    }
}

void Mont_INVTDFT_OPT2_AS_GENE_R_1(sfixn l, sfixn * rootsPtr, sfixn * newv1Ptr, MONTP_OPT2_AS_GENE * pPtr){
    register sfixn i;
    // l is the size of result poly. 
    sfixn p, BRsft, inv12RSFT, R,  n=1, m, halfn, power=0;
    sfixn s, t, u, invn, Bno, start=0, tmp, tmp2, left=0, right=0,  end1, end2;
    sfixn * lsPtr,  * wPtr, * lPtr1,  * tmpPtr;

    // compute the n=2^power.
    for(i=l; i>1; i>>=1) {n<<=1; power++;}
    if (n<l) {n<<=1; power++;}  

    if (l==n) return EX_Mont_INVDFT_OPT2_AS_GENE_R_1( n, power, rootsPtr, newv1Ptr, pPtr);
    
    p=pPtr->P;
    BRsft=pPtr->Base_Rpow;
    R=(1L<<(pPtr->Rpow))%p;
    inv12RSFT=inverseMod(2,p);
    inv12RSFT=MulMod(R,inv12RSFT,p);
    inv12RSFT=inv12RSFT<<BRsft;
    halfn=n>>1;

    lsPtr=(sfixn *)my_calloc(power+1,sizeof(sfixn));
    m=halfn;
    for(i=1; i<=power; i++ ){
        // WAS lsPtr[i]=(l/m+1)*m;
        // WAS m>>=1; 
        lsPtr[i] = (((l-1)>>(power-i))+1)<<(power-i);
    }

    wPtr=(sfixn *)my_calloc(halfn, sizeof(sfixn));

    end1=(l>>1)<<1;   
    for (i=0; i<end1; i+=2){
        tmp=newv1Ptr[i];
        newv1Ptr[i]=AddMod(newv1Ptr[i],newv1Ptr[i+1],p);
        newv1Ptr[i+1]=SubMod(tmp,newv1Ptr[i+1],p);
    }
    m=2;
    wPtr[0]=rootsPtr[0];
    for(s=power-2; s>0; s--){
        for(i=1; i<m; i++) wPtr[i]=rootsPtr[n-(i<<s)];
        tmp=m<<1; 
        end2=(l/tmp)*tmp;
        for(start=0; start<end2; start+=tmp){
	        tmpPtr=newv1Ptr+start;
            for(u=0; u<m; u++) bfOpt_new_0(tmpPtr+u, tmpPtr+u+m, wPtr[u], pPtr);
        }
        m<<=1;
    }

    //step-2: Compute the top level.
    Bno=halfn;
    for(i=l; i<n; i++){
        tmp=i-Bno;
        newv1Ptr[i]= MontMulMod_OPT2_AS_GENE(newv1Ptr[tmp], rootsPtr[tmp], pPtr);
        newv1Ptr[tmp]=AddMod(newv1Ptr[tmp],newv1Ptr[tmp],p); 
    }

    //step-3: right part of FFT-tree, top-down go  as deep as possible.
    Bno=halfn>>1;
    for (s=1; s<power; s++ ){
        tmp=lsPtr[s]-l;
        // later on we will use inverse of roots.
        if(tmp<Bno){
            tmp2=Bno-tmp;
            for(i=0; i<tmp; i++ ) wPtr[i]=rootsPtr[(i+tmp2)<<s];
            end1=lsPtr[s]-l;
            tmpPtr=newv1Ptr+l;
            for(t=0; t<end1; t++ ) bfOpt_new_1(tmpPtr+t-Bno, tmpPtr+t, wPtr+t, pPtr);
        }
        if(tmp>Bno){ 
            end1=lsPtr[s]-Bno;
            for(tmpPtr=newv1Ptr+l; tmpPtr<newv1Ptr+end1; tmpPtr++)
                lhalfbfOpt_new_2(tmpPtr, tmpPtr+Bno, inv12RSFT, pPtr);
        }
        if(tmp==Bno){
            lPtr1=newv1Ptr-Bno+l; tmpPtr=newv1Ptr+l;
            for(; lPtr1<tmpPtr; lPtr1++) lhalfBfOpt_new_1(lPtr1, lPtr1+Bno, pPtr);
            left=l-Bno; right=l;
            break;
        }
        Bno>>=1; 
    }
 
    //step-4: bottom up again.
    wPtr[0]=rootsPtr[0];
    
    for(; s>1; s--){
        Bno<<=1;
        if(lsPtr[s]==lsPtr[s-1]){
            for(i=1; i<Bno; i++ ) wPtr[i]=rootsPtr[n-(i<<(s-1))];
            for(t=left; t<right; t++) bfOpt_new_0(newv1Ptr+t-Bno, newv1Ptr+t, wPtr[t-left], pPtr);
            left=left-Bno;
        }else{
            for(t=left; t<right; t++) lhalfBfOpt_new_1(newv1Ptr+t, newv1Ptr+t+Bno, pPtr);}
    }

    //step-5: fill up the missing points at the top level.
    for(i=1; i<halfn; i++ ) wPtr[i]=rootsPtr[n-i];
    for(i=0; i<l-halfn; i++) bfOpt_new_0(newv1Ptr+i, newv1Ptr+i+halfn, wPtr[i],pPtr);  

    invn=inverseMod(n,p);
    invn=MulMod(R,invn,p);
    invn=MulMod(R,invn,p);
    invn<<=BRsft;
    for(i=0; i<l; i++) newv1Ptr[i]=MontMulMod_OPT2_AS_GENE(newv1Ptr[i], invn, pPtr);
    my_free(wPtr);
    my_free(lsPtr);
}

void Mont_INVTDFT_OPT2_AS_GENE_SPE_R_1(sfixn l, sfixn * rootsPtr, sfixn * newv1Ptr, MONTP_OPT2_AS_GENE * pPtr){
    register sfixn i;
    // l is the size of result poly. 
    sfixn p, BRsft, inv12RSFT, R,  n=1, m, halfn, power=0;
    sfixn s, t, u, invn, Bno, start=0, tmp, tmp2, left=0, right=0,  end1, end2;
    sfixn * lsPtr,  * wPtr, * lPtr1,  * tmpPtr;

    // compute the n=2^power.
    for(i=l; i>1; i>>=1) {n<<=1; power++;}
    if (n<l) {n<<=1; power++;}  

    if (l==n) return EX_Mont_INVDFT_OPT2_AS_GENE_R_1( n, power, rootsPtr, newv1Ptr, pPtr);

    p=pPtr->P;
    BRsft=pPtr->Base_Rpow;
    R=(1L<<(pPtr->Rpow))%p;
    inv12RSFT=inverseMod(2,p);
    inv12RSFT=MulMod(R,inv12RSFT,p);
    inv12RSFT=inv12RSFT<<BRsft;
    halfn=n>>1;

    lsPtr=(sfixn *)my_calloc(power+1,sizeof(sfixn));
    m=halfn;
    for(i=1; i<=power; i++ ){
        // WAS lsPtr[i]=(l/m+1)*m;
        // WAS m>>=1;
        lsPtr[i] = (((l-1)>>(power-i))+1)<<(power-i);
    }
    wPtr=(sfixn *)my_calloc(halfn, sizeof(sfixn));
    end1=(l>>1)<<1;   
    for(i=0; i<end1; i+=2){
        tmp=newv1Ptr[i];
        newv1Ptr[i]=AddMod(newv1Ptr[i],newv1Ptr[i+1],p);
        newv1Ptr[i+1]=SubMod(tmp,newv1Ptr[i+1],p);
    }
    m=2;
    wPtr[0]=rootsPtr[0];
    for(s=power-2; s>0; s--){
        for(i=1; i<m; i++) wPtr[i]=rootsPtr[n-(i<<s)];
        tmp=m<<1; 
        end2=(l/tmp)*tmp;
        for(start=0; start<end2; start+=tmp){
	        tmpPtr=newv1Ptr+start;
            for(u=0; u<m; u++) bfOpt_new_0_SPE(tmpPtr+u, tmpPtr+u+m, wPtr[u], pPtr);
        }
        m<<=1;
    }

    //step-2: Compute the top level.
    Bno=halfn;
    for(i=l; i<n; i++){
        tmp=i-Bno;
        newv1Ptr[i]= MontMulMod_OPT2_AS_GENE_SPE(newv1Ptr[tmp], rootsPtr[tmp], pPtr);
        newv1Ptr[tmp]=AddMod(newv1Ptr[tmp],newv1Ptr[tmp],p); 
    }

    //step-3: right part of FFT-tree, top-down go  as deep as possible.
    Bno=halfn>>1;
    for (s=1; s<power; s++ ){
        tmp=lsPtr[s]-l;
        // later on we will use inverse of roots.
        
        if(tmp<Bno){
            tmp2=Bno-tmp;
            for(i=0; i<tmp; i++ ) wPtr[i]=rootsPtr[(i+tmp2)<<s];
            end1=lsPtr[s]-l;
            tmpPtr=newv1Ptr+l;
            for(t=0; t<end1; t++ ) bfOpt_new_SPE_1(tmpPtr+t-Bno, tmpPtr+t, wPtr+t, pPtr);
        }
        if(tmp>Bno){ 
            end1=lsPtr[s]-Bno;
            for(tmpPtr=newv1Ptr+l; tmpPtr<newv1Ptr+end1; tmpPtr++ ) 
                  lhalfbfOpt_new_2_SPE(tmpPtr, tmpPtr+Bno, inv12RSFT, pPtr);
        }
        if(tmp==Bno){
            lPtr1=newv1Ptr-Bno+l; tmpPtr=newv1Ptr+l;
            for(; lPtr1<tmpPtr; lPtr1++) lhalfBfOpt_new_SPE_1(lPtr1, lPtr1+Bno, pPtr);
            left=l-Bno; right=l;
            break;
        }
        Bno>>=1; 
    }

    //step-4: bottom up again.
    wPtr[0]=rootsPtr[0];
    for(; s>1; s--){
        Bno<<=1;
        if(lsPtr[s]==lsPtr[s-1]){
            for(i=1; i<Bno; i++ ) wPtr[i]=rootsPtr[n-(i<<(s-1))];
            for(t=left; t<right; t++) bfOpt_new_0_SPE(newv1Ptr+t-Bno, newv1Ptr+t, wPtr[t-left], pPtr);
            left=left-Bno;
        } else{
            for(t=left; t<right; t++) lhalfBfOpt_new_SPE_1(newv1Ptr+t, newv1Ptr+t+Bno, pPtr);
        }
    }

    //step-5: fill up the missing points at the top level.
    for(i=1; i<halfn; i++ ) wPtr[i]=rootsPtr[n-i];
    for(i=0; i<l-halfn; i++) bfOpt_new_0(newv1Ptr+i, newv1Ptr+i+halfn, wPtr[i],pPtr);  

    invn=inverseMod(n,p);
    invn=MulMod(R,invn,p);
    invn=MulMod(R,invn,p);
    invn<<=BRsft;
    for(i=0; i<l; i++) newv1Ptr[i]=MontMulMod_OPT2_AS_GENE_SPE(newv1Ptr[i], invn, pPtr);
    my_free(wPtr);
    my_free(lsPtr);
}

void EX_Mont_INVTDFT_OPT2_AS_GENE_R_1 ( sfixn l, sfixn * rootsPtr, sfixn * tmpVecPtr, MONTP_OPT2_AS_GENE * pPtr){

    if((pPtr->c_pow==0) || ((pPtr->c_pow+pPtr->Rpow)>=BASE) ){
        Mont_INVTDFT_OPT2_AS_GENE_R_1(l, rootsPtr, tmpVecPtr, pPtr);   
    } else{
        Mont_INVTDFT_OPT2_AS_GENE_SPE_R_1(l, rootsPtr, tmpVecPtr, pPtr);
    }
}

//==============================================================================
// TFT Mul (TDFT/TINVDFT)
//==============================================================================
 /**
 * EX_Mont_TFTMul_OPT2_AS_GENE:
 * @resPtr: coefficient vector of result
 * @degA: degree of A
 * @APtr: coefficient vector of A 
 * @degB: degree of B
 * @BPtr: coefficient vector of B
 * @pPtr: prime number structure
 * 
 * TFT-based multiplication of A by B. Result is in resPtr.
 * 
 * Return value: the product of two polynomials.
 **/
void EX_Mont_TFTMul_OPT2_AS_GENE(sfixn * resPtr, sfixn degA, sfixn * APtr,
     sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr){

    register sfixn i;
    sfixn e, n, l_AB, BRsft=pPtr->Base_Rpow;
    sfixn * rootsPtr, * dftAPtr, * dftBPtr;

    l_AB=degA+degB+1; // compute l_AB for TFT(A,B).
    e=logceiling(l_AB);
    if(e > (pPtr->Npow)){
        Interrupted=1; PrimeError=1; 
        return;
    }
    n=1L<<e;
    // force this FFT use the n from parameter if n<n1.
    rootsPtr=(sfixn *)my_calloc(n, sizeof(sfixn));
    dftAPtr=(sfixn *)my_calloc(n, sizeof(sfixn));
    dftBPtr=(sfixn *)my_calloc(n, sizeof(sfixn));
    for(i=0; i<=degA; i++) dftAPtr[i]=APtr[i];
    for(i=0; i<=degB; i++) dftBPtr[i]=BPtr[i];

    EX_Mont_GetNthRoots_OPT2_AS_GENE(e, n, rootsPtr, pPtr);
    EX_Mont_TDFT_OPT2_AS_GENE_1( l_AB, rootsPtr, dftAPtr, pPtr);
    EX_Mont_TDFT_OPT2_AS_GENE_1( l_AB, rootsPtr, dftBPtr, pPtr);
    for(i=0; i<l_AB; i++) dftAPtr[i]=MontMulMod_OPT2_AS_GENE(dftAPtr[i], dftBPtr[i]<<BRsft, pPtr);
    EX_Mont_INVTDFT_OPT2_AS_GENE_R_1(l_AB, rootsPtr, dftAPtr, pPtr);

    for(i=0; i<l_AB; i++) resPtr[i]=dftAPtr[i];
    my_free(dftAPtr);
    my_free(dftBPtr);
    my_free(rootsPtr);
}

/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#include "GCD.h"


extern int Interrupted;

/**
 * zeroUni:
 * @deg: degree of input univeriate polynomial.
 * @cof: coefficient vector of input univeriate polynomial.
 * 
 *  Test if the input univariate polynomial is zero or not.
 *
 * Return value: if input polynomial is zero, returns 1, otherwise returns 0.
 **/

static 
int zeroUni(sfixn deg, sfixn * cof){
  register sfixn i;
  for(i=0; i<=deg; i++){
    if(cof[i]) return 0; 
  }
  return 1;
}


/**
 * consUni:
 * @deg: degree of input univeriate polynomial.
 * @cof: coefficient vector of input univeriate polynomial.
 * 
 *  Test if the input univariate polynomial is a constant or not.
 *
 * Return value: if input polynomial is a constant, returns 1, otherwise returns 0.
 **/
static 
int consUni(sfixn deg, sfixn * cof){
  register sfixn i;
  for(i=1; i<=deg; i++){
    if(cof[i]) return 0; 
  }
  return 1;
}



/**
 * normalize_1:
 * @deg: degree of input univeriate polynomial.
 * @cof: (output) coefficient vector of input univeriate polynomial.
 * @pPtr: info for prime number. 
 * 
 * make the input univariate polynomial monic.
 * Namely, in-placely, multiply the input by the inverse of its leading coefficent. 
 * Return value: void
 **/
void
normalize_1(sfixn deg, sfixn * cof, MONTP_OPT2_AS_GENE * pPtr){
    register sfixn i;
    sfixn p=pPtr->P, R=(1L<<pPtr->Rpow)%p,SFT=pPtr->Base_Rpow;
    sfixn inv;
    inv=inverseMod(cof[deg], p);
    R=MulMod(inv,R,p)<<SFT;
    for(i=0;i<=deg;i++) cof[i]=MontMulMod_OPT2_AS_GENE(cof[i],R,pPtr);
}



/**
 * normalizer:
 * @c: a scalar.
 * @deg: degree of input univeriate polynomial.
 * @cof: (output) coefficient vector of input univeriate polynomial.
 * @pPtr: info for prime number. 
 * 
 * In-placely, multiply the input univariate polynomial by the inverse of c. 
 * Return value: inverse of c.
 **/
static 
sfixn normalizer(sfixn c, sfixn deg, sfixn * cof, MONTP_OPT2_AS_GENE * pPtr){
    register sfixn i;
    sfixn p=pPtr->P, R=(1L<<pPtr->Rpow)%p,SFT=pPtr->Base_Rpow;
    sfixn inv;
    if(!c) return 0;
    inv=inverseMod(c,p);
    R=MulMod(inv,R,p)<<SFT;
    for(i=0;i<=deg;i++) cof[i]=MontMulMod_OPT2_AS_GENE(cof[i],R,pPtr);
    return inv;
}


// Input: APtr, BPtr.
// create a QPtr keeps the intermediate quotient.
// Output: APtr keeps gcd. and the return value is its degree.
// door=0, means using plainDiv
// door=1, means using FDiv
sfixn
gcd_Uni_1(sfixn * change,sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr,MONTP_OPT2_AS_GENE * pPtr, sfixn door){
  sfixn degQ=degA-degB;
  sfixn degR;
  sfixn * QPtr=(sfixn *)my_malloc((degA+1)*sizeof(sfixn));
  sfixn * tmpPtr;
  sfixn tmp;
  * change=0;
  // swtich=0 means APtr->[...A...], BPtr->[...B...]
  // swtich=1 means BPtr->[...A...], APtr->[...A...]
  normalizer(APtr[degA], degA, APtr, pPtr);
  normalizer(BPtr[degB], degB, BPtr, pPtr);
  while(! zeroUni(degB, BPtr)){
    degQ=degA-degB;
    cleanVec(degA, QPtr);
    normalizer(BPtr[degB], degB, BPtr, pPtr);
    if(door) fastDiv(APtr, degQ, QPtr, degA, APtr, degB, BPtr, pPtr);
    else     plainDivMonic_1(degQ, QPtr, degA, APtr, degB, BPtr, pPtr);
    degR=degB-1;
    degR=shrinkDegUni(degR, APtr);
    degA=degR;
   
    { tmpPtr=APtr;
      APtr=BPtr;
      BPtr=tmpPtr;
      tmp=degA;
      degA=degB;
      degB=tmp;
      if(* change) * change=0; else * change=1;
    }

  }

  my_free(QPtr);

  return degA;
}




// door=0, means using plainDiv
// door=1, means using FDiv
sfixn
ExGcd_Uni_1(sfixn * change,sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr,MONTP_OPT2_AS_GENE * pPtr, sfixn * degS, sfixn * degT, sfixn * S1Ptr, sfixn * S2Ptr,sfixn * T1Ptr, sfixn * T2Ptr, sfixn door){
  register sfixn j;
  sfixn degQ;
  sfixn degR;
  sfixn * QPtr;
  sfixn * tmpPtr, * resPtr;
  sfixn degRes; 
  sfixn tmp;
  sfixn degS1, degS2, degT1, degT2, inv1, inv2;
    * change=0;
  // swtich=0 means APtr->[...A...], BPtr->[...B...]
  // swtich=1 means BPtr->[...A...], APtr->[...A...]


  
  if(degA<degB){ 
    return  ExGcd_Uni_1(change, degB, BPtr, degA, APtr, pPtr, 
                 degT, degS, T1Ptr, T2Ptr, S1Ptr, S2Ptr, door);
  }

  degQ=degA-degB;
  QPtr=(sfixn *)my_malloc((degA+1)*sizeof(sfixn));

  inv1=normalizer(APtr[degA], degA, APtr, pPtr);
  inv2=normalizer(BPtr[degB], degB, BPtr, pPtr);
  S1Ptr[0]=inv1;T2Ptr[0]=inv2;
  degS1=degS2=degT1=degT2=0;


  while(! zeroUni(degB, BPtr)){
    degQ=degA-degB;
    cleanVec(degA, QPtr);
    normalizer(BPtr[degB], degB, BPtr, pPtr);
    if(door) fastDiv(APtr, degQ, QPtr, degA, APtr, degB, BPtr, pPtr);
    else     plainDivMonic_1(degQ, QPtr, degA, APtr, degB, BPtr, pPtr);
    degR=degB-1;
    degR=shrinkDegUni(degR, APtr);
    degA=degR;
    // to normalizer APtr[degR]

// normal S1Ptr <- S1Ptr-FFT(S2Ptr, Qtr);
    degRes=degS2+degQ+1;
    resPtr=(sfixn *)my_calloc(degRes+1, sizeof(sfixn));
    if((degS2>64) && (degQ>64)){
         EX_Mont_FFTMul(degRes, resPtr,degS2, S2Ptr, degQ, QPtr, pPtr );
     }
    else{
        EX_Mont_PlainMul_OPT2_AS_GENE(degRes, resPtr, degS2, S2Ptr, degQ, QPtr, pPtr  );}
    //if(degS1<degRes) printf("out of range!!!!!!!!!!!!!!!\n");
    for(j=0; j<=degRes; j++) S1Ptr[j]=SubMod(S1Ptr[j],resPtr[j],pPtr->P);

    //printf("dead-0 ------------------------>\n");


    my_free(resPtr);
    if(degRes>degS1) degS1=degRes;
    normalizer(APtr[degR], degS1, S1Ptr, pPtr);

    degRes=degT2+degQ+1;
    resPtr=(sfixn *)my_calloc(degRes+1, sizeof(sfixn));
    if((degT2>2) && (degQ>2)){
      
         
         EX_Mont_FFTMul(degRes, resPtr, degT2, T2Ptr, degQ, QPtr, pPtr );
     }
    else{ 
          EX_Mont_PlainMul_OPT2_AS_GENE(degRes, resPtr, degT2, T2Ptr, degQ, QPtr, pPtr);
     }

    // printf("dead-0.2 ------------------------>\n");

    for(j=0; j<=degRes; j++) T1Ptr[j]=SubMod(T1Ptr[j],resPtr[j],pPtr->P);

    //printf("dead-1 ------------------------>\n");


    my_free(resPtr);


    if(degRes>degT1) degT1=degRes;
    normalizer(APtr[degR], degT1, T1Ptr, pPtr);


    //printf("dead-2 ------------------------>\n");


    { tmpPtr=APtr;
      APtr=BPtr;
      BPtr=tmpPtr;
      tmp=degA;
      degA=degB;
      degB=tmp;

      tmpPtr=S1Ptr;
      S1Ptr=S2Ptr;
      S2Ptr=tmpPtr;
      tmp=degS1;
      degS1=degS2;
      degS2=tmp;

      tmpPtr=T1Ptr;
      T1Ptr=T2Ptr;
      T2Ptr=tmpPtr;
      tmp=degT1;
      degT1=degT2;
      degT2=tmp;

      if(*change) *change=0; else *change=1;
    }

  }


  my_free(QPtr);


  if(*change){
         *degS=degS2;
          *degT=degT2;
         *degS=shrinkDegUni(*degS, S2Ptr);
         *degT=shrinkDegUni(*degT, T2Ptr);}
  else{
         *degS=degS1;
         *degT=degT1;
         *degS=shrinkDegUni(*degS, S1Ptr);
         *degT=shrinkDegUni(*degT, T1Ptr); }


  return degA;
}





/**
 * ExGcd_Uni:
 * @uPtr: Coefficient vector for Bezout coefficent 1.
 * @dC: Pointer for the degree of Bezout coefficent 1.
 * @vPtr: Coefficient vector for Bezout coefficent 2.
 * @dD: Pointer for the degree of Bezout coefficent 2.
 * @gcdPtr: Coefficient vector for GCD.
 * @dG: Pointer for the degree of GCD.
 * @fPtr: Coefficient vector for the input polynomial 1.
 * @dA: Degree of input polynomial 1.
 * @gPtr: Coefficient vector for the input polynomail 2.
 * @dB: Degree of input polynomial 2.
 * @pPtr: Info for prime number.
 * 
 *  Extended Euclidean Algorithm.
 *      u * f  +  v * g  = gcd.
 * Return value: 
 **/
void
ExGcd_Uni(sfixn *uPtr, sfixn *dC, sfixn * vPtr, sfixn *dD, sfixn * gcdPtr, sfixn *dG, 
	  sfixn *fPtr, sfixn dA, sfixn * gPtr, sfixn dB, MONTP_OPT2_AS_GENE * pPtr){
       
    
    sfixn change;
    sfixn * f2Ptr, * g2Ptr, * u1Ptr, * v1Ptr, * u2Ptr, * v2Ptr;


    

       f2Ptr=(sfixn *) my_calloc(dA+1, sizeof(sfixn));
       copyVec_0_to_d(dA, f2Ptr, fPtr);
       g2Ptr=(sfixn *) my_calloc(dB+1, sizeof(sfixn));
       copyVec_0_to_d(dB, g2Ptr, gPtr);
       *dC=2*(dA+dB); *dD=2*(dA+dB);
       u1Ptr=(sfixn *) my_calloc(*dC+1, sizeof(sfixn));
       v1Ptr=(sfixn *) my_calloc(*dD+1, sizeof(sfixn));
       u2Ptr=(sfixn *) my_calloc(*dC+1, sizeof(sfixn));
       v2Ptr=(sfixn *) my_calloc(*dD+1, sizeof(sfixn));

       //printf("before inner gcd\n");
       //fflush(stdout);


       *dG=ExGcd_Uni_1(&change, dA, f2Ptr, dB, g2Ptr, pPtr, dC, dD, 
      	              u1Ptr, u2Ptr, v1Ptr, v2Ptr, 0);

       //printf("after inner gcd\n");
       //fflush(stdout);

       if(change){
         
         copyVec_0_to_d(*dG, gcdPtr, g2Ptr);
         copyVec_0_to_d(*dC, uPtr, u2Ptr);
         copyVec_0_to_d(*dD, vPtr, v2Ptr);
       }
       else{  
         
         copyVec_0_to_d(*dG, gcdPtr, f2Ptr);
         copyVec_0_to_d(*dC, uPtr, u1Ptr);
         copyVec_0_to_d(*dD, vPtr, v1Ptr);
       }

       my_free(g2Ptr);
       my_free(f2Ptr);
       my_free(u1Ptr);
       my_free(v1Ptr);

       //printf("--1--\n");
       //fflush(stdout);

       my_free(v2Ptr);

       //printf("--2--\n");
       //fflush(stdout);


       my_free(u2Ptr);

       ////printf("--3--\n");
       //fflush(stdout);


}





/**
 * Gcd_Uni:
 * @gcdPtr: Coefficient vector for GCD.
 * @dG: Pointer for the degree of GCD.
 * @fPtr: Coefficient vector for the input polynomial 1.
 * @dA: Degree of input polynomial 1.
 * @gPtr: Coefficient vector for the input polynomail 2.
 * @dB: Degree of input polynomial 2.
 * @pPtr: Info for prime number.
 * 
 *  Extended Euclidean Algorithm.
 *      gcd = GCD (f, g)
 * Return value: 
 **/
void
Gcd_Uni(sfixn * gcdPtr, sfixn *dG, 
	  sfixn * fPtr, sfixn dA, sfixn * gPtr, sfixn dB, MONTP_OPT2_AS_GENE * pPtr){
       
    
    sfixn change;
    sfixn * f2Ptr, * g2Ptr;
    

       f2Ptr=(sfixn *) my_calloc(dA+1, sizeof(sfixn));
       copyVec_0_to_d(dA, f2Ptr, fPtr);
       g2Ptr=(sfixn *) my_calloc(dB+1, sizeof(sfixn));
       copyVec_0_to_d(dB, g2Ptr, gPtr);
       
       *dG=gcd_Uni_1(&change, dA, f2Ptr, dB, g2Ptr, pPtr, 0);


       if(change){
         
         copyVec_0_to_d(*dG, gcdPtr, g2Ptr);
       }
       else{  
         
         copyVec_0_to_d(*dG, gcdPtr, f2Ptr);
       }

       my_free(g2Ptr);
       my_free(f2Ptr);

}


// output (*dGAddr, gcdPtr) is the gcd with exact degree *dGAddr.



/**
 * EX_GCD_UNI:
 * @dGAddr: Pointer for the degree of GCD.
 * @fPtr: Coefficient vector for the input polynomial 1.
 * @dA: Degree of input polynomial 1.
 * @gPtr: Coefficient vector for the input polynomail 2.
 * @dB: Degree of input polynomial 2.
 * @pPtr: Info for prime number.
 * 
 *  Extended Euclidean Algorithm.
 *      gcd = GCD (f, g)
 *
 * Return value: The coefficient vector of gcd.
 **/
sfixn *
EX_GCD_UNI(sfixn *dGAddr, 
	  sfixn * fPtr, sfixn dA, sfixn * gPtr, sfixn dB, MONTP_OPT2_AS_GENE * pPtr){
  
  sfixn *gcdPtr, *tmpGcdPtr;

  if(dA<dB) return EX_GCD_UNI(dGAddr, gPtr, dB,  fPtr, dA, pPtr);

  (*dGAddr)=dA;
  tmpGcdPtr=(sfixn *)my_calloc((*dGAddr)+1, sizeof(sfixn));
  Gcd_Uni(tmpGcdPtr, dGAddr, fPtr, dA, gPtr, dB, pPtr);
  gcdPtr=(sfixn *)my_calloc((*dGAddr)+1, sizeof(sfixn));
  copyVec_0_to_d((*dGAddr), gcdPtr, tmpGcdPtr);
  my_free(tmpGcdPtr);  
  return gcdPtr;
}




// (n, m)=(degA, Aptr)
// (x, f)=(degB, Bptr)
// d the rational reconstruction bound.




sfixn
ExGcd_Uni_RFR_1(sfixn d, sfixn * change,sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr,MONTP_OPT2_AS_GENE * pPtr, sfixn * degS, sfixn * degT, sfixn * S1Ptr, sfixn * S2Ptr,sfixn * T1Ptr, sfixn * T2Ptr, sfixn door){
  register sfixn j;
  sfixn degQ;
  sfixn degR;
  sfixn * QPtr;
  sfixn * tmpPtr, * resPtr;
  sfixn degRes; 
  sfixn tmp;

  sfixn degS1, degS2, degT1, degT2, inv1, inv2;
    * change=0;
  // swtich=0 means APtr->[...A...], BPtr->[...B...]
  // swtich=1 means BPtr->[...A...], APtr->[...A...]


  
  if(degA<degB){
    printf("Error: deg(m) Must bigger than deg(f). \n"); 
    fflush(stdout);
    Interrupted=1;
    return 0;
  }

  degQ=degA-degB;
  QPtr=(sfixn *)my_malloc((degA+1)*sizeof(sfixn));

  inv1=normalizer(APtr[degA], degA, APtr, pPtr);
  inv2=normalizer(BPtr[degB], degB, BPtr, pPtr);
  S1Ptr[0]=inv1;T2Ptr[0]=inv2;
  degS1=degS2=degT1=degT2=0;


  while(degA>=d){
    degQ=degA-degB;
    cleanVec(degA, QPtr);
    normalizer(BPtr[degB], degB, BPtr, pPtr);
    if(door) fastDiv(APtr, degQ, QPtr, degA, APtr, degB, BPtr, pPtr);
    else     plainDivMonic_1(degQ, QPtr, degA, APtr, degB, BPtr, pPtr);
    degR=degB-1;
    degR=shrinkDegUni(degR, APtr);
    degA=degR;
    // to normalizer APtr[degR]

// normal S1Ptr <- S1Ptr-FFT(S2Ptr, Qtr);
    degRes=degS2+degQ+1;
    resPtr=(sfixn *)my_calloc(degRes+1, sizeof(sfixn));
    if((degS2>64) && (degQ>64)){
         EX_Mont_FFTMul(degRes, resPtr,degS2, S2Ptr, degQ, QPtr, pPtr );
     }
    else{
        EX_Mont_PlainMul_OPT2_AS_GENE(degRes, resPtr, degS2, S2Ptr, degQ, QPtr, pPtr  );}
    //if(degS1<degRes) printf("out of range!!!!!!!!!!!!!!!\n");
    for(j=0; j<=degRes; j++) S1Ptr[j]=SubMod(S1Ptr[j],resPtr[j],pPtr->P);

    //printf("dead-0 ------------------------>\n");

    my_free(resPtr);
    if(degRes>degS1) degS1=degRes;
    normalizer(APtr[degR], degS1, S1Ptr, pPtr);
// normal T1Ptr <- T1Ptr-FFT(T2Ptr, Qtr);

    // printf("dead-0.1 ------------------------>\n");

    degRes=degT2+degQ+1;
    resPtr=(sfixn *)my_calloc(degRes+1, sizeof(sfixn));
    if((degT2>2) && (degQ>2)){
      
         
         EX_Mont_FFTMul(degRes, resPtr, degT2, T2Ptr, degQ, QPtr, pPtr );
     }
    else{ 
          EX_Mont_PlainMul_OPT2_AS_GENE(degRes, resPtr, degT2, T2Ptr, degQ, QPtr, pPtr);
     }

    // printf("dead-0.2 ------------------------>\n");

    for(j=0; j<=degRes; j++) T1Ptr[j]=SubMod(T1Ptr[j],resPtr[j],pPtr->P);

    //printf("dead-1 ------------------------>\n");

    my_free(resPtr);
    if(degRes>degT1) degT1=degRes;
    normalizer(APtr[degR], degT1, T1Ptr, pPtr);


    //printf("dead-2 ------------------------>\n");


    { tmpPtr=APtr;
      APtr=BPtr;
      BPtr=tmpPtr;
      tmp=degA;
      degA=degB;
      degB=tmp;

      tmpPtr=S1Ptr;
      S1Ptr=S2Ptr;
      S2Ptr=tmpPtr;
      tmp=degS1;
      degS1=degS2;
      degS2=tmp;

      tmpPtr=T1Ptr;
      T1Ptr=T2Ptr;
      T2Ptr=tmpPtr;
      tmp=degT1;
      degT1=degT2;
      degT2=tmp;

      if(* change) * change=0; else * change=1;
    }

  }


  my_free(QPtr);

  //printf("dead-3 ------------------------>\n");

  // *degS=degS1;
  // *degT=degT1;

  if(*change){

           *degS=degS2;
           *degT=degT2;
         *degS=shrinkDegUni(*degS, S2Ptr);
         *degT=shrinkDegUni(*degT, T2Ptr);}
  else{
          *degS=degS1;
          *degT=degT1;

         *degS=shrinkDegUni(*degS, S1Ptr);
         *degT=shrinkDegUni(*degT, T1Ptr);}

  return degA;
}





/* RationalFunctionInterpolate(N, [q_0, ..., q_{N-1}], [x_0, ..., x_{N-1}], d) */
/*   # the x_i are pairwise different; they are the evaluation points */
/*     -> these could be given by subproduct tree */
/*   # q_i is the value of a rational function q at x_i */
/*   # d is a boud for the degree of the denominator */

/* (1) let f := Interpolate([q_0, ..., q_N], [x_0, ..., x_N]) */
/*      f has degree N-1 */
/*     let $m$ be the product of the $X - x_i$; m has degree N */
/*     let n := N -1 */
/*     let d be in the range 1..n */

/* (2)  We want RFR(m, n, f, d) */

/* ------------------------------ */

// -1 failed
// 0 fine.
// the input 
// the output (*dG, gcdPtr) / (*dD, vPtr)

/**
 * ExGcd_Uni_RFR:
 * @d: d the rational reconstruction bound.
 * @vPtr: (output) demoniator's coefficient vector.
 * @dD: pointer for degree of (output) demoniator.
 * @gcdPtr: (output) numerator's coefficient vector.
 * @dG: pointer for degree of (output) numerator.
 * @fPtr: Coefficient vector for univariate polynomial f.
 * @dA: Degree for f.
 * @gPtr: Coefficient vector for univariate polynomial g.
 * @dB Degree for g:
 * @pPtr: info for the prime.
 *
 * Give univariate polynomial f and g, 
 * to reconstruct gcd / v such that:  
 *  gcd / v =  f mod g.
 *
 * Return value: -1 means reconstruction failed. 0 means succeeded.
 **/
int
ExGcd_Uni_RFR(sfixn d, sfixn * vPtr, sfixn *dD, sfixn * gcdPtr, sfixn *dG,  sfixn * fPtr, sfixn dA, sfixn * gPtr, sfixn dB, MONTP_OPT2_AS_GENE * pPtr){
       
    
    sfixn change;
    sfixn * f2Ptr, * g2Ptr, * u1Ptr, * v1Ptr, * u2Ptr, * v2Ptr;
    int isCons;
    sfixn degC=0, *dC=&degC, *uPtr; 
	sfixn *Ti, *Ri;
    sfixn dTi,  dRi;     
    sfixn dgcd, *dgcdAddr, *gcd;
    sfixn invlcTi;

    if (dB==0) {gcdPtr[0]=gPtr[0]; vPtr[0]=1; return 0;}

    *dC=*dD, 
       uPtr=(sfixn *)my_calloc((*dC)+1, sizeof(sfixn));

       f2Ptr=(sfixn *) my_calloc(dA+1, sizeof(sfixn));
       copyVec_0_to_d(dA, f2Ptr, fPtr);
       g2Ptr=(sfixn *) my_calloc(dB+1, sizeof(sfixn));
       copyVec_0_to_d(dB, g2Ptr, gPtr);
       * dC=(dA+dB); *dD=(dA+dB);
       u1Ptr=(sfixn *) my_calloc(*dC+1, sizeof(sfixn));
       v1Ptr=(sfixn *) my_calloc(*dD+1, sizeof(sfixn));
       u2Ptr=(sfixn *) my_calloc(*dC+1, sizeof(sfixn));
       v2Ptr=(sfixn *) my_calloc(*dD+1, sizeof(sfixn));



 
          
       *dG=ExGcd_Uni_RFR_1(d, &change, dA, f2Ptr, dB, g2Ptr, pPtr, dC, dD, 
      	              u1Ptr, u2Ptr, v1Ptr, v2Ptr, 0);

       if(change){

         copyVec_0_to_d(*dG, gcdPtr, g2Ptr);
         copyVec_0_to_d(*dC, uPtr, u2Ptr);
         copyVec_0_to_d(*dD, vPtr, v2Ptr);




       }
       else{  
 
         copyVec_0_to_d(*dG, gcdPtr, f2Ptr);
         copyVec_0_to_d(*dC, uPtr, u1Ptr);
         copyVec_0_to_d(*dD, vPtr, v1Ptr);
       }


 


       my_free(g2Ptr);
       my_free(u1Ptr);
       my_free(v1Ptr);
       my_free(u2Ptr);
       my_free(v2Ptr);
       my_free(f2Ptr);
     


     


    *dG=shrinkDegUni(*dG, gcdPtr);
    *dC=shrinkDegUni(*dC, uPtr);
    *dD=shrinkDegUni(*dD, vPtr);


       Ti=vPtr; Ri=gcdPtr;
       dTi=*dD;  dRi=*dG;     
       dgcd=0, dgcdAddr=&dgcd;
       


       gcd=EX_GCD_UNI(dgcdAddr, 
		   Ri, dRi, Ti, dTi, pPtr);
       
       isCons=consUni(dgcd, gcd);

       if(isCons==0) { //printf("rational reconstruction Failed!\n"); 
                       //fflush(stdout);     
                      my_free(uPtr); 
                      return -1;}

       invlcTi=inverseMod(Ti[dTi], pPtr->P);

       coMulVec_1(invlcTi, dRi, Ri, pPtr);

       coMulVec_1(invlcTi, dTi, Ti, pPtr);

       my_free(uPtr);

       return 0;
 
}




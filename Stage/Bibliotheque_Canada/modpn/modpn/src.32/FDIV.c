/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */

#include "FDIV.h"

extern int Interrupted;
//===================================================================
// Newton Inverse.
//===================================================================

// type: exported.
// Input: degG=n-1, degF<l where x^l is the modulus.
//        degF< l implies F is reduced r.w.t x^l.
//        r=ceiling(log_2(l)), n=2^r.
// Output: G with degG=n-1. G is the inverse of F modulo x^r.
//         =>  G is also the inverse of F modulo x^l.
// note: good for all Fourier Primes.
 /**
 * modularInvPM: Computes the inverse of F modulo x^r 
 *               This assumes F has a non-zero trailing coefficient.
 *               This works for all Fourier Primes.
 * @degG: degree of G (output) 
 * @GPtr: coefficient vector of G (output) 
 * @degF: degree of F
 * @FPtr: coefficient vector of F
 * @r: 
 * @n: equals 2^r
 * @pPtr: prime number structure
 * 
 * Using the Middle product trick.
 * So the running time is expected to be in 2 M(n) + o(M(n)) machine 
 * word operations.
 * Return value: G, the inverse of F modulo x^r 
 **/   
sfixn *
modularInvPM(sfixn degG, // degG=n-1;
       sfixn * GPtr, sfixn degF, sfixn * FPtr, 
       sfixn r, sfixn n,
       MONTP_OPT2_AS_GENE * pPtr)
 {
  int i,j;
  sfixn nn, halfnn;
  sfixn * rootsPtr=(sfixn *)my_calloc(n, sizeof(sfixn)), 
         * tmpGPtr=(sfixn *)my_calloc(n, sizeof(sfixn)), 
         * tmpFPtr=(sfixn *)my_calloc(n, sizeof(sfixn));
  

  GPtr[0]=1;

  for(i=1;i<=r;i++){
       nn=1<<i;
       halfnn=nn>>1;

       EX_Mont_GetNthRoots_OPT2_AS_GENE(i, nn, rootsPtr, pPtr);

       EX_Mont_DFT_OPT2_AS_GENE( nn, i, rootsPtr, tmpGPtr, nn-1, GPtr, pPtr);

       if(degF>=(nn-1)) EX_Mont_DFT_OPT2_AS_GENE( nn, i, rootsPtr, tmpFPtr, nn-1, FPtr, pPtr);
       else    EX_Mont_DFT_OPT2_AS_GENE( nn, i, rootsPtr, tmpFPtr, degF, FPtr, pPtr);
       EX_Mont_PairwiseMul_OPT2_AS_R(nn, tmpFPtr, tmpGPtr, pPtr);


       EX_Mont_INVDFT_OPT2_AS_GENE_R_1 (nn, i, rootsPtr, tmpFPtr, pPtr);

       for(j=0;j<halfnn; j++) tmpFPtr[j]=tmpFPtr[j+halfnn];

       for(j=halfnn; j<nn; j++) tmpFPtr[j]=0;


       EX_Mont_DFT_OPT2_AS_GENE_1 ( nn, i, rootsPtr, tmpFPtr, pPtr );

       EX_Mont_PairwiseMul_OPT2_AS_R(nn, tmpFPtr, tmpGPtr, pPtr);


       EX_Mont_INVDFT_OPT2_AS_GENE_R_1 (nn, i, rootsPtr, tmpFPtr, pPtr);

       for(j=halfnn; j<nn; j++) GPtr[j]=SubMod(GPtr[j],tmpFPtr[j-halfnn],pPtr->P);

     };

  my_free (rootsPtr);
  my_free (tmpGPtr);
  my_free (tmpFPtr);
  return GPtr;
}






//===================================================================
// Fast Division.
//===================================================================


// type: Only for benchmark use. 
// Input: A, B. Suppose deg(A)>=deg(B)
//        B is monic.
// Ouput: (Q , R), A=QB+R. 
// note: good for all Fourier Primes.
 /**
 * fastDiv_bench: Computes the division of A by B, returning the
 *                quotient Q and the coefficient vector of the remainder R.
 * @RPtr: (output) Coefficient vector for the reaminder (size degB)
 * @degQ: (output) degree of the quotient
 * @QPtr: (output) Coefficient vector  of the quotient
 * @degA: degree of A
 * @APtr: Coefficient vector  of A
 * @degB: degree of B
 * @BPtr: Coefficient vector  of B 
 * @pPtr: prime number structure
 * 
 *               This works for all Fourier Primes.
 * 
 * Return value: the quotient and the remainder of A by B
 **/   
double
fastDiv_bench(sfixn * RPtr, sfixn degQ, sfixn * QPtr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr ){
   register sfixn j;
   sfixn * FPtr, * GPtr;
   sfixn degF, degG, power1,power2,power3, n1,n2,n3,l1,l2,l3, tmp, sz1, sz2, sz3;
   sfixn dg, da;


   degF=degA-degB;

   if(degF<0) { //printf("Attension: degA<degB!\n");
               for(j=0;j<=degA;j++) RPtr[j]=APtr[j];
               for(j=0;j<=degQ;j++) QPtr[j]=0;
               return 0;}

   n1=1; power1=0; l1=degF+1; tmp=l1;
   while(tmp){tmp>>=1; n1<<=1; power1++;}

   n2=1; power2=0; l2=(l1<<1)-1;tmp=l2;
   while(tmp){tmp>>=1; n2<<=1; power2++;}

   n3=1; power3=0;  l3=degB+degQ+1, tmp=l3;
   while(tmp){tmp>>=1; n3<<=1; power3++;}

   dg=da=degF;
   sz1=sz2=l2;
   if(sz1<n1) sz1=n1;
   if(sz2<(degA+1)) sz2=degA+1;
 
   degG=n1-1;
   sz3=degF+1;
   if(sz3<degB+1) sz3=degB+1;
   if(sz3<sz2) sz3=sz2;
   if(sz3<n2) sz3=n2;
   if(sz2<n3) sz3=n3;
   sz1=sz3;


   //  1. F=rev_m(B);
   FPtr=(sfixn * )my_calloc(sz3 ,sizeof(sfixn));
   FPtr=reverseUni(degB, FPtr, BPtr);

  
   //  2. get G such that GF=1 mod x^n;   my_free(rootsPtr);
   GPtr=(sfixn * )my_calloc(sz1, sizeof(sfixn));

   //===============================================================>
   GPtr=modularInvPM(n1-1, GPtr, degF, FPtr, power1, n1, pPtr);
   //===============================================================>

   //  3. rev_n(A)*G mod n; 
   FPtr=reverseUni(degA, FPtr, APtr);

   EX_Mont_FFTMul_OPT2_AS_GENE_1(n2, power2, da+dg, da, FPtr, dg, GPtr, pPtr);

   QPtr=reverseUni(degQ, QPtr, FPtr);

   cleanVec(n3-1, FPtr);
   EX_Mont_FFTMul_OPT2_AS_GENE(n3, power3, degQ+degB, FPtr, degQ, QPtr, degB, BPtr, pPtr);

   for(j=0; j<l3; j++) RPtr[j]=SubMod(APtr[j], FPtr[j],pPtr->P);

   my_free(FPtr);
   my_free(GPtr); 

   return 0; 

 } 





// type: exported 
// Input: A, B. Suppose deg(A)>=deg(B)
//        B is monic.
// Ouput: (Q , R), A=QB+R. 
// note: good for all Fourier Primes.
 /**
 * fastDiv: Computes the division of A by B, returning the
 *                quotient Q and the coefficient vector of the remainder R.
 * @RPtr: (output) Coefficient vector for the reaminder (size degB)
 * @degQ: (output) degree of the quotient
 * @QPtr: (output) Coefficient vector  of the quotient
 * @degA: degree of A
 * @APtr: Coefficient vector  of A
 * @degB: degree of B
 * @BPtr: Coefficient vector  of B 
 * @pPtr: prime number structure
 * 
 *               This works for all Fourier Primes.
 * 
 * Return value: the quotient and the remainder of A by B
 **/   
void 
fastDiv(sfixn * RPtr, sfixn degQ, sfixn * QPtr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr ){
   register sfixn j;
   sfixn * FPtr, * GPtr;
   sfixn degF, degG, power1,power2,power3, n1,n2,n3,l1,l2,l3, tmp, sz1, sz2, sz3;
   sfixn dg, da;

   degF=degA-degB;

   if(degF<0) { //printf("Attension: degA<degB!\n");
               for(j=0;j<=degA;j++) RPtr[j]=APtr[j];
               for(j=0;j<=degQ;j++) QPtr[j]=0;
               return;}

   n1=1; power1=0; l1=degF+1; tmp=l1;
   while(tmp){tmp>>=1; n1<<=1; power1++;}

   n2=1; power2=0; l2=(l1<<1)-1;tmp=l2;
   while(tmp){tmp>>=1; n2<<=1; power2++;}

   n3=1; power3=0;  l3=degB+degQ+1, tmp=l3;
   while(tmp){tmp>>=1; n3<<=1; power3++;}

  

   dg=da=degF;
   sz1=sz2=l2;
   if(sz1<n1) sz1=n1;
   if(sz2<(degA+1)) sz2=degA+1;
 
   degG=n1-1;
   sz3=degF+1;
   if(sz3<degB+1) sz3=degB+1;
   if(sz3<sz2) sz3=sz2;
   if(sz3<n2) sz3=n2;
   if(sz2<n3) sz3=n3;
   sz1=sz3;


   //  1. F=rev_m(B);
   FPtr=(sfixn * )my_calloc(sz3 ,sizeof(sfixn));
   FPtr=reverseUni(degB, FPtr, BPtr);

  
   //  2. get G such that GF=1 mod x^n;   my_free(rootsPtr);
   GPtr=(sfixn * )my_calloc(sz1, sizeof(sfixn));

 
   //===============================================================>
   GPtr=modularInvPM(n1-1, GPtr, degF, FPtr, power1, n1, pPtr);
   //===============================================================>

   //  3. rev_n(A)*G mod n; 
   FPtr=reverseUni(degA, FPtr, APtr);

   EX_Mont_FFTMul_OPT2_AS_GENE_1(n2, power2, da+dg, da, FPtr, dg, GPtr, pPtr);

   QPtr=reverseUni(degQ, QPtr, FPtr);

   cleanVec(n3-1, FPtr);
   EX_Mont_FFTMul_OPT2_AS_GENE(n3, power3, degQ+degB, FPtr, degQ, QPtr, degB, BPtr, pPtr);
 
   for(j=0; j<l3; j++) RPtr[j]=SubMod(APtr[j], FPtr[j],pPtr->P);

   my_free(FPtr);
   my_free(GPtr); 


 } 


 /**
 * fastQuo: Computes the quotient  in the Euclidean
 *          (fast) division of A by B
 * @degQ: (output) degree of the quotient
 * @QPtr: (output) Coefficient vector  of the quotient
 * @degA: degree of A
 * @APtr: Coefficient vector  of A
 * @degB: degree of B
 * @BPtr: Coefficient vector  of B 
 * @pPtr: prime number structure
 * 
 *               This works for all Fourier Primes.
 * 
 * Return value: the quotient  in the Euclidean division of A by B.
 **/   
void 
fastQuo(sfixn degQ, sfixn * QPtr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr ){
   register sfixn j;
   sfixn *FPtr, *GPtr, *RPtr;
   sfixn degF, degG, power1,power2,power3, n1,n2,n3,l1,l2,l3, tmp, sz1, sz2, sz3;
   sfixn dg, da;

   degF=degA-degB;

   if(degF<0) { //printf("Attension: degA<degB!\n");
               //for(j=0;j<=degA;j++) RPtr[j]=APtr[j];
               for(j=0;j<=degQ;j++) QPtr[j]=0;
               return;}


   RPtr=(sfixn *)my_calloc(degB, sizeof(sfixn));


   n1=1; power1=0; l1=degF+1; tmp=l1;
   while(tmp){tmp>>=1; n1<<=1; power1++;}

   n2=1; power2=0; l2=(l1<<1)-1;tmp=l2;
   while(tmp){tmp>>=1; n2<<=1; power2++;}

   n3=1; power3=0;  l3=degB+degQ+1, tmp=l3;
   while(tmp){tmp>>=1; n3<<=1; power3++;}

  


   // sz1, sz2 are the size of vector for GPtr, ArevPtr. They are used twice in this function!
   dg=da=degF;
   sz1=sz2=l2;
   if(sz1<n1) sz1=n1;
   if(sz2<(degA+1)) sz2=degA+1;
 
   degG=n1-1;
   sz3=degF+1;
   if(sz3<degB+1) sz3=degB+1;
   if(sz3<sz2) sz3=sz2;
   if(sz3<n2) sz3=n2;
   if(sz2<n3) sz3=n3;
   sz1=sz3;


   //  1. F=rev_m(B);
   FPtr=(sfixn * )my_calloc(sz3 ,sizeof(sfixn));
   FPtr=reverseUni(degB, FPtr, BPtr);
  
   //  2. get G such that GF=1 mod x^n;   my_free(rootsPtr);
   GPtr=(sfixn * )my_calloc(sz1, sizeof(sfixn));

   //===============================================================>
   GPtr=modularInvPM(n1-1, GPtr, degF, FPtr, power1, n1, pPtr);
   //===============================================================>

   //  3. rev_n(A)*G mod n; 
   FPtr=reverseUni(degA, FPtr, APtr);

   EX_Mont_FFTMul_OPT2_AS_GENE_1(n2, power2, da+dg, da, FPtr, dg, GPtr, pPtr);

   QPtr=reverseUni(degQ, QPtr, FPtr);

   my_free(FPtr);
   my_free(GPtr); 
   my_free(RPtr);

 } 









// type: exported 
// Input: A, B. Suppose deg(A)>=deg(B)
//        B is monic.
// Ouput: (Q , R), A=QB+R. 
// note: good for all Fourier Primes.
 /**
 * fastRem: Computes the remainder of A by B.
 * @degRAddr: (output) degree of the remainder
 * @RPtr: (output) Coefficient vector for the reaminder (size degB)
 * @degA: degree of A
 * @APtr: Coefficient vector  of A
 * @degB: degree of B
 * @BPtr: Coefficient vector  of B 
 * @pPtr: prime number structure
 * 
 *               This works for all Fourier Primes.
 * 
 * Return value:  remainder of A by B
 **/   
void 
fastRem(sfixn *degRAddr, sfixn * RPtr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr ){
   register sfixn j;
   sfixn * FPtr, * GPtr, *QPtr;
   sfixn degQ, degF, degG, power1,power2,power3, n1,n2,n3,l1,l2,l3, tmp, sz1, sz2, sz3;
   sfixn dg, da;

   degF=degA-degB;
   degQ=degF;


   if(degF<0) { //printf("Attension: degA<degB!\n");
               for(j=0;j<=degA;j++) RPtr[j]=APtr[j];
               //for(j=0;j<=degQ;j++) QPtr[j]=0;
               return;}

   QPtr=(sfixn *)my_calloc(degQ+1, sizeof(sfixn));


   n1=1; power1=0; l1=degF+1; tmp=l1;
   while(tmp){tmp>>=1; n1<<=1; power1++;}

   n2=1; power2=0; l2=(l1<<1)-1;tmp=l2;
   while(tmp){tmp>>=1; n2<<=1; power2++;}

   n3=1; power3=0;  l3=degB+degQ+1, tmp=l3;
   while(tmp){tmp>>=1; n3<<=1; power3++;}

  


   // sz1, sz2 are the size of vector for GPtr, ArevPtr. They are used twice in this function!
   dg=da=degF;
   sz1=sz2=l2;
   if(sz1<n1) sz1=n1;
   if(sz2<(degA+1)) sz2=degA+1;
 
   degG=n1-1;
   sz3=degF+1;
   if(sz3<degB+1) sz3=degB+1;
   if(sz3<sz2) sz3=sz2;
   if(sz3<n2) sz3=n2;
   if(sz2<n3) sz3=n3;
   sz1=sz3;


   //  1. F=rev_m(B);
   FPtr=(sfixn * )my_calloc(sz3 ,sizeof(sfixn));
   FPtr=reverseUni(degB, FPtr, BPtr);

  
   //  2. get G such that GF=1 mod x^n;   my_free(rootsPtr);
   GPtr=(sfixn * )my_calloc(sz1, sizeof(sfixn));
 
   //===============================================================>
   GPtr=modularInvPM(n1-1, GPtr, degF, FPtr, power1, n1, pPtr);
   //===============================================================>

   //  3. rev_n(A)*G mod n; 
   FPtr=reverseUni(degA, FPtr, APtr);


   EX_Mont_FFTMul_OPT2_AS_GENE_1(n2, power2, da+dg, da, FPtr, dg, GPtr, pPtr);


   //printf("FPtr is:");
   // printVec(da, FPtr);

   QPtr=reverseUni(degQ, QPtr, FPtr);


   cleanVec(n3-1, FPtr);
   EX_Mont_FFTMul_OPT2_AS_GENE(n3, power3, degQ+degB, FPtr, degQ, QPtr, degB, BPtr, pPtr);


  for(j=0; j<=(*degRAddr); j++) RPtr[j]=SubMod(APtr[j], FPtr[j],pPtr->P);


  while((! RPtr[*degRAddr]) && ((*degRAddr)>0)) (*degRAddr)--;

   my_free(QPtr);
   my_free(FPtr);
   my_free(GPtr); 


 } 


 /**
 * UniRem: Computes the remainder of A by B.
 * @degRAddr: (output) degree of the remainder
 * @RPtr: (output) Coefficient vector for the reaminder (size degB)
 * @degA: degree of A
 * @APtr: Coefficient vector  of A
 * @degB: degree of B
 * @BPtr: Coefficient vector  of B 
 * @pPtr: prime number structure
 *               Using classical division for low quotient degree
 *               and fast division otherwise
 *               This works for all Fourier Primes.
 * 
 * Return value: remainder of A by B
 **/   
void 
UniRem(sfixn *degRAddr, sfixn * RPtr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr ){
   if (degB==0){
       *degRAddr=0;
        return;}
 
  if(degA-degB>60)
    fastRem(degRAddr, RPtr, degA, APtr, degB, BPtr, pPtr);
  else
    plainRem(degRAddr, RPtr, degA, APtr, degB, BPtr, pPtr);

}




 /**
 * UniQuo: Computes the quotient  in the Euclidean
 *          (fast) division of A by B
 * @degQ: (output) degree of the quotient
 * @QPtr: (output) Coefficient vector  of the quotient
 * @degA: degree of A
 * @APtr: Coefficient vector  of A
 * @degB: degree of B
 * @BPtr: Coefficient vector  of B 
 * @pPtr: prime number structure
 * 
 *               This works for all Fourier Primes.
 * 
 * Return value: the quotient  in the Euclidean division of A by B.
 **/   
void 
UniQuo(sfixn degQ, sfixn * QPtr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr ){
  int i;
  sfixn inv;
 

  if (degB==0){
       assert(BPtr[0]!=0);
       inv=inverseMod(BPtr[0], pPtr->P);
       for(i=0; i<=degA; i++) QPtr[i]=MulMod(inv, APtr[i], pPtr->P);
       return;}
 
  if(0){
    fastQuo(degQ, QPtr, degA, APtr, degB, BPtr, pPtr);
  }
  else{

    plainQuo(degQ, QPtr, degA, APtr, degB, BPtr, pPtr);

  }
    
}

 /**
 * UniPseuQuo: Computes the pseudo-quotient  in the 
 *          pseudo-division of A by B
 * @degQ: (output) degree of the pseudo-quotient
 * @QPtr: (output) Coefficient vector  of the pseudo-quotient
 * @degA: degree of A
 * @APtr: Coefficient vector  of A
 * @degB: degree of B
 * @BPtr: Coefficient vector  of B 
 * @pPtr: prime number structure
 * 
 *               This works for all Fourier Primes.
 * 
 * Return value: pseudo-quotient  in the pseudo-division of A by B.
 **/   
void 
UniPseuQuo(sfixn degQ, sfixn * QPtr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr ){
  int i;
  sfixn inv;
 

  if (degB==0){
       assert(BPtr[0]!=0);
       inv=inverseMod(BPtr[0], pPtr->P);
       for(i=0; i<=degA; i++) QPtr[i]=MulMod(inv, APtr[i], pPtr->P);
       return;}
   if(0){
 
  }
  else{

    //printf("APtr=\n");
    //printPolyUni(degA, APtr);
    //printf("BPtr=\n");
   // printPolyUni(degB, BPtr);

    PlainPseudoQuotient(&degQ, QPtr, degA, APtr,degB, BPtr, pPtr);


    //printf("QPtr=\n");
    //printPolyUni(degQ, QPtr);

  }
    
}



 /**
 * EX_UniRem: Computes the remainder of A by B (univariate monic division)
 * @degRAddr: (output) degree of the remainder
 * @degA: degree of A
 * @APtr: Coefficient vector  of A
 * @degB: degree of B
 * @BPtr: Coefficient vector  of B 
 * @pPtr: prime number structure
 *               Using classical division for low quotient degree
 *               and fast division otherwise
 *               This works for all Fourier Primes.
 * 
 * Return value: the coefficient vector of the remainder of A by B
 **/ 
sfixn *
EX_UniRem(sfixn *degRAddr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr ){
  int i;
  sfixn * tmpRPtr, *RPtr;
  *degRAddr=degB-1;
  tmpRPtr=(sfixn *)my_calloc((*degRAddr)+1, sizeof(sfixn));
  UniRem(degRAddr, tmpRPtr, degA, APtr, degB, BPtr, pPtr);
  RPtr=(sfixn *)my_calloc((*degRAddr)+1, sizeof(sfixn));
  for(i=0; i<=(*degRAddr); i++) RPtr[i]=tmpRPtr[i];
  my_free(tmpRPtr);
  return RPtr;
}


 /**
 * EX_UniQuo: Computes the quotient  in the Euclidean
 *          (fast) univariate division of A by B
 * @degQAddr: (output) degree of the quotient
 * @degA: degree of A
 * @APtr: Coefficient vector  of A
 * @degB: degree of B
 * @BPtr: Coefficient vector  of B 
 * @pPtr: prime number structure
 * 
 *               This works for all Fourier Primes.
 * 
 * Return value: the coefficient vector of the quotient  
 *                in the Euclidean division of A by B.
 **/   
sfixn *
EX_UniQuo(sfixn *degQAddr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr ){
  int i;
  sfixn * tmpQPtr, *QPtr;
  *degQAddr=degA;
  tmpQPtr=(sfixn *)my_calloc((*degQAddr)+1, sizeof(sfixn));
  UniQuo(*degQAddr, tmpQPtr, degA, APtr, degB, BPtr, pPtr);
  *degQAddr=shrinkDegUni(*degQAddr, tmpQPtr);
  QPtr=(sfixn *)my_calloc((*degQAddr)+1, sizeof(sfixn));
  for(i=0; i<=(*degQAddr); i++) QPtr[i]=tmpQPtr[i];
  my_free(tmpQPtr);
  return QPtr;
}



// type: exported, in-place.
// Input: A, B. Suppose deg(A)>=deg(B)
//        B is monic.
// Ouput: (Q , R), A=QB+R. 
// note: good for all Fourier Primes.
//  A is over-written by the remainder
// fast-division
void 
fastDiv_1(sfixn degQ, sfixn * QPtr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, sfixn * BRevInvPtr, MONTP_OPT2_AS_GENE * pPtr ){
   register sfixn j;
   sfixn * FPtr, * GPtr;
   sfixn degF, degG, power1,power2,power3, n1,n2,n3,l1,l2,l3, tmp, sz1, sz2, sz3;
   sfixn dg, da;

   degF=degA-degB;

   if(degF<0) {//printf("Attension: degA<degB!\n");
               for(j=0;j<=degQ;j++) QPtr[j]=0;
               return;}

   n1=1; power1=0; l1=degF+1; tmp=l1;
   while(tmp){tmp>>=1; n1<<=1; power1++;}

   n2=1; power2=0; l2=(l1<<1)-1;tmp=l2;
   while(tmp){tmp>>=1; n2<<=1; power2++;}

   n3=1; power3=0;  l3=degB+degQ+1, tmp=l3;
   while(tmp){tmp>>=1; n3<<=1; power3++;}

  


   // sz1, sz2 are the size of vector for GPtr, ArevPtr. They are used twice in this function!
   dg=da=degF;
   sz1=sz2=l2;
   if(sz1<n1) sz1=n1;
   if(sz2<(degA+1)) sz2=degA+1;
 
   degG=n1-1;
   sz3=degF+1;
   if(sz3<degB+1) sz3=degB+1;
   if(sz3<sz2) sz3=sz2;
   if(sz3<n2) sz3=n2;
   if(sz2<n3) sz3=n3;
   sz1=sz3;

   //  1. F=rev_m(B);
   FPtr=(sfixn * )my_calloc(sz3 ,sizeof(sfixn));
   FPtr=reverseUni(degB, FPtr, BPtr);
  
   //  2. get G such that GF=1 mod x^n;   my_free(rootsPtr);
   GPtr=(sfixn * )my_calloc(sz1, sizeof(sfixn));
   for(j=0;j<=dg; j++) GPtr[j]=BRevInvPtr[j];

   //  3. rev_n(A)*G mod n; 
   FPtr=reverseUni(degA, FPtr, APtr);

   EX_Mont_FFTMul_OPT2_AS_GENE_1(n2, power2, da+dg, da, FPtr, dg, GPtr, pPtr);

   QPtr=reverseUni(degQ, QPtr, FPtr);
  
   cleanVec(n3-1, FPtr);
   EX_Mont_FFTMul_OPT2_AS_GENE(n3, power3, degQ+degB, FPtr, degQ, QPtr, degB, BPtr, pPtr);

   for(j=0; j<l3; j++) APtr[j]=SubMod(APtr[j], FPtr[j],pPtr->P);
   for(j=l3; j<=degA; j++) APtr[j]=0;

   my_free(FPtr);
   my_free(GPtr); 

 } 




//===================================================================
// fmecg(res,e,r,p2) finds X :       res - r * X**e * p2
//===================================================================

// deg(res)>=deg(p2)+e
// note: good for all Fourier Primes.  
void fmedg_1(sfixn degRes, sfixn * resPtr, sfixn e, sfixn r, sfixn degp2, sfixn * p2Ptr, MONTP_OPT2_AS_GENE * pPtr ){
    register sfixn i, p=pPtr->P, R=(1L<<pPtr->Rpow)%p;
    R=(MulMod(r, R, p))<<pPtr->Base_Rpow;
    for(i=0; i<=degp2; i++) resPtr[i+e]=SubMod(resPtr[i+e], MontMulMod_OPT2_AS_GENE(p2Ptr[i], R, pPtr), p);
}

//===================================================================
// Plain Division.
//===================================================================
// note: good for all Fourier Primes.
 /**
 * plainDivMonic_1: Computes the monic division of A by B, producing the
 *                quotient Q and the coefficient vector of the remainder R.
 * @RPtr: (output) Coefficient vector for the reaminder (size degB)
 * @degQ: (output) degree of the quotient
 * @QPtr: (output) Coefficient vector  of the quotient
 * @degA: degree of A
 * @APtr: Coefficient vector  of A
 * @degB: degree of B
 * @BPtr: Coefficient vector  of B 
 * @pPtr: prime number structure
 * 
 *               This works for all Fourier Primes.
 *               Use the classical / plain algorithm.
 *               In place: coeff vector of A is over-written 
 *               by that of the  remainder.
 *               The degree of the remainder is not returned.
 * 
 * Return value: the quotient and the remainder of A by (monic) B
 **/ 
void 
plainDivMonic_1(sfixn degQ, sfixn * QPtr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr ){
  register sfixn i,j;
  sfixn tmp;

  if((degA-degB)<0) {//printf("Attension: degA<degB!\n");
               for(j=0;j<=degQ;j++) QPtr[j]=0;
               return;}

  for(i=degA-degB; i>=0; i--){
     if (APtr[degB+i] != 0){
        QPtr[i]=APtr[degB+i];
        tmp=MontMulMod_OPT2_AS_GENE(QPtr[i], pPtr->R2BRsft, pPtr)<<(pPtr->Base_Rpow);
        for(j=0; j<=degB; j++) APtr[i+j]=SubMod(APtr[i+j], MontMulMod_OPT2_AS_GENE(BPtr[j],tmp,pPtr), pPtr->P);
      }
    else{QPtr[i]=0;}
 
  }   

}



 /**
 * plainRemMonic_1: Computes the remainder in the monic division of A by B.
 8                  A is overwritten.
 * @degA: degree of A
 * @APtr: Coefficient vector  of A
 * @degB: degree of B
 * @BPtr: Coefficient vector  of B 
 * @pPtr: prime number structure
 * 
 *               This works for all Fourier Primes.
 *               Use the classical / plain algorithm.
 *               In place: coeff vector of A is over-written 
 *               by that of the  remainder.
 *               The degree of the remainder is not returned.
 * 
 * Return value: void
 **/ 
void 
plainRemMonic_1(sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr ){
  register sfixn i,j;
  sfixn tmp, p, R, SFT;

  if((degA-degB)<0) {//printf("Attension: degA<degB!\n");
               return;}

  p=pPtr->P; R=(1L<<pPtr->Rpow)%p; SFT=pPtr->Base_Rpow;
  R=MulMod(R,R,p)<<SFT;
  for(i=degA-degB; i>=0; i--){
     if (APtr[degB+i] != 0){
        tmp=MontMulMod_OPT2_AS_GENE(APtr[degB+i], R, pPtr)<<SFT;
        for(j=0; j<=degB; j++) APtr[i+j]=SubMod(APtr[i+j], MontMulMod_OPT2_AS_GENE(BPtr[j],tmp,pPtr), pPtr->P);
      }
 
  } 
  
}


// note: good for all Fourier Primes.
 /**
 * plainDiv_1: Computes the plain division of A by B, returning the
 *                quotient Q and the coefficient vector of the remainder R.
 * @degQ:          degree of the quotient
 * @QPtr: (output) Coefficient vector  of the quotient
 * @degA: degree of A
 * @APtr: (overwritten) Coefficient vector  of A (input) / R (output)
 * @degB: degree of B
 * @BPtr: Coefficient vector  of B 
 * @pPtr: prime number structure
 * 
 *               This works for all Fourier Primes.
 *               The coeff vector of A is overwritten by 
 *               that of the remainder. 
 * 
 * Return value: void
 **/   
void
plainDiv_1(sfixn degQ, sfixn * QPtr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr ){
  register sfixn i,j;
  sfixn degR, u, tmp, p, R, U, SFT;

  if((degA-degB)<0) {//printf("Attension: degA<degB!\n");
               for(j=0;j<=degQ;j++) QPtr[j]=0;
               return;}


  degR=degA, u=inverseMod(BPtr[degB],pPtr->P);
  p=pPtr->P; R=(1L<<pPtr->Rpow)%p; SFT=pPtr->Base_Rpow;
  U=MulMod(u,R,p)<<SFT;
  R=MulMod(R, R, p)<<SFT;

  for(i=degA-degB; i>=0; i--){
    //  if (degR == m+i){
    
    if (APtr[degB+i] != 0){
      QPtr[i]=MontMulMod_OPT2_AS_GENE(APtr[degB+i],U,pPtr);
      tmp=MontMulMod_OPT2_AS_GENE(QPtr[i], R, pPtr)<<SFT;
      for(j=0; j<=degB; j++) APtr[i+j]=SubMod(APtr[i+j], MontMulMod_OPT2_AS_GENE(BPtr[j],tmp,pPtr), pPtr->P);

      //      degR--;
    }
    else{QPtr[i]=0;}
  } 

}


// note: good for all Fourier Primes.
 /**
 * plainDiv: Computes the plain division of A by B, returning the
 *                quotient Q and the coefficient vector of the remainder R.
 * @RPtr: (output) Coefficient vector  of the remainder
 * @degQ:          degree of the quotient
 * @QPtr: (output) Coefficient vector  of the quotient
 * @degA: degree of A
 * @APtr: Coefficient vector  of A 
 * @degB: degree of B
 * @BPtr: Coefficient vector  of B 
 * @pPtr: prime number structure
 * 
 *               This works for all Fourier Primes.
 * 
 * Return value: void
 **/   
void 
plainDiv(sfixn * RPtr,sfixn degQ, sfixn * QPtr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr ){
  sfixn * tmpA=(sfixn *)my_calloc(degA+1, sizeof(sfixn));
  register sfixn i, degR;
  if(degA<degB) return;
  for(i=0; i<=degA; i++) tmpA[i]=APtr[i];
  
  plainDiv_1(degQ, QPtr, degA, tmpA, degB, BPtr, pPtr);

  degR=degB-1;
  while((! tmpA[degR]) && (degR>0)) degR--;

  for(i=0; i<=degR; i++) RPtr[i]=tmpA[i];

  my_free(tmpA);
 
}


void 
plainRem(sfixn *degRAddr, sfixn * RPtr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr ){
  sfixn * tmpA, *tmpQ, degQ;
  register sfixn i,j;
  if(degA<degB) {for(j=0;j<=degA;j++) RPtr[j]=APtr[j]; return;}
  tmpA=(sfixn *)my_calloc(degA+1, sizeof(sfixn));
  for(i=0; i<=degA; i++) tmpA[i]=APtr[i];
  degQ=degA-degB;
  tmpQ=(sfixn *)my_calloc(degQ+1, sizeof(sfixn));

  plainDiv_1(degQ, tmpQ, degA, tmpA, degB, BPtr, pPtr);

  while((! tmpA[* degRAddr]) && ((*degRAddr)>0)) (*degRAddr)--;

  for(i=0; i<=(*degRAddr); i++) RPtr[i]=tmpA[i];
  my_free(tmpQ);
  my_free(tmpA);
 
}



// note: good for all Fourier Primes.
void 
plainQuo(sfixn degQ, sfixn * QPtr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr ){
  sfixn * tmpA=(sfixn *)my_calloc(degA+1, sizeof(sfixn));
  register sfixn i;
  if(degA<degB) return;
  for(i=0; i<=degA; i++) tmpA[i]=APtr[i];
  
  plainDiv_1(degQ, QPtr, degA, tmpA, degB, BPtr, pPtr);

  my_free(tmpA);
 
}




void
plainDivNew(sfixn * RPtr,sfixn degQ, sfixn * QPtr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr ){
  sfixn * tmpA=(sfixn *)my_calloc(degA+1, sizeof(sfixn));
  register sfixn i, degR;
  if(degA<degB) { //printf("Attension: degA<degB!\n");
               for(i=0;i<=degA;i++) RPtr[i]=APtr[i];
               for(i=0;i<=degQ;i++) QPtr[i]=0;
               return;}
  for(i=0; i<=degA; i++) tmpA[i]=APtr[i];
  
  plainDiv_1(degQ, QPtr, degA, tmpA, degB, BPtr, pPtr);

  degR=degB-1;
  while((! tmpA[degR]) && (degR>0)) degR--;

  for(i=0; i<=degR; i++) RPtr[i]=tmpA[i];

  my_free(tmpA);
 
}


 /**
 * PlainPseudoRemainder: Computes the pseudo-remainder of A by B.
 * @degRAddr: (output) degree of the pseudo-remainder
 * @RPtr: (output) Coefficient vector for the reaminder (size degB)
 * @degA: degree of A
 * @APtr: Coefficient vector  of A
 * @degB: degree of B
 * @BPtr: Coefficient vector  of B 
 * @pPtr: prime number structure
 *               Using classical pseudo-division.
 *               This works for all Fourier Primes.
 * 
 * Return value: void
 **/   
void
PlainPseudoRemainder(sfixn *degRAddr, sfixn * RPtr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr){
  int i;
  sfixn * monicBPtr, Blc, BlcInv, e, co;
  if(degA<degB) {
    printf("In PlainPseudoRemainder(), degA is smaller than degB which is unexpected!\n");
    fflush(stdout);
    Interrupted=1;
    return;
  } 
  monicBPtr=(sfixn *)my_calloc(degB+1, sizeof(sfixn));
  Blc=BPtr[degB];
  BlcInv=inverseMod(Blc, pPtr->P);
  for(i=0; i<degB; i++) monicBPtr[i]=MulMod(BlcInv, BPtr[i], pPtr->P);
  monicBPtr[degB]=1;
  plainRem(degRAddr, RPtr, degA, APtr, degB, monicBPtr, pPtr);

  e=degA - degB +1;
  co=PowerMod(Blc, e, pPtr->P);
  for(i=0; i<=(*degRAddr); i++) RPtr[i]=MulMod(co, RPtr[i], pPtr->P);

  my_free(monicBPtr);
}


 /**
 * PlainPseudoQuotient: Computes the pseudo-quotient of A by B.
 * @degQAddr: (output) degree of the pseudo-remainder
 * @RPtr: (output) Coefficient vector for the reaminder (size degB)
 * @degA: degree of A
 * @APtr: Coefficient vector  of A
 * @degB: degree of B
 * @BPtr: Coefficient vector  of B 
 * @pPtr: prime number structure
 *               Using classical pseudo-division.
 *               This works for all Fourier Primes.
 * 
 * Return value: void
 **/   
void
PlainPseudoQuotient(sfixn *degQAddr, sfixn * QPtr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr){
  int i;
  sfixn * monicBPtr, Blc, BlcInv, e, co;
  if(degA<degB) {
    printf("In PlainPseudoRemainder(), degA is smaller than degB which is unexpected!\n");
    fflush(stdout);
    Interrupted=1;
    return;
  } 
  monicBPtr=(sfixn *)my_calloc(degB+1, sizeof(sfixn));
  Blc=BPtr[degB];
  BlcInv=inverseMod(Blc, pPtr->P);
  for(i=0; i<degB; i++) monicBPtr[i]=MulMod(BlcInv, BPtr[i], pPtr->P);
  monicBPtr[degB]=1;
  plainQuo(*degQAddr, QPtr, degA, APtr, degB, monicBPtr, pPtr);
 
  //printf("plainQuo=");
  //printPolyUni(*degQAddr, QPtr);
  //fflush(stdout);
  e=degA - degB +1;
  co=PowerMod(Blc, e, pPtr->P);
  for(i=0; i<=(*degQAddr); i++) QPtr[i]=MulMod(co, QPtr[i], pPtr->P);

  my_free(monicBPtr);
}

 /**
 * EX_PQuo_Uni: Computes the pseudo-quotient of A by B.
 * @degQAddr: (output) degree of the pseudo-remainder
 * @degA: degree of A
 * @APtr: Coefficient vector  of A
 * @degB: degree of B
 * @BPtr: Coefficient vector  of B 
 * @pPtr: prime number structure
 *               Using classical pseudo-division.
 *               This works for all Fourier Primes.
 * 
 * Return value: Coefficient vector for the pseudo-reaminder (accurate size)
 **/  
sfixn *
EX_PQuo_Uni(sfixn *degQAddr, sfixn degA, sfixn * APtr, sfixn degB, sfixn * BPtr, MONTP_OPT2_AS_GENE * pPtr){
  
  int i;
  sfixn * tmpQPtr, *QPtr;
  *degQAddr=degA;
  tmpQPtr=(sfixn *)my_calloc((*degQAddr)+1, sizeof(sfixn));
  PlainPseudoQuotient(degQAddr, tmpQPtr, degA, APtr, degB, BPtr, pPtr);
  *degQAddr=shrinkDegUni(*degQAddr, tmpQPtr);
  QPtr=(sfixn *)my_calloc((*degQAddr)+1, sizeof(sfixn));
  for(i=0; i<=(*degQAddr); i++) QPtr[i]=tmpQPtr[i];
  my_free(tmpQPtr);
  return QPtr;
  }

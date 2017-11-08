/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#include "SubResultantSeq.h"



sfixn
PowOfMinusOne(sfixn e){
  sfixn r=e%2;
  if(r) return -1;
  return 1;
}


void
negateVec_1(sfixn d, sfixn *vec, sfixn p){
  int i;
  for(i=0; i<=d; i++) if(vec[i]>0) vec[i]=p-vec[i];
}


/* // return 1, means YES, it's a zero uni poly. */
/* // return 0, means NO, it's not. */
/* int */
/* isZeroPolyUni(sfixn d, sfixn *vec){ */
/*   int i; */
/*   for(i=0; i<=d; i++) if(vec[i]) return 0;  */
/*   return 1; */
/* } */


void
printSRS(sfixn w, sfixn *SRS){
  int i, offset=0;
  if(SRS==NULL) {printf("SRS is NULL\n"); return;}
  if(w==0){
    //printf("0-th pseudo-remainder: ");
    //printVec(0, SRS);
    return;
  }
  for(i=0; i<w; i++){
    //printf("%d-th pseudo-remainder: ", i);
    //printVec(w-1, SRS+offset);
    offset+=w;
  }
}







// Collin's algorithm, only computes the defective ones?

// Ssz=w*w which is the size of buffer S.
sfixn *
SubResultantSeq_1(sfixn w, sfixn Ssz, sfixn *S, sfixn dg1, sfixn *f1, sfixn dg2, sfixn *f2, MONTP_OPT2_AS_GENE *pPtr)
{

  FILE *file;
  int i;
  sfixn degR, degRBefore, base_i, *degRAddr=&degR, *RPtr, base1=0, base2=0, base3=0;
  sfixn m1, m2, m3, d1, d2, a, b, bInv, g1, g2;


  // Is this correct?????????????
  if(dg1<dg2) {
    Throw 1112;
    //return SubResultantSeq_1(dg2, f2, dg1, f1, pPtr);
  }

  
  if(dg2) {w=dg2;} else {S=NULL; return S;}

  //S=(sfixn *)my_calloc(Ssz, sizeof(sfixn));
  RPtr=(sfixn *)my_calloc(w, sizeof(sfixn));
  // r_1 = pesudoRemainder (f1, f2)

  m1=dg1; m2=dg2;  d1=m1-m2;
  b= PowOfMinusOne(d1+1);
  
/*   printf("f1="); */
/*   printPolyUni(m1, f1); */
/*   printf("f2="); */
/*   printPolyUni(m2, f2); */
/*   fflush(stdout); */

  degR=m2-1;

  degRBefore = degR;
  base_i=degRBefore*w;

  PlainPseudoRemainder(degRAddr, RPtr, m1, f1, m2, f2, pPtr);

/*   printf("RPtr="); */
/*   printPolyUni(degR, RPtr); */
/*   fflush(stdout); */
  

  g1=(pPtr->P)-1;
  m3=degR;
  base1=m3*w;
  if(b==-1){
    b=pPtr->P -1;
    bInv = inverseMod(b,pPtr->P);
   
       for(i=0; i<=m3; i++) S[base1+i]=MulMod(RPtr[i], bInv, pPtr->P);
       if(base1 != base_i)copyVec_0_to_d(m3, S+base_i, S+base1);
 
      }
  else{
      copyVec_0_to_d(m3, S+base1, RPtr);
      if(base1 != base_i)copyVec_0_to_d(m3, S+base_i, S+base1);
  }
  //if(isZeroPolyUni(degR, S+base3)) return S;
  if(degR==0){
    S[0]=MulMod(S[0], S[0], pPtr->P);
      file = fopen("XXX1", "a"); 
       fprintf(file, "S --->\n");
       fprintVec(file,w*w-1, S);
      fclose(file);

    return S;}




  // r_2= pesudoRemainder (f2, r_1)
  d2=m2-m3;
  a=f2[m2];
  g2=MulMod(PowerMod((pPtr->P)-a, d1, pPtr->P), inverseMod(PowerMod(g1, d1-1, pPtr->P), pPtr->P), pPtr->P);
  b=MulMod((pPtr->P)-a, PowerMod(g2, d2, pPtr->P), pPtr->P);
  //printf("dead-2.\n");
  degR=degR-1;
  cleanVec(w-1, RPtr);



/*   printf("f1="); */
/*   printPolyUni(m2, f2); */
/*   printf("f2="); */
/*   printPolyUni(degR+1, S+base1); */
/*   fflush(stdout); */

  degRBefore = degR;
  base_i=degRBefore*w;

  PlainPseudoRemainder(degRAddr, RPtr, m2, f2, degR+1, S+base1, pPtr);


/*   printf("RPtr="); */
/*   printPolyUni(degR, RPtr); */
/*   fflush(stdout); */



  //printf("dead-3.\n");
  m2=m3; g1=g2; d1=d2;
  m3=degR;
  base2=m3*w;
  // printf("dead-3.0. b=%d\n", b);
  bInv=inverseMod(b, pPtr->P);
  //printf("dead-3.1.\n");
  for(i=0; i<=m3; i++) S[base2+i]=MulMod(bInv, RPtr[i], pPtr->P);

  if(base2 != base_i)copyVec_0_to_d(m3, S+base_i, S+base2);

  //if(isZeroPolyUni(degR, S+base3)) return S;
  if(degR==0) {
      file = fopen("XXX1", "a"); 
       fprintf(file, "S --->\n");
       fprintVec(file,w*w-1, S);
      fclose(file);
    return S;}

  

  // printf("dead-4.\n");

  // r_i+2= pesudoRemainder (r_i, r_i+1)
  
  //while(! isZeroPolyUni(m3, S+base1)){
  while(m3){
    //printf("dead-%d", ++j);
    //fflush(stdout);
    d2=m2-m3;
    a=S[base1+m2];
    g2=MulMod(PowerMod((pPtr->P)-a, d1, pPtr->P), inverseMod(PowerMod(g1, d1-1, pPtr->P), pPtr->P), pPtr->P);
    b=MulMod((pPtr->P)-a, PowerMod(g2, d2, pPtr->P), pPtr->P);
    degR=m3-1;
    cleanVec(w-1, RPtr);

    degRBefore = degR;
    base_i=degRBefore*w;

    PlainPseudoRemainder(degRAddr, RPtr, m2, S+base1, m3, S+base2, pPtr);

 

    m2=m3; g1=g2; d1=d2;
    m3=degR;
 
    base3=m3*w;

    bInv=inverseMod(b, pPtr->P);
    for(i=0; i<=m3; i++) S[base3+i]=MulMod(bInv, RPtr[i], pPtr->P);

    if(base3 != base_i)copyVec_0_to_d(m3, S+base_i, S+base3);


    base1=base2;
    base2=base3;
 
    // printf("<==========degR=%d===========>", degR);
    //printPRStest(w, S);
     
  }

  
      file = fopen("XXX1", "a"); 
       fprintf(file, "S --->\n");
       fprintVec(file,w*w-1, S);
      fclose(file);

   return S;
}





// double checked!!!!!!!!
sfixn
Lazard (sfixn x, sfixn y, int n, sfixn p){
  int a,b;
  sfixn c;
  if(n==0){printf("Error: n is zero in Lazard()!\n"); Throw 1112;}
  if(n==1){return x;}


  a=1;
  b=2*a;
  while (n>=b){
    a=b;
    b=2*a;
  }
  

  c = x;
  n = n - a;

  while(1){
    if(a==1) return c;
    a>>=1;

    c = QuoMod(MulMod(c, c, p), y, p);
    if(n >= a){
       c =  QuoMod(MulMod(c, x, p), y, p);
       n -= a;
    }
  }
}



// double checked!!!!!!!!
// return (*dgRAddr, R)
sfixn *
Lazard2(sfixn *dgRAddr, sfixn dgF, sfixn *F, sfixn x, sfixn y, int n, MONTP_OPT2_AS_GENE *pPtr){
  sfixn *R, co;

  if(n == 0) {
    printf("Error: n is zero in Lazard2()!\n"); 
    fflush(stdout);
    Throw 1112;}

  R=(sfixn *)my_calloc(dgF+1, sizeof(sfixn));
  copyVec_0_to_d(dgF, R, F);
  *dgRAddr = dgF;
  if(n == 1) { return R;}
  x = Lazard (x, y, n-1, pPtr->P);
  co = QuoMod(x, y, pPtr->P);
  //for(i=0; i<=(*dgRAddr); i++){
  coMulVec_1(co, *dgRAddr, R, pPtr);
  //}
  return R;
}





// returning -1, implys return 0.
sfixn
Reductum_1_1 (sfixn dg, sfixn *vec){
  sfixn i, dgnew;
  if(dg==0){
    vec[0]=0;
    return 0;
  }
  dgnew = shrinkDegUni(dg-1, vec);
  for(i=dgnew+1; i<=dg; i++) vec[i]=0;
  return dgnew;
}


// double checked!!!!!!!!
void copyNegateVec_0_to_d(int d, sfixn * desV, sfixn * srcV, sfixn p){
  register int i;
  for(i=0; i<=d; i++) {
     if(srcV[i]>0) desV[i]= p - srcV[i];
     if(srcV[i]==0) desV[i]=0;
     if(srcV[i]<0) Throw 1127;
  }
}

// double checked!!!!!!!!
void copyCoMulVec_0_to_d(int d, sfixn co, sfixn * desV, sfixn * srcV, MONTP_OPT2_AS_GENE *pPtr){
  register int i;
  sfixn tmp;
  tmp=(MontMulMod_OPT2_AS_GENE(co,pPtr->R2BRsft,pPtr))<<pPtr->Base_Rpow;
  for(i=0; i<=d; i++) desV[i]= MontMulMod_OPT2_AS_GENE(srcV[i], tmp, pPtr);
}

// double checked!!!!!!!!
sfixn
subMulMod_1(sfixn d1, sfixn *vec1, sfixn co, sfixn d2, sfixn *vec2, MONTP_OPT2_AS_GENE *pPtr){
  register int i;
  sfixn d=d1, tmp;
  tmp=(MontMulMod_OPT2_AS_GENE(co,pPtr->R2BRsft,pPtr))<<pPtr->Base_Rpow;
  for(i=0; i<=d2; i++)
     vec1[i] = SubMod(vec1[i], MontMulMod_OPT2_AS_GENE(vec2[i], tmp, pPtr), pPtr->P);
  if(d2>d1) {
    for(i=d2+1; i<=d1; i++){
      vec1[i]=SubMod(vec1[i], vec2[i], pPtr->P);
    }
    d = d2;}
  d = shrinkDegUni(d, vec1);
  return d;
}

// double checked!!!!!!!!
sfixn
addMulMod_1(sfixn d1, sfixn *vec1, sfixn co, sfixn d2, sfixn *vec2, MONTP_OPT2_AS_GENE *pPtr){
  register int i;
  sfixn d=d1, tmp;
  tmp=(MontMulMod_OPT2_AS_GENE(co,pPtr->R2BRsft,pPtr))<<pPtr->Base_Rpow;
  for(i=0; i<=d2; i++)
     vec1[i] = AddMod(vec1[i], MontMulMod_OPT2_AS_GENE(vec2[i], tmp, pPtr), pPtr->P);
  if(d2>d1) d = d2;
  d = shrinkDegUni(d, vec1);
  return d;
}


// double checked!!!!!!!!
sfixn
addMod_1(sfixn d1, sfixn *vec1, sfixn d2, sfixn *vec2, sfixn p){
  register int i;
  sfixn d=d1;
  for(i=0; i<=d2; i++) vec1[i] = AddMod(vec1[i], vec2[i], p);
  if(d2>d1) d = d2;
  d = shrinkDegUni(d, vec1);
  return d;
}


sfixn
coefficient(sfixn dg, sfixn *vec, int i){
  if (i>dg || i<0) return 0;
  return vec[i];
}






// double checked!!!!!!!!
// return (*dgRAddr, R)
sfixn *
next_sousResultant2 (sfixn *dgRAddr, sfixn dgP, sfixn *P, sfixn dgQ, sfixn *Q,
                     sfixn dgZ, sfixn *Z, sfixn s,  MONTP_OPT2_AS_GENE *pPtr){
  sfixn lcP, invlcP, invs, lcH, c, se, e1d1, dgPRed, dgQRed, dgZRed;
  sfixn dgHAbuf, dgH, dgHRed, *Hbuf, dgA, *Abuf, co, d, e;
  sfixn *R, *Pcopy, *Qcopy, *Zcopy;
  int i;
  lcP = P[dgP];
  c = Q[dgQ];
  se = Z[dgZ];
  d = dgP; e = dgQ;





  Pcopy=(sfixn *)my_calloc(dgP+1, sizeof(sfixn));
  Qcopy=(sfixn *)my_calloc(dgQ+1, sizeof(sfixn));
  Zcopy=(sfixn *)my_calloc(dgZ+1, sizeof(sfixn));
 
  copyVec_0_to_d(dgP, Pcopy, P);
  copyVec_0_to_d(dgQ, Qcopy, Q);
  copyVec_0_to_d(dgZ, Zcopy, Z);
  
  dgPRed = Reductum_1_1 (dgP, Pcopy);
  dgQRed = Reductum_1_1 (dgQ, Qcopy);
  dgZRed = Reductum_1_1 (dgZ, Zcopy);
  

  // get the precise maximum buffer size for H & A.
  e1d1 = d - e - 2;
  dgHAbuf = dgZRed;
  if(e1d1>0) dgHAbuf += e1d1;
  if(dgHAbuf<dgQRed) dgHAbuf = dgQRed;
  if(dgHAbuf<dgPRed) dgHAbuf = dgPRed;

  Hbuf = (sfixn *)my_calloc(dgHAbuf+1, sizeof(sfixn));
  Abuf = (sfixn *)my_calloc(dgHAbuf+1, sizeof(sfixn));

  
  dgH = dgZRed;
  copyNegateVec_0_to_d(dgH, Hbuf, Zcopy, pPtr->P);

  dgA = dgH;
  copyCoMulVec_0_to_d(dgA, coefficient(dgPRed, Pcopy, e), Abuf, Hbuf, pPtr);

  for(i=e+1; i<=d-1; i++){
    if (dgH == e-1){
      lcH = Hbuf[dgH];
      dgHRed = Reductum_1_1(dgH, Hbuf);
      dgH = shiftBigger_1(dgHRed, Hbuf, 1);
      co  = QuoMod(lcH, c, pPtr->P);
      dgH = subMulMod_1(dgH, Hbuf, co, dgQRed, Qcopy, pPtr);
    }
    else{
      dgH = shiftBigger_1(dgH, Hbuf, 1);
    }
    
    dgA = addMulMod_1(dgA, Abuf, coefficient(dgPRed, Pcopy, i), dgH, Hbuf, pPtr);
  }

  while (dgPRed >= e){
    dgPRed = Reductum_1_1 (dgPRed, Pcopy);
  }

  dgA = addMulMod_1(dgA, Abuf, se, dgPRed, Pcopy, pPtr);

  invlcP = inverseMod(lcP, pPtr->P);

  coMulVec_1(invlcP, dgA, Abuf, pPtr);

  if(dgH == e-1){
    lcH = Hbuf[dgH];
    dgH = Reductum_1_1 (dgH, Hbuf);
    dgH = shiftBigger_1(dgH, Hbuf, 1);
    dgA = addMod_1(dgA, Abuf, dgH, Hbuf, pPtr->P);
    coMulVec_1(c, dgA, Abuf, pPtr);
    dgA = subMulMod_1(dgA, Abuf, lcH, dgQRed, Qcopy, pPtr);
  }
  else{
    dgH = shiftBigger_1(dgH, Hbuf, 1);
    dgA = addMod_1(dgA, Abuf, dgH, Hbuf, pPtr->P);
    coMulVec_1(c, dgA, Abuf, pPtr);
  }

  invs = inverseMod(s, pPtr->P);
  coMulVec_1(invs, dgA, Abuf, pPtr);

  if(! ((d-e)%2)){
    negateVec_1(dgA, Abuf, pPtr->P);
  }

  *dgRAddr = dgA;
  R = (sfixn *) my_calloc((*dgRAddr) + 1, sizeof(sfixn));
  copyVec_0_to_d ((*dgRAddr), R, Abuf);
  my_free(Hbuf);
  my_free(Abuf);
  my_free(Pcopy); my_free(Qcopy); my_free(Zcopy);
  return R;
}


/**
 * EX_Resultant_Uni:
 * @dgP: the degree of univariate polynomial 'P'.
 * @P: the coefficient vector for univariate polynomial 'P'.
 * @dgQ: the degree of univariate polynomial 'Q'.
 * @Q: the coefficient vector for univariate polynomial 'Q..
 * @pPtr: the information for the prime number p.
 * 
 * To compuate the resultant of two input univariate polynomial P, and Q. 
 *
 * Return value: the resultant of P and Q.
 **/
sfixn EX_Resultant_Uni(sfixn dgP, sfixn *P, sfixn dgQ, sfixn *Q, MONTP_OPT2_AS_GENE *pPtr){
  sfixn w, Ssz, *S, res;
  if(dgP<dgQ) return EX_Resultant_Uni(dgQ, Q, dgP, P, pPtr);
  w = dgQ;
  Ssz = w*w;
  S = (sfixn *) my_calloc(Ssz, sizeof(sfixn));
  SubResultantSeq_1_new(w, Ssz, S, dgP, P, dgQ, Q, pPtr);
  res=S[0];
  my_free(S);
  return res;
}



// The function to computer sub-resultant chain.
// Ssz=w*w which is the size of buffer S.
sfixn *
SubResultantSeq_1_new(sfixn w, sfixn Ssz, sfixn *S, sfixn dgP, sfixn *P, sfixn dgQ, sfixn *Q, MONTP_OPT2_AS_GENE *pPtr)
{ sfixn s, dgf1, *f1, dgf2, *f2, *tmpPtr, tmp, *dgRAddr, dgZ, *Z=NULL, base;

  if (isZeroPolyUni(dgP, P)) return S;
  if (isZeroPolyUni(dgQ, Q)) return S;
  
  if (dgP<dgQ){
    printf("deg(p)<deg(Q) in  SubResultantSeq_1_new()!\n");
    fflush(stdout);
    Throw 111;
  }

  if (dgQ==0) {
    S[0] = PowerMod(Q[dgQ], dgP, pPtr->P);
    return S;
  }
  s = PowerMod(Q[dgQ], dgP - dgQ, pPtr->P);

  // 0-f1 to be freed.
  f1 = (sfixn *) my_calloc(dgQ+1, sizeof(sfixn));
  dgf1 = dgQ;
  copyNegateVec_0_to_d(dgf1, f1, Q, pPtr->P);
  dgf2 = dgQ-1;
  dgRAddr = &dgf2;
  f2 = (sfixn *) my_calloc(dgf2+1, sizeof(sfixn));
  PlainPseudoRemainder(dgRAddr, f2, dgP, P, dgf1, f1, pPtr);

  copyVec_0_to_d(dgf2, S+(dgf1-1)*w, f2);

  negateVec_1(dgf1, f1, pPtr->P);



  if(isZeroPolyUni(dgf2, f2)){
        my_free(f2);

        return S;}





  while(1){
    if(isZeroPolyUni(dgf2, f2)){
        if(Z != NULL) my_free(Z);
         my_free(f1); my_free(f2);
         return S;
    }
  
    // TRICK!!!! 
    copyVec_0_to_d(dgf2, S+(dgf1-1)*w, f2);
    

    // 2--Z to be freed.
    if(Z != NULL) my_free(Z);

    dgRAddr = &dgZ;




    Z = Lazard2(dgRAddr, dgf2, f2, f2[dgf2], s, dgf1-dgf2, pPtr);

    base=dgZ*w;

    if (dgZ >= w){
      printf("writing out of range!!\n");
      printf("dgZ=%ld, w-1=%ld\n", dgZ, w-1);
      fflush(stdout);
      Throw 1212;

    }
     
    if (( dgf1-dgf2) > 1) copyVec_0_to_d(dgZ, S+base, Z);



    if(dgZ == 0){
       my_free(f1); my_free(f2);
       if(Z != NULL) my_free(Z);
       return S;
    }

    
    tmpPtr = f2;
    tmp = dgf2;
    dgRAddr = &dgf2;
    //f2 to be freed.
    f2 = next_sousResultant2 (dgRAddr, dgf1, f1, dgf2, f2, dgZ, Z, s,  pPtr);
    // 0-f1 is freed.
    my_free(f1);
    f1 = tmpPtr;
    dgf1 = tmp;
    s = Z[dgZ];
  }


  return S;
}

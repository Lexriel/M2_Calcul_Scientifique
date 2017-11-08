/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#include "FINTERP.h"

/////////////////////////////////////////////////////////////////////////////////////////////
//
// (1) Functions TFTEvalMultiD and TFTInterpMultiD added:
//    
//     TFTEvalMultiD is used to evaluate a multivariate polynomial in [x1, ..., xn]
//     to a polynomial in x(M+1),..., x(N).
//
//     TFTInterpMultiD is used to interpolate a multivariate polynomial in [x1, ... xn]
//     from a polynomial in x(M+1),... x(N). This is the inverse of TFTEvalMultiD.
//
//     WARNING, TFTInterpMultiD will create a polynomial structure. Make sure NOT to
//     allocate space for the output polynomial and make sure to free the polynomial returned
//     by this function.  
//
//     WP, 2009/09/03/
//
// (2) TFT related functions for retrieving data from a TFT cube 
//
//     getSubResultantChainsTFT
//     interpIthSliceTFT
//     interpIthDthSliceTFT
//     interpNextCandidateSliceLCTFTDefective
//     interpNextCandidateSliceLCTFT
//
//     They are all similar to their DFT counterparts; TFT interpolations are used.
//
//     WP, 2009/09/11/
//
// (3) Create findGoodRoots function for finding primitive roots of unity such the initials
//     of two input polynomials do not vanish at the grid defined by those roots. 
//     This function is the cleaned version of createGoodRootsForf1f2.
//
/////////////////////////////////////////////////////////////////////////////////////////////

extern int Interrupted;
extern int PrimeError;

/**
 * Subproduct tree related methods
 */
void mulNodes(sfixn dr, sfixn *srcAddr, sfixn ds, sfixn *Addr1, sfixn *Addr2, 
    sfixn nodeSz, MONTP_OPT2_AS_GENE *pPtr)
{
    if (nodeSz == 1) 
        srcAddr[0] = (Addr1[0] * Addr2[0]) % (pPtr->P);
    else
        EX_Mont_FFTMul(dr, srcAddr, ds, Addr1, ds, Addr2, pPtr);
}

/**
 * subProdTreeCre:
 * @itemNo: number of points.
 * @itemSz: data size of each leaf in terms of sfixn.
 * @items: the leave of the sub-product tree.
 * @p: prime number.
 * 
 * Create a sub-product tree.
 * 
 * Return value: sub-product tree.
 **/
subProdTree *subProdTreeCre(sfixn itemNo, sfixn itemSz, sfixn *items, sfixn p)
{
  subProdTree *tree;
  sfixn h=logceiling(itemNo), levels;
  int i, j, k, totSZ=itemNo*itemSz;
  int q, r, base1, base2, Wdouble, tmp;
  MONTP_OPT2_AS_GENE  prime;

  signal(SIGINT,catch_intr);

  tree =(subProdTree *)my_malloc(sizeof(subProdTree));
  tree->h=h;
  levels=h+1;
  tree->W = (sfixn *)my_calloc(levels+1, sizeof(sfixn));
  tree->NoNodes =(sfixn *)my_calloc(levels+1, sizeof(sfixn));
  tree->Bases =(sfixn *)my_calloc(levels+1, sizeof(sfixn));

  EX_MontP_Init_OPT2_AS_GENE(&prime, p);
  (tree->W)[1]=itemSz; 
  (tree->NoNodes)[1]=itemNo; // W_s=(sfixn *)my_calloc(WNB_sSz, sizeof(sfixn));
  for(i=2; i<=levels; i++) {
    (tree->W)[i]=(tree->W)[i-1]*2-1;
    (tree->NoNodes)[i]=(int)ceil((tree->NoNodes)[i-1]/(double)2);
    totSZ+=(tree->W)[i]*(tree->NoNodes)[i];
    //printf("W[%d]=%d, NoNodes[%d]=%d.\n", i, (tree->W)[i], i, (tree->NoNodes)[i]);
  } 
  tree->data=(sfixn *)my_calloc(totSZ, sizeof(sfixn));
  tmp=itemNo*itemSz;
  base2=totSZ-tmp;
  copyVec_0_to_d(tmp-1, (tree->data)+base2, items);

  for(i=1; i<levels; i++){
    tree->Bases[i]=base2;
    q=(tree->NoNodes)[i]/2;
    r=(tree->NoNodes)[i]-q*2;
    base1=base2-(tree->NoNodes)[i+1]*(tree->W)[i+1];
    Wdouble=(tree->W)[i]*2;
    for(j=0, k=0; j<q*Wdouble; j+=Wdouble ){
      mulNodes((tree->W)[i+1]-1, (tree->data)+base1+k, (tree->W)[i]-1, 
        (tree->data)+base2+j, (tree->data)+base2+j+(tree->W)[i], (tree->W)[i], &prime);
      if(PrimeError==1) {
        subProdTreeFree(tree);
        return NULL;
	  } 
      k+=(tree->W)[i+1];
    }
    if(r){
      copyVec_0_to_d((tree->W)[i]-1, (tree->data)+base1+k, (tree->data)+base2+j);
    }
    base2=base1;
    
  } 
  return tree;
}

/**
 * subProdTreeFree:
 * @tree: a sub-product tree data structure.
 * 
 * Deallocate 'tree' (deep free). 
 *
 * Return value: 
 **/
void subProdTreeFree(subProdTree * tree){
  if(tree){
    if(tree->W) {my_free(tree->W); (tree->W)=NULL;}
    if(tree->NoNodes) {my_free(tree->NoNodes); (tree->NoNodes)=NULL;}
    if(tree->data){my_free(tree->data); (tree->data)=NULL;}
    my_free(tree);
    tree=NULL;
  }
}

/**
 * printSubProdTree:
 * @tree: a sub-product tree.
 * Print 'tree'.
 * Return value: 
 **/
void printSubProdTree(subProdTree * tree){
  int i,j,k;
  sfixn *base;
  printf("==============Sub Product Tree================\n");
  printf("The height of the tree: %d.\n", tree->h);
  printf("the node size from level-1 to level-%d are:", (tree->h)+1);
  printVecFrom1((tree->h)+1, (tree->W));
  printf("the number of nodes from level-1 to level-%d are:", (tree->h)+1);
  printVecFrom1((tree->h)+1, (tree->NoNodes));
  printf("the bases of nodes from level-1 to level-%d are:", (tree->h)+1);
  printVecFrom1((tree->h)+1, (tree->Bases));

  for(i=(tree->h)+1; i>=1; i--){
    printf("level=%d [", i);
    fflush(stdout);
    base=tree->data+(tree->Bases)[i];
    for(j=1; j<=(tree->NoNodes)[i]; j++){
      if(j>1) {printf(", ");     fflush(stdout);}
      printf("%d", base[0]);     fflush(stdout); 
      for(k=1; k<(tree->W)[i]; k++){
        if(base[k]!=0) { 
             printf("+%dx", base[k]);
                 fflush(stdout);
	     if(k>1) printf("^%d", k);
                  fflush(stdout);
        }
      }
      base+=(tree->W)[i];
    }
    printf("]\n");
        fflush(stdout);
  }
  printf("==============================================\n");
}

/**
 * SlowEvaluation1pt:
 * @degf: the degree of univariate polynomial 'f'.
 * @fPtr: coefficient vector of univariate polynomial 'f'.
 * @pt: a point value.
 * @p: a prime number.
 * 
 * Evaluate the the univariate polynomial 'f' at point 'pt'. 
 *
 * Return value: the image value after the evaluation.
 **/
sfixn SlowEvaluation1pt(sfixn degf, sfixn *fPtr, sfixn pt, sfixn p){
  int j;
  sfixn pow;
  sfixn r;
  r=fPtr[0];
  for(j=1; j<=degf; j++){
         pow=PowerMod(pt, j, p);
         r=AddMod(r, MulMod(fPtr[j],pow,p), p);    
  }
  return r;
}

/**
 * SlowEvaluation:
 * @degf: the degree of the univariate polynomial 'f'.
 * @fPtr: the coefficient vector of the univariate polynomial 'f'.
 * @nopts: number of point for the evaluation.
 * @pts: multiple point for the evalution.
 * @p: the prime number.
 *
 * Multiple points evaluation.
 * There are 'nopts' points kept in the sfixn vector 'pts'. 
 * The input univariate polynomial 'f' with degree 'degf' and 
 * coefficients 'fPtr' will be evaluated at 'pts[i], i=0,...,nopts-1'.
 *
 * Return value: the evaluation images (a sfixn vector) after the evaluation. 
 *               E.g. the first image is for the evaluation at the first point
 *               pts[0].
 **/
sfixn *SlowEvaluation(sfixn degf, sfixn *fPtr, sfixn nopts, sfixn *pts, sfixn p)
{
  int i,j;
  sfixn pow;
  sfixn r;
  sfixn * result=(sfixn *)my_calloc(nopts, sizeof(sfixn));
  for(i=0; i<nopts; i++){
     r=fPtr[0];
     for(j=1; j<=degf; j++){
         pow=PowerMod(pts[i], j, p);
         r=AddMod(r, MulMod(fPtr[j],pow,p), p);    
     }
     result[i]=r;
  }
  return result;
}

/**
 * FastEvaluation:
 * @degf: the degree of 'f'.
 * @fPtr: the coefficient vector of 'f'.
 * @tree: the sub-product tree.
 * @p: the prime number.
 * 
 * Fast evaluation of the input univariate polynomial 'f' by
 * the give sub-product tree.
 *
 * Return value: a vector keeps the evaluation images. 
 *               The first image is for the evaluation at the first point. 
 **/
sfixn *FastEvaluation(sfixn n, sfixn degf, sfixn *fPtr, subProdTree * tree, sfixn p)
{
  sfixn *result;  
  int i,j,  offset, offset2, dr, dd, q, r;
  sfixn drr1, *drrAddr1=&drr1, drr2, *drrAddr2=&drr2 ;
  sfixn * tmpRem, * tmpRem2;
  int tmpRemsz=(tree->W)[tree->h]-1, resultsz, resultszi, 
      resultsziPre;
  MONTP_OPT2_AS_GENE  prime;

  EX_MontP_Init_OPT2_AS_GENE(&prime, p);

  if (tree->h ==0) {
    result=(sfixn *)my_calloc(n, sizeof(sfixn));
    copyVec_0_to_d(degf, result, fPtr);
    return result;
  }

  resultsz=1<<tree->h;
  result=(sfixn *)my_calloc(resultsz, sizeof(sfixn));
  tmpRem=(sfixn *)my_calloc(tmpRemsz, sizeof(sfixn));
  tmpRem2=(sfixn *)my_calloc(tmpRemsz, sizeof(sfixn));
  resultsziPre=resultsz;
  resultszi=resultsz>>1;
  // the level next to the root. root is level tree->h +1.
  // keep the remainders in the intermediate steps.
  // Do the first two divisions.
  dd=(tree->W)[tree->h]-1;
  
  dd=shrinkDegUni(dd, (tree->data)+(tree->Bases)[tree->h]); 
  drr1=dd-1;
  UniRem(drrAddr1, tmpRem, degf, fPtr, dd, 
	      (tree->data)+(tree->Bases)[tree->h], &prime );
    
  // Only last node may be shorter.
  dd=shrinkDegUni((tree->W)[tree->h]-1, (tree->data)+(tree->Bases)[tree->h]+(tree->W[tree->h]));  
  drr2=dd-1;
  UniRem(drrAddr2, tmpRem2, degf, fPtr, dd, 
           (tree->data)+(tree->Bases)[tree->h]+(tree->W[tree->h]), &prime );
 
  cleanVec((tree->NoNodes)[1]-1, result); 
  copyVec_0_to_d(drr1, result+0, tmpRem);
  copyVec_0_to_d(drr2, result+resultszi, tmpRem2);
  
  for(i=(tree->h)-1; i>=1; i--){
    offset=0;
    offset2=0;
    resultsziPre=resultszi;
    resultszi>>=1;
    q=(tree->NoNodes[i])/2;
    r=(tree->NoNodes[i])%2;
    for(j=1; j<=q; j++){
    
      dr= shrinkDegUni(resultsziPre-1, result + offset);

      dd= shrinkDegUni((tree->W)[i]-1, (tree->data)+(tree->Bases)[i]+offset2);
      drr1=dd-1;
      cleanVec(tmpRemsz-1, tmpRem); 
      UniRem(drrAddr1,tmpRem, dr, result+offset, dd, 
           (tree->data)+(tree->Bases)[i]+offset2, &prime );

      offset2+=tree->W[i];

      dd= shrinkDegUni((tree->W)[i]-1, (tree->data)+(tree->Bases)[i]+offset2);
      drr2=dd-1;
      cleanVec(tmpRemsz-1, tmpRem2);
      UniRem(drrAddr2,tmpRem2, dr, result+offset, dd, 
           (tree->data)+(tree->Bases)[i]+offset2, &prime );

      cleanVec(resultsziPre-1, result+offset);
      copyVec_0_to_d(drr1, result+offset, tmpRem);
      copyVec_0_to_d(drr2, result+offset+resultszi, tmpRem2);
     
      offset+=resultsziPre;
      offset2+=tree->W[i];
    
    }

    if (r){
      dr= shrinkDegUni(resultsziPre-1, result + offset); 
      dd=shrinkDegUni((tree->W)[i]-1, 
           (tree->data)+(tree->Bases)[i]+offset2); 
      drr1=dd-1;
      cleanVec(tmpRemsz-1, tmpRem);
      UniRem(drrAddr1,tmpRem, dr, result+offset, dd, 
           (tree->data)+(tree->Bases)[i]+offset2, &prime );
      cleanVec(resultsziPre-1, result+offset);
      copyVec_0_to_d(drr1, result+offset, tmpRem);
    }
  }
  my_free(tmpRem);
  my_free(tmpRem2);
   return result;
}

/**
 * FastEvaluation_1:
 * @resultsz: the size of the vector 'result'.
 * @result: the vector keeps the evaluation images.
 * @degf: the degree of 'f'.
 * @fPtr: the coefficient vector of 'f'.
 * @tree: the sub-product tree.
 * @p: the prime number.
 * 
 * Fast evaluation of the input univariate polynomial 'f' by
 * the give sub-product tree.
 * The vector 'result' keeps the evaluation images. 
 *       The first image is for the evaluation at the first point.
 * Return value:  
 **/
void FastEvaluation_1(sfixn resultsz, sfixn *result, sfixn degf, sfixn *fPtr, 
    subProdTree *tree, sfixn p)
{
  int i,j,offset, offset2, dr, dd, q, r;
  sfixn drr1, *drrAddr1=&drr1, drr2, *drrAddr2=&drr2 ;
  sfixn * tmpRem, * tmpRem2;
  int tmpRemsz=(tree->W)[tree->h]-1,  resultszi, 
      resultsziPre;
  MONTP_OPT2_AS_GENE  prime;

  EX_MontP_Init_OPT2_AS_GENE(&prime, p);

  //if (tree->h ==0) return;
  if(tree->h == 0){
    copyVec_0_to_d(degf, result, fPtr);
    return;
  }

  // resultsz=1<<tree->h;
  //result=my_calloc(resultsz, sizeof(sfixn));
  tmpRem=(sfixn *)my_calloc(tmpRemsz, sizeof(sfixn));
  tmpRem2=(sfixn *)my_calloc(tmpRemsz, sizeof(sfixn));
  resultsziPre=resultsz;
  resultszi=resultsz>>1;
  // the level next to the root. root is level tree->h +1.
  // keep the remainders in the intermediate steps.
  // Do the first two divisions.
  dd=(tree->W)[tree->h]-1;

  dd=shrinkDegUni(dd, (tree->data)+(tree->Bases)[tree->h]); 
  drr1=dd-1;
  UniRem(drrAddr1, tmpRem, degf, fPtr, dd, 
	      (tree->data)+(tree->Bases)[tree->h], &prime );
    
  // Only last node may be shorter.
  dd=shrinkDegUni((tree->W)[tree->h]-1, (tree->data)+(tree->Bases)[tree->h]+(tree->W[tree->h]));  
  drr2=dd-1;
  UniRem(drrAddr2, tmpRem2, degf, fPtr, dd, 
           (tree->data)+(tree->Bases)[tree->h]+(tree->W[tree->h]), &prime );
 
  cleanVec((tree->NoNodes)[1]-1, result); 
  copyVec_0_to_d(drr1, result+0, tmpRem);
  copyVec_0_to_d(drr2, result+resultszi, tmpRem2);
  
  for(i=(tree->h)-1; i>=1; i--){
    offset=0;
    offset2=0;
    resultsziPre=resultszi;
    resultszi>>=1;
    q=(tree->NoNodes[i])/2;
    r=(tree->NoNodes[i])%2;
    for(j=1; j<=q; j++){
   
      dr= shrinkDegUni(resultsziPre-1, result + offset);

      dd= shrinkDegUni((tree->W)[i]-1, (tree->data)+(tree->Bases)[i]+offset2);
      drr1=dd-1;
      cleanVec(tmpRemsz-1, tmpRem); 
      UniRem(drrAddr1,tmpRem, dr, result+offset, dd, 
           (tree->data)+(tree->Bases)[i]+offset2, &prime );


      offset2+=tree->W[i];

      dd= shrinkDegUni((tree->W)[i]-1, (tree->data)+(tree->Bases)[i]+offset2);
      drr2=dd-1;
      cleanVec(tmpRemsz-1, tmpRem2);
      UniRem(drrAddr2,tmpRem2, dr, result+offset, dd, 
           (tree->data)+(tree->Bases)[i]+offset2, &prime );

      cleanVec(resultsziPre-1, result+offset);
      copyVec_0_to_d(drr1, result+offset, tmpRem);
      copyVec_0_to_d(drr2, result+offset+resultszi, tmpRem2);
     
      offset+=resultsziPre;
      offset2+=tree->W[i];
    
    }

    if (r){
  
      dr= shrinkDegUni(resultsziPre-1, result + offset); 
      dd=shrinkDegUni((tree->W)[i]-1, 
           (tree->data)+(tree->Bases)[i]+offset2); 
      drr1=dd-1;
      cleanVec(tmpRemsz-1, tmpRem);
      UniRem(drrAddr1,tmpRem, dr, result+offset, dd, 
           (tree->data)+(tree->Bases)[i]+offset2, &prime );
      cleanVec(resultsziPre-1, result+offset);
      copyVec_0_to_d(drr1, result+offset, tmpRem);
    }
  }

  my_free(tmpRem);
  my_free(tmpRem2);
}

//inner function of linearCombineModulus().
void vecCrossMul(sfixn* R, sfixn Anodesz, sfixn *A, sfixn BNoNodes, 
    sfixn Bnodesz, sfixn *B, MONTP_OPT2_AS_GENE *pPtr, int firsttime)
{
  int r=BNoNodes%2,nn;
  int i, j, k, m, doubleBsz=Bnodesz*2, doubleAsz=Anodesz*2;

  sfixn dB;
  sfixn dA;
  sfixn degRes;
  sfixn *tmpPtr1;
  sfixn *tmpPtr2;
  if(r) nn=BNoNodes-1; else nn=BNoNodes;

  dB=Bnodesz-1;
  dA=Anodesz-1;
  degRes=dB+dA;
  tmpPtr1=(sfixn *)my_calloc(degRes+1, sizeof(sfixn));
  tmpPtr2=(sfixn *)my_calloc(degRes+1, sizeof(sfixn));

    for (i=0, j=0, k=0; i<nn*Bnodesz; i+=doubleBsz, j+=doubleAsz, k+=(degRes+1)){
      cleanVec(degRes, tmpPtr1);
      cleanVec(degRes, tmpPtr2);
      //dA=shrinkDegUni(dA, A+j);
      //dB=shrinkDegUni(dB, B+i+Bnodesz); 
      EX_Mont_FFTMul(degRes, tmpPtr1, dA, A+j, dB, B+i+Bnodesz, pPtr);
      //dA=shrinkDegUni(dA,  A+j+Anodesz);
      //dB=shrinkDegUni(dB, B+i); 
      EX_Mont_FFTMul(degRes, tmpPtr2, dA, A+j+Anodesz, dB, B+i, pPtr);
      for(m=0; m<=degRes; m++){
	R[k+m]=AddMod(tmpPtr1[m], tmpPtr2[m], pPtr->P);
      }

  }

  if(r){

    if(firsttime){
      //printf("B[%d]=%d.\n", nn, B[nn]);
      R[k]=B[nn];
      } else{
        for(m=0; m<=dA; m++){
          R[k+m]=A[j+m];
        }
    }

  }
      my_free(tmpPtr1); my_free(tmpPtr2);
}

// the number of elems in Cs[] should equal to the leaves of tree.
// Inner function of fastInterp().
sfixn * linearCombineModulus(sfixn *Cs, subProdTree * tree, sfixn p){

  int i;
  sfixn * R, * A, *B=Cs;
  sfixn dB=0, dA=1, degRes, BNodeSz=1, ANodeSz=2, 
    Asz=((tree->W)[1])*((tree->NoNodes)[1]);
  MONTP_OPT2_AS_GENE  prime;
  EX_MontP_Init_OPT2_AS_GENE(&prime, p);

 
  A=(sfixn *)my_calloc( Asz,sizeof(sfixn));
  copyVec_0_to_d(Asz-1, A, (tree->data)+(tree->Bases)[1]);
  for(i=1; i<=(tree->h); i++){
   
    degRes=dB+dA;
    R=(sfixn *)my_calloc(Asz, sizeof(sfixn));  
    //A=(tree->data)+(tree->Bases)[i];  
    if(i==1)
     vecCrossMul(R, ANodeSz, A, (tree->NoNodes)[i], BNodeSz, B, &prime, 1);
    else
     vecCrossMul(R, ANodeSz, A, (tree->NoNodes)[i], BNodeSz, B, &prime, 0);
    dB=(tree->W)[i+1]-1;
    BNodeSz=(tree->W[i+1]);
    B=(tree->data)+(tree->Bases)[i+1]; 
    dA=degRes;
    ANodeSz=degRes+1;
    copyVec_0_to_d(Asz-1, A, R);
    // printf("result is:\n");
    //printVec(Asz-1, A);
    my_free(R);
  }
    return A;
}

// Inner function of fastInterp().
sfixn *createLeaves(sfixn n, sfixn *Us, sfixn p) {
    sfixn *Leaves = (sfixn *)my_calloc(2*n, sizeof(sfixn));
    int i, j;
    for(i=0, j=0; i<n; i++, j+=2){
        Leaves[j] = p - Us[i];
        Leaves[j+1] = 1;
    }
    return Leaves;
}

// return the (D(f))
// Inner function of direvative().
/**
 * direvative:
 * @deg: the degree of the input univariate polynomial 'f'.
 * @coef: the coefficient vector of the input univariate polynomial 'f'.
 * @p: the prime number.
 * 
 * Compute the derivative of intput univariate polynomial 'f'.
 *
 * Return value: the coefficient vector the derivative of 'f'
 **/
sfixn *direvative(sfixn deg, sfixn *coef, sfixn p) {
  int i;
  sfixn *newcoef;
  if(deg==0) {
    newcoef=(sfixn *)my_calloc(1, sizeof(sfixn));
  }
  else{
    newcoef=(sfixn *)my_calloc(deg+1, sizeof(sfixn));
     for(i=0; i<deg; i++){
       newcoef[i]=MulMod(coef[i+1], i+1, p);
     }
  }
  return newcoef;
}

// n # of points.
// Us the evaluation points.
// Vs the evaluation images.
// Return the interpolated polynomial with degree "(*polyDg)".
/**
 * fastInterp:
 * @polyDg: pointer to the degree of the output univariate polynomial.
 * @n: the number of points.
 * @Us: the evalution points.
 * @tree: sub-product tree.
 * @Vs: the evaluation images (the output of fast evaluation).
 * @p: the prime number p.
 * 
 * Return value: the interpolated polynoimal.
 **/
sfixn *fastInterp(sfixn *polyDg, sfixn n, sfixn *Us, subProdTree *tree, 
    sfixn *Vs, sfixn p) 
{
    int i;
    sfixn *Leaves, *tmpresult, *result, *SsVs, *Dm, dgDm;

    if (tree->h == 0) {
        result = (sfixn *)my_calloc(n, sizeof(sfixn));
        copyVec_0_to_d(n-1, result, Vs);
        *polyDg = n-1;
        return result;
    }

    Leaves = createLeaves(n, Us, p);
    Dm = direvative((tree->W)[(tree->h)+1]-1, (tree->data) +(tree->Bases)[(tree->h)+1], p);
    dgDm = shrinkDegUni((tree->W)[(tree->h)+1]-1, Dm);
    SsVs = FastEvaluation(n, dgDm, Dm, tree, p);
    for(i=0; i<n; i++) SsVs[i]=MulMod(inverseMod(SsVs[i], p), Vs[i], p);

    tmpresult=linearCombineModulus(SsVs, tree, p);
    (* polyDg) = shrinkDegUni(((tree->W)[1])*((tree->NoNodes)[1])-1, tmpresult);  

    result=(sfixn *)my_calloc(((*polyDg)+1), sizeof(sfixn));
    dgDm=shrinkDegUni((tree->W)[(tree->h)+1]-1, Dm);
    copyVec_0_to_d((* polyDg), result, tmpresult);

    my_free(SsVs);
    my_free(Leaves);
    my_free(tmpresult);
    return result;
}

/**
 * fastInterp_1:
 * @polyDg: pointer to the degree of the output univariate polynomial.
 * @result: (output) the coefficients of output univariate polynomial.
 * @n: the number of points.
 * @Us: the evalution points.
 * @tree: sub-product tree.
 * @Vs: the evaluation images (the output of fast evaluation).
 * @p: the prime number p.
 * 
 * Return value: the interpolated polynoimal.
 **/
void fastInterp_1(sfixn *polyDg, sfixn *result, sfixn n, sfixn *Us, 
    subProdTree * tree, sfixn *Vs, sfixn p)
{
    int i;
    sfixn *Leaves, *tmpresult, *SsVs, *Dm, dgDm;

    if (tree->h == 0) {
        copyVec_0_to_d(n-1, result, Vs);
        *polyDg=n-1;
        return;
    }
    Leaves = createLeaves(n, Us, p);
    Dm = direvative((tree->W)[(tree->h)+1]-1, (tree->data) +(tree->Bases)[(tree->h)+1], p);
    dgDm = shrinkDegUni((tree->W)[(tree->h)+1]-1, Dm);
    SsVs = FastEvaluation(n, dgDm, Dm, tree, p);
    for(i=0; i<n; i++) SsVs[i]=MulMod(inverseMod(SsVs[i], p), Vs[i], p);
    tmpresult=linearCombineModulus(SsVs, tree, p);
    (* polyDg)=shrinkDegUni(((tree->W)[1])*((tree->NoNodes)[1])-1, tmpresult);  
    copyVec_0_to_d((* polyDg), result, tmpresult);
    my_free(SsVs);
    my_free(Leaves);
    my_free(tmpresult);
}

// generating n distinct points from the startVal+1.
// return the last value generated.
// interval >1
sfixn createRandPtsFromTo(sfixn n, sfixn *vec, sfixn startVal, sfixn interval){
  int i=0, incr=0;
  // generated nothing.
  if(n==0) return -1;
  vec[i++]=startVal+1;
  while(i<n){
    while(incr==0) {incr=rand()%interval+1;}
    //while(incr==0) {incr=6+1;}
    vec[i]=incr+vec[i-1];
    i++;
  }
  return(vec[--i]);
}

// bounds[0] is not used.
// bounds[1]..bounds[m], they are the size of the buffer.
sfixn * createPts(sfixn start, sfixn m, sfixn *bounds, sfixn p){
  int i;
  sfixn sz=0, *pts,  offset=0, interval=10;
  for(i=1; i<=m; i++){sz+=bounds[i];}
  pts=(sfixn *)my_calloc(sz, sizeof(sfixn));
  for(i=1; i<=m; i++){
     start=createRandPtsFromTo(bounds[i], pts+offset, start, interval);
     offset+=bounds[i];
  }
  if(start>=p){
    Interrupted=1; PrimeError=1;
    return pts;
  }
  return pts;
}

// inner function of createGoodPtsForf1f2().
sfixn ** convertpts2ptsPtr(sfixn m, sfixn *bounds, sfixn *pts){
  int i, j, offset=0;
  sfixn **ptsPtr=(sfixn **)my_calloc(m+1, sizeof(sfixn *));
  for(i=1; i<=m; i++){
    ptsPtr[i]=(sfixn *)my_calloc(bounds[i], sizeof(sfixn));
    for(j=0; j<bounds[i]; j++) (ptsPtr[i])[j]=pts[offset+j];
    offset+=bounds[i];
  }
  return ptsPtr;
}


// To create points for evaluting input C-Cube polynomials 'f1' and 'f2'.
// The points are "good" in sense of they will make the initals of 'f1' and
// 'f2' vanish.
// Return a data structure "result:
//  result->no             is for the number of points;
//  result->ptsPtr         is a vector keeping the points;
//  result->trees=trees;   a vecotr of sub-product trees. 
//  result->times=times;   developer information.
PTS_TREE* createGoodPtsForf1f2(sfixn N, sfixn d1, preFFTRep *f1, sfixn d2, 
    preFFTRep *f2, sfixn m, sfixn *bounds, sfixn *dims1, 
    sfixn *dims2, MONTP_OPT2_AS_GENE *pPtr)
{
    int times=1, i, nogood1=1, nogood2=1;
    sfixn start = 1, n, jump = 2;
    sfixn *pts, *coE, coEsz, const1, const2;
    PTS_TREE* result;
    subProdTree** trees;
    sfixn **ptsPtr;

    backupData(f1);
    decreaseOneDim(f1); 
    nextMCoefData(f1,N,d1);
    backupData(f2);
    decreaseOneDim(f2); 
    nextMCoefData(f2,N,d2); 
    const1 = constantPolyp(f1);
    const2 = constantPolyp(f2);
#ifndef DUBNOGOODPOINTS
    if (!(const1 && const2)){
#else
    if (0) {
#endif
    // n=N-1 in this case.
    n = N(f1);
    coEsz = 1;
    for(i=1; i<=n; i++) { coEsz*=dims1[i]; }
    coE = (sfixn *)my_calloc(coEsz, sizeof(sfixn));

    while(nogood1 || nogood2){
        if ((Interrupted==1)||(PrimeError==1)) break; 
        pts = createPts(start, m, bounds, pPtr->P);
        trees = createArrOfSPTrees(m, bounds, pts, pPtr->P);
        if (trees==NULL) { break; }
        ptsPtr = convertpts2ptsPtr(m, bounds, pts);
        my_free(pts);
        
        cleanVec(coEsz-1, coE);
        fastEvalMulti_test(m, dims1, coE, f1, trees, pPtr);
        nogood1=isVecContainsZero(coEsz, coE);
        
        cleanVec(coEsz-1, coE);
        fastEvalMulti_test(m, dims2, coE, f2, trees, pPtr);
        nogood2=isVecContainsZero(coEsz, coE);

        if (nogood1 || nogood2) {
            freeVecVec(m, ptsPtr);
            freeArrOfSPTTrees(m, trees);
            start += jump;
            times++;
            if (times>1000) {
	            printf("run out of good points!\n");
                fflush(stdout);
                Interrupted=1; 
                PrimeError=1;
            }
            continue;
        } 
    }
    my_free(coE);
    } else {
        pts = createPts(start, m, bounds, pPtr->P);
        trees = createArrOfSPTrees(m, bounds, pts, pPtr->P);
        ptsPtr = convertpts2ptsPtr(m, bounds, pts);
        my_free(pts);
        times = 1;
    }

    increaseOneDim(f1);
    restoreData(f1);
    increaseOneDim(f2);
    restoreData(f2);
    if ((Interrupted==1)|| (PrimeError==1)) return NULL; 

    result=(PTS_TREE*)my_malloc(sizeof(PTS_TREE));
    result->no=m;
    result->ptsPtr=ptsPtr;
    result->trees=trees;
    result->times=times;
    return result;
}

void freePTS_TREE(PTS_TREE *pt) {
    if(pt){
        // free 1..no.
        if(pt->ptsPtr) freeVecVec((pt->no), pt->ptsPtr);
        if(pt->trees) freeArrOfSPTTrees((pt->no), pt->trees);
        my_free(pt);
    }
}

sfixn * Pts2Leaves(sfixn m, sfixn *bounds, sfixn *pts, sfixn p){
  int i; 
  sfixn sz=0, *leaves;
  for(i=1; i<=m; i++){sz+=bounds[i];}
  leaves=(sfixn *)my_calloc(sz*2, sizeof(sfixn));
  for(i=0; i<sz; i++) {leaves[i*2]=p-pts[i]; leaves[2*i+1]=1;}
  return leaves;
}

//  trees[0] is not used.
//  trees[i] is for evaluating x_i.
subProdTree** createArrOfSPTrees(sfixn m, sfixn *bounds, sfixn *pts, sfixn p){
  int i, offset=0;
  sfixn *leaves;
  subProdTree** trees;
  trees=(subProdTree**)my_malloc((m+1) * sizeof(subProdTree *));
  leaves=Pts2Leaves(m, bounds, pts, p);
  for(i=1; i<=m; i++){
    trees[i]=subProdTreeCre(bounds[i], 2, leaves+offset, p);
    if(trees[i]==NULL){
       my_free(leaves);
       my_free(trees);
       return NULL;
    }
    offset+=bounds[i]*2;
  }
  my_free(leaves);
  return trees;
}

void freeArrOfSPTTrees(sfixn m, subProdTree **trees){
    int i;
    if(trees){
        for(i=1; i<=m; i++){ subProdTreeFree(trees[i]); }
        my_free(trees);
    }
}

void printArrOfSPTTrees(sfixn m, subProdTree **trees){
    int i;
    for(i=1; i<=m; i++) printSubProdTree(trees[i]);
}

//sfixn *
//FastEvaluation(sfixn degf, sfixn *fPtr, subProdTree * tree, sfixn p)
// bounds[0,1..mm], bounds[0] is not used
// dims is the size of E. dim[0] is not used, 
// dim[i] is # of elements on dimension#i of E.
// E has dimension N(poly). so dim array has N(poly)+1 slots.
// the bounds for C level is the buffer size.
// At Maple level maybe the degrees, which has 1 less.
// M is how many dimensions or variables we want to evaluate.
void fastEvalMulti_test(sfixn M, sfixn *dims, sfixn *E, preFFTRep* poly, 
    subProdTree** trees, MONTP_OPT2_AS_GENE * pPtr)
{ 
    sfixn i, j, tmp, m=0, n=1, N=N(poly);
    // the uni-one to be evaluated.
    sfixn degf, *fPtr, NoPts;
    sfixn *accumE;
    sfixn *result, resultsz;
    subProdTree* tmpTree;

    assert(M<=N);
    signal(SIGINT,catch_intr);

    accumE=(sfixn *)my_calloc(N+1, sizeof(sfixn));
    accumE[1]=1;
    for(i=1; i<=N; i++) {n*=dims[i]; m+=dims[i];} 
    for(i=2; i<=N; i++) {accumE[i]=accumE[i-1]*dims[i-1];}
    fromtofftRepMultiD(N, accumE, E, CUM(poly), BUSZS(poly), DAT(poly));
    my_free(accumE);

    resultsz=1<<(trees[1]->h);
    if (resultsz==0) resultsz=dims[1];
    result=(sfixn *)my_calloc(resultsz, sizeof(sfixn));

    for(j=0; j<n; j+=dims[1]){
        if((Interrupted==1)||(PrimeError)) { my_free(result); return; }
        fPtr=E+j;
        degf=shrinkDegUni(dims[1]-1, fPtr);
        cleanVec(resultsz-1, result);
        FastEvaluation_1(resultsz, result, degf, fPtr, trees[1], pPtr->P);
        NoPts=(trees[1]->NoNodes)[1];
        copyVec_0_to_d(NoPts - 1, fPtr, result);
    }  
    my_free(result);

    for(i=2;i<=M;i++){
        multi_mat_transpose (N, n, i, dims, E);
        resultsz=1<<(trees[i]->h);
        if(resultsz==0)  resultsz=dims[i];
        result=(sfixn *)my_calloc(resultsz, sizeof(sfixn));
        for(j=0; j<n; j+=dims[i]){
            if((Interrupted==1)||(PrimeError==1)) { my_free(result); return; }
            fPtr=E+j;
            degf=shrinkDegUni(dims[i]-1, fPtr);
            cleanVec(resultsz-1, result);
            FastEvaluation_1(resultsz, result, degf, fPtr, trees[i], pPtr->P);
            NoPts=(trees[i]->NoNodes)[1];
            copyVec_0_to_d(NoPts - 1, fPtr, result);
        }
        my_free(result);
        tmp=dims[1]; dims[1]=dims[i]; dims[i]=tmp;
        tmpTree=trees[1]; trees[1]=trees[i]; trees[i]=tmpTree;
    }
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //  different from multi-DFT, here we transpose the
    //  the data back into the origianl order.
    //  to void subsequent confusion.
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    for(i=M;i>=2;i--){
        multi_mat_transpose (N, n, i, dims, E);
        tmp=dims[1]; dims[1]=dims[i]; dims[i]=tmp; 
        tmpTree=trees[1]; trees[1]=trees[i]; trees[i]=tmpTree; 
    }
}

// Us -- evaluating points.
// Sub-product tree technique based fast interpolation.
preFFTRep* fastInterpMulti_test(sfixn N, sfixn M, sfixn *dims, sfixn *EEE, 
    sfixn **UsPtr, subProdTree** trees, MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn i, j, tmp, m=0, n=1;
    // the uni-one to be evaluated.
    sfixn *result, resultsz;
    sfixn polyDg, *polyDgAddr = &polyDg;
    sfixn *dgs, *accumE;
    preFFTRep *poly;
    subProdTree* tmpTree;
    sfixn *tmpUs;
    sfixn *E;

    assert(M<=N);
    signal(SIGINT, catch_intr);

    for(i = 1; i <= N; i++) { n *= dims[i]; m += dims[i]; } 
    dgs = (sfixn *)my_calloc(N + 1, sizeof(sfixn));

    result = (sfixn *)my_calloc(dims[1], sizeof(sfixn));
    resultsz = dims[1];

    E = (sfixn *)my_calloc(n, sizeof(sfixn));
    copyVec_0_to_d(n - 1, E, EEE);

    for(j = 0; j < n; j += dims[1]) {
        cleanVec(resultsz-1, result);   
        fastInterp_1(polyDgAddr, result, dims[1], UsPtr[1], trees[1], E + j, pPtr->P);
        cleanVec(dims[1] - 1, E + j);
        copyVec_0_to_d(polyDg, E + j, result);
        if (polyDg > dgs[1]) dgs[1] = polyDg;
    }
    my_free(result);

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //  different from multi-DFT, 
    //  the input data for interplation is not transposed.
    //  They are already tansposed back into the origianl
    //  order, so here, we just use the same order and
    //  transpose as in fast evaluation.
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    for (i = 2; i <= M; i++) {
        multi_mat_transpose(N, n, i, dims, E);
        result = (sfixn *)my_calloc(dims[i], sizeof(sfixn)); 
        if ((Interrupted==1)||(PrimeError==1)) { my_free(result); return NULL; }

        resultsz = dims[i];
        for (j = 0; j < n; j += dims[i]) {
            cleanVec(resultsz-1, result);
            fastInterp_1(polyDgAddr, result, dims[i], UsPtr[i], trees[i], E + j, pPtr->P);
            cleanVec(dims[i] - 1, E + j);
            copyVec_0_to_d(polyDg, E + j, result);
            if (polyDg > dgs[i]) dgs[i] = polyDg;
        }
        my_free(result);
        // multi_mat_transpose didn't change dims accordings, 
        // so we need to do this.
        tmp     = dims[1];   dims[1] = dims[i];   dims[i] = tmp; 
        tmpTree = trees[1]; trees[1] = trees[i]; trees[i] = tmpTree; 
        tmpUs   = UsPtr[1]; UsPtr[1] = UsPtr[i]; UsPtr[i] = tmpUs;  
        tmp     = dgs[1];     dgs[1] = dgs[i];     dgs[i] = tmp; 
    }

    for (i = M; i >= 2; i--) {
        multi_mat_transpose(N, n, i, dims, E);
        tmp     = dims[1];   dims[1] = dims[i];   dims[i] = tmp;
        tmpTree = trees[1]; trees[1] = trees[i]; trees[i] = tmpTree;
        tmpUs   = UsPtr[1]; UsPtr[1] = UsPtr[i]; UsPtr[i] = tmpUs;  
        tmp     = dgs[1];     dgs[1] = dgs[i];     dgs[i] = tmp;
    }
    for (i = M + 1; i <= N; i++) dgs[i] = dims[i] - 1;
    accumE = (sfixn *)my_calloc(N + 1, sizeof(sfixn));
    accumE[1] = 1;
    for (i = 2; i <= N; i++) { accumE[i] = accumE[i-1] * dims[i-1]; }
    poly = (preFFTRep *)my_malloc(sizeof(preFFTRep));
    InitOnePoly(poly, N, dgs);
    fromtofftRepMultiD(N, CUM(poly), DAT(poly), accumE, dgs, E);

    my_free(dgs);
    my_free(accumE);
    my_free(E);
    return poly;
}

/**
 * FFT and TFT related methods for construct subresultant chains
 */

// return -1 means, can't do it.
// return 0 means, can do it.
int fillTheRoots(sfixn M, sfixn *rootsPtr, sfixn *es, sfixn *dims, 
    MONTP_OPT2_AS_GENE *pPtr)
{
    int i;
    sfixn *tmprootsPtr = rootsPtr;
    for (i = 1; i <= M;i++){
        if (dims[i] < 2) return -1;
        EX_Mont_GetNthRoots_OPT2_AS_GENE(es[i], dims[i], tmprootsPtr, pPtr);
        tmprootsPtr += dims[i];
    }
    return 0;
}

void cleanTheRoots(sfixn M, sfixn *rootsPtr, sfixn *dims) {
    int i;
    sfixn *tmprootsPtr=rootsPtr;
    for (i = 1;i <= M; i++){
        cleanVec(dims[i] - 1, tmprootsPtr);
        tmprootsPtr += dims[i];
    }
}

/**
 * To find primitive roots of unity, in the sense that the initials of P and Q
 * do not vanish at all of its powers.
 *
 * @f1, the first multivariate polynomial 
 * @f2, the second multivariate polynomial
 * @d1, the actual degree in the largest variable
 * @d2, the actual degree in the largest variable
 * @es, the exponent vector of the degree bounds, of length N
 * @dims, the degree bound vector of f1 and f2, of length N
 * @rootsPtr, the primitive roots of unity, of size dims[1] + ... + dims[N-1]
 * @return, -1 if and only if it fails
 *
 * We assume that d1 >= d2 and
 *
 *   es[0] = dims[0] = dims[0] = 0, 2^es[i] = dims[i] for i from 1 to N - 1
 *
 * For example, 
 *
 * f1 = (1 + 2 x) + (3 + 4 x) y + (5 + 6 x) y^2 + (7 + 8 x) y^3 
 * f2 = (6 + 5 x) + (4 + 3 x) y + (2 + 1 x) y^2 
 *
 * We have N = 2, d1 = 3, d2 = 2, es = [0, 3], and dims = [0, 8]. 
 */
int findGoodRoots(sfixn N, preFFTRep *f1, preFFTRep *f2, sfixn *es, sfixn *dims,
    sfixn *rootsPtr, MONTP_OPT2_AS_GENE *pPtr)
{
    preFFTRep *h1 = EX_getInitial(f1);  // initial of h1
    preFFTRep *h2 = EX_getInitial(f2);  // initial of h2
    sfixn coEsz = 1, *coE;              // place to evaluate initials    
    sfixn const1 = constantPolyp(h1);   // check if h1 is constant
    sfixn const2 = constantPolyp(h2);   // check if h2 is constant
    sfixn nogood1 = 1, nogood2 = 1;     // roots not found initially
    sfixn i, goodroots, times = 1;

    // if both h1 and h2 are nonzero constant, then any roots will do
    if (const1 && const2) {
        goodroots = fillTheRoots(N - 1, rootsPtr, es, dims, pPtr);
        if (goodroots == -1) {
            PrimeError = 1;
            EX_FreeOnePoly(h1);
            EX_FreeOnePoly(h2);
        }
        return goodroots;
    } 

    // one of initials is non-constant, we try to find primitive roots of 
    // unity such that both h1 and h2 do not vanish at powers of those roots.
    for(i = 1; i <= N - 1; i++) { coEsz *= dims[i]; }
    coE = (sfixn *)my_calloc(coEsz, sizeof(sfixn));

    while (nogood1 && nogood2) {
        cleanTheRoots(N - 1, rootsPtr, dims); 
        goodroots = fillTheRoots(N - 1, rootsPtr, es, dims, pPtr);
        if (goodroots == -1){
            PrimeError = 1;
            my_free(coE);
            EX_FreeOnePoly(h1);
            EX_FreeOnePoly(h2);
            return -1;
        }

        // check if h1 vanishes at some point
        cleanVec(coEsz - 1, coE);
        fastDftMulti_test(N - 1, es, dims, coE, h1, rootsPtr, pPtr);
        nogood1 = isVecContainsZero(coEsz, coE);
        if (nogood1) {
            times++;
            if (times > 10) {
                my_free(coE);
                EX_FreeOnePoly(h1);
                EX_FreeOnePoly(h2);
                return -1;
            }
            break;
        }

        // check if h2 vanishes at some point
        cleanVec(coEsz - 1, coE);
        fastDftMulti_test(N - 1, es, dims, coE, h2, rootsPtr, pPtr);
        nogood2 = isVecContainsZero(coEsz, coE);
        if (nogood2) {
            times++;
            if (times > 10) {
                my_free(coE);
                EX_FreeOnePoly(h1);
                EX_FreeOnePoly(h2);
                return -1;
            }
        }
    }

    my_free(coE);
    EX_FreeOnePoly(h1);
    EX_FreeOnePoly(h2);
    return times;
}

// To find a n-th primitive root of unity, in the sense that the initials of 
// f1 and f2 do not vanish at all of its powers.
int createGoodRootsForf1f2(sfixn N, sfixn M, sfixn d1, 
    preFFTRep *f1, sfixn d2, preFFTRep *f2, sfixn *es, 
    sfixn *dims, sfixn *rootsPtr, MONTP_OPT2_AS_GENE *pPtr)
{
    int times = 1, i, nogood1 = 1, nogood2 = 1;
    sfixn n, const1, const2, goodroots, *coE, coEsz;
    
    backupData(f1);
    decreaseOneDim(f1); 
    nextMCoefData(f1, N, d1);
    backupData(f2);
    decreaseOneDim(f2); 
    nextMCoefData(f2, N, d2); 

    const1 = constantPolyp(f1);
    const2 = constantPolyp(f2);
    
    if (!(const1 && const2)) {
        // n=N-1 in this case.
        n = N(f1);
        coEsz = 1;
        for(i = 1; i <= n; i++) { coEsz *= dims[i]; }
        coE = (sfixn *)my_calloc(coEsz, sizeof(sfixn));

        while(nogood1 || nogood2){
            cleanTheRoots(M, rootsPtr, dims); 
            goodroots=fillTheRoots(M, rootsPtr, es, dims, pPtr);
            if (goodroots == -1){
                PrimeError = 1;
                my_free(coE);
                increaseOneDim(f1);
                restoreData(f1);
                increaseOneDim(f2);
                restoreData(f2);
                return -1;
            }

            cleanVec(coEsz-1, coE);
            fastDftMulti_test(M, es, dims, coE, f1, rootsPtr, pPtr);
            nogood1=isVecContainsZero(coEsz, coE);
            cleanVec(coEsz-1, coE);
            fastDftMulti_test(M, es, dims, coE, f2, rootsPtr, pPtr);
            nogood2=isVecContainsZero(coEsz, coE);

            if(nogood1 || nogood2){
                times++;
                if(times>10){ break; }
            }
        }
        my_free(coE);
    } else {
        goodroots = fillTheRoots(M, rootsPtr, es, dims, pPtr);
        if (goodroots == -1) {
            PrimeError = 1;
            increaseOneDim(f1);
            restoreData(f1);
            increaseOneDim(f2);
            restoreData(f2);
            return -1;
        }
    }
    increaseOneDim(f1);
    restoreData(f1);
    increaseOneDim(f2);
    restoreData(f2);
    return times;
}

// DFT based fast evaluation.
void fastDftMulti_test(sfixn M, sfixn *es, sfixn *dims, sfixn *E, 
    preFFTRep* poly, sfixn *rootsPtr, MONTP_OPT2_AS_GENE * pPtr)
{ 
    sfixn i, j, tmp, n = 1, N = N(poly);
    sfixn * tmprootsPtr;
    sfixn *fPtr;
    sfixn *accumE;
    
    assert(M <= N);
    signal(SIGINT, catch_intr);
    accumE = (sfixn *)my_calloc(N + 1, sizeof(sfixn));
    accumE[1]=1;
    for(i = 1; i <= N; i++) { n *= dims[i]; } 
    for(i = 2; i <= N; i++) { accumE[i] = accumE[i-1] * dims[i-1]; }
    fromtofftRepMultiD(N, accumE, E, CUM(poly), BUSZS(poly), DAT(poly));
    my_free(accumE);
    tmprootsPtr = rootsPtr;
    
    // univariate FFTs on the rows
    for(j = 0; j < n; j += dims[1]) {
        fPtr = E + j;
        EX_Mont_DFT_OPT2_AS_GENE_1(dims[1], es[1], tmprootsPtr, fPtr, pPtr);
    }
      
    for (i = 2;i <= M; i++){
        if ((Interrupted==1) || (PrimeError==1)) { return; }
        tmprootsPtr += dims[1];
        multi_mat_transpose (N, n, i, dims, E);
        // univariate FFTs on the columns
        for(j = 0; j < n; j += dims[i]) {
            fPtr = E + j;
            EX_Mont_DFT_OPT2_AS_GENE_1(dims[i], es[i], tmprootsPtr, fPtr, pPtr);
        }
        tmp = dims[1]; dims[1] = dims[i]; dims[i] = tmp; 
        tmp = es[1]; es[1] = es[i]; es[i] = tmp;
    }

    // recovering the orginal data layout
    for (i = M; i >= 2; i--) {
        multi_mat_transpose(N, n, i, dims, E);
        tmp = dims[1]; dims[1] = dims[i]; dims[i] = tmp;
        tmp = es[1]; es[1] = es[i]; es[i] = tmp;
    }
}

/***
 *  @N, the number of variables
 *  @M, the number of variables evaluated, M <= N
 *  @es, the exponent vector, of size M + 1 
 *  @dims, the size vector, of size N + 1  
 *  @EEE, the evaluations, of size dims[1] * .. * dims[N]
 *
 *  Example:
 *
 *  N = 3 
 *  M = 2
 *  es = [0, 2, 2]
 *  dims = [0, 4, 4, 2]
 * 
 *  Where F = f0(X, Y) + f1(X, Y) * Z, with deg(F, x) <= 3 and deg(F, y) <= 3.
 *
 *  DFT(F) is E = [DFT(f0), DFT(f1)], which can be expanded as
 *
 *  f0(1,    1)  f0(wx^2,     1)  f0(wx,     1)  f0(wx^3,     1)
 *  f0(1, wy^2)  f0(wx^2,  wy^2)  f0(wx,  wy^2)  f0(wx^3,  wy^2)
 *  f0(1,   wy)  f0(wx^2,    wy)  f0(wx,    wy)  f0(wx^3,    wy)
 *  f0(1, wy^3)  f0(wx^2,  wy^3)  f0(wx,  wy^3)  f0(wx^3,  wy^3)
 *  f1(1,    1)  f1(wx^2,     1)  f1(wx,     1)  f1(wx^3,     1)
 *  f1(1, wy^2)  f1(wx^2,  wy^2)  f1(wx,  wy^2)  f1(wx^3,  wy^2)
 *  f1(1,   wy)  f1(wx^2,    wy)  f1(wx,    wy)  f1(wx^3,    wy)
 *  f1(1, wy^3)  f1(wx^2,  wy^3)  f1(wx,  wy^3)  f1(wx^3,  wy^3)
 *
 *  In this function, we interpolate F from DFT(F) as follows:
 *
 *  (1) run inverse univariate fft on the rows obtaining
 *  
 *  a0  a1  a2  a3  
 *  b0  b1  b2  b3  
 *  c0  c1  c2  c3  
 *  d0  d1  d2  d3  
 *  e0  e1  e2  e3  
 *  k0  k1  k2  k3  
 *  g0  g1  g2  g3  
 *  h0  h1  h2  h3  
 *
 *  where
 *
 *  f0(X, 1)    = a0 + a1 * X + a2 * X^2 + a3 * X^3 
 *  f0(X, wy^2) = b0 + b1 * X + b2 * X^2 + b3 * X^3 
 *  f0(X, wy)   = c0 + c1 * X + c2 * X^2 + c3 * X^3 
 *  f0(X, wy^3) = d0 + d1 * X + d2 * X^2 + d3 * X^3 
 *  f1(X, 1)    = e0 + e1 * X + e2 * X^2 + e3 * X^3 
 *  k1(X, wy^2) = k0 + k1 * X + k2 * X^2 + k3 * X^3 
 *  f1(X, wy)   = g0 + g1 * X + g2 * X^2 + g3 * X^3 
 *  f1(X, wy^3) = h0 + h1 * X + h2 * X^2 + h3 * X^3 
 *
 * (2) "transpose" the data
 *
 *  a0  b0  c0  d0    
 *  e0  k0  g0  h0 
 *  a1  b1  c1  d1  
 *  e1  k1  g1  h1 
 *  a2  b2  c2  d2  
 *  e2  k2  g2  h2  
 *  a3  b3  c3  d3  
 *  e3  k3  g3  h3
 *
 * (3) run inverse fft along the rows again obtaining
 *
 *  s0 s1 s2 s3  
 *  t0 t1 t2 t3  
 *  u0 u1 u2 u3  
 *  v0 v1 v2 v3  
 *  w0 w1 w2 w3  
 *  x0 x1 x2 x3  
 *  y0 y1 y2 y3  
 *  z0 z1 z2 z3  
 *
 *  where 
 *  
 *  f0(X, Y) = (s0 + s1 * Y + s2 * Y^2 + s3 * Y^3)
 *           + (t0 + t1 * Y + t2 * Y^2 + t3 * Y^3) * X
 *           + (u0 + u1 * Y + u2 * Y^2 + u3 * Y^3) * X^2
 *           + (v0 + v1 * Y + v2 * Y^2 + v3 * Y^3) * X^3 
 *
 *  f1(X, Y) = (w0 + w1 * Y + w2 * Y^2 + w3 * Y^3)
 *           + (x0 + x1 * Y + x2 * Y^2 + x3 * Y^3) * X
 *           + (y0 + y1 * Y + y2 * Y^2 + y3 * Y^3) * X^2
 *           + (z0 + z1 * Y + z2 * Y^2 + z3 * Y^3) * X^3 
 * 
 * (4) "transpose" data back
 *
 *  s0  t0  u0  v0  w0  x0  y0  z0  
 *  s1  t1  u1  v1  w1  x1  y1  z1
 *  s2  t2  u2  v2  w2  x2  y2  z2
 *  s3  t3  u3  v3  w3  x3  y3  z3
 *
 *  we interpolate back F(X, Y, Z) as
 *
 *  F(X, Y, Z) = 
 *    ( (s0 + t0 * X + u0 * X^2 + v0 * X^3) 
 *    + (w0 + x0 * X + y0 * X^2 + z0 * X^3) * Y 
 *    + (s1 + t1 * X + u1 * X^2 + v1 * X^3) * Y^2
 *    + (w1 + x1 * X + y1 * X^2 + z1 * X^3) * Y^3 )
 *  + ( (s2 + t2 * X + u2 * X^2 + v2 * X^3)  
 *    + (w2 + x2 * X + y2 * X^2 + z2 * X^3) * Y 
 *    + (s3 + t3 * X + u3 * X^2 + v3 * X^3) * Y^2
 *    + (w3 + x3 * X + y3 * X^2 + z3 * X^3) * Y^3 ) * Z
 */
preFFTRep* fastInvDftMulti_test(sfixn N, sfixn M, sfixn *es, sfixn *dims, 
    sfixn *EEE, sfixn *rootsPtr, MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn i, j, tmp, n = 1, polyDg;
    sfixn *tmprootsPtr, *dgs, *accumE, *E;
    preFFTRep *poly;

    assert(M <= N);
    signal(SIGINT, catch_intr);

    // n is the size of EEE = dims[1] * ... * dims[N]
    for (i = 1; i <= N; i++) { n *= dims[i]; } 
    E = (sfixn *)my_calloc(n, sizeof(sfixn));

    dgs = (sfixn *)my_calloc(N + 1, sizeof(sfixn));
    copyVec_0_to_d(n - 1, E, EEE);
    tmprootsPtr = rootsPtr;

    // run inverse along the rows and keep the partial degree
    for(j = 0; j < n; j += dims[1]) {
        EX_Mont_INVDFT_OPT2_AS_GENE_1(dims[1], es[1], tmprootsPtr, E + j, pPtr);
        polyDg = shrinkDegUni(dims[1] - 1, E + j);
        if (polyDg > dgs[1]) dgs[1] = polyDg;
    }

    for (i = 2; i <= M; i++) {
        if ((Interrupted == 1) || (PrimeError == 1)) { break; }
        tmprootsPtr += dims[1];
        // transpose the data
        multi_mat_transpose(N, n, i, dims, E);
        for(j = 0; j < n; j += dims[i]){
      	    EX_Mont_INVDFT_OPT2_AS_GENE_1(dims[i], es[i], tmprootsPtr, E + j, pPtr);
            polyDg = shrinkDegUni(dims[i] - 1, E + j);
            if (polyDg > dgs[i]) dgs[i] = polyDg;
        }
        // multi_mat_transpose didn't change dims accordings, 
        // so we need to do this.
        tmp = dims[1]; dims[1] = dims[i]; dims[i] = tmp;
        tmp = es[1];     es[1] = es[i];     es[i] = tmp;
        tmp = dgs[1];   dgs[1] = dgs[i];   dgs[i] = tmp;
    }
    // again, as in the fast evluation,
    // we transform them back to GOOD OLD order.
    for (i = M; i >= 2; i--){
        multi_mat_transpose(N, n, i, dims, E);
        tmp = dims[1]; dims[1] = dims[i]; dims[i] = tmp;
        tmp = es[1];     es[1] = es[i];     es[i] = tmp;
        tmp = dgs[1];   dgs[1] = dgs[i];   dgs[i] = tmp;
    }

    for (i = M + 1; i <= N; i++) dgs[i] = dims[i] - 1;
    accumE = (sfixn *)my_calloc(N + 1, sizeof(sfixn));
    accumE[1] = 1;
    for (i = 2; i <= N; i++) { accumE[i] = accumE[i-1] * dims[i-1]; }
    poly = (preFFTRep *)my_malloc(sizeof(preFFTRep));
    InitOnePoly(poly, N, dgs);
    fromtofftRepMultiD(N, CUM(poly), DAT(poly), accumE, dgs, E);
    my_free(dgs);
    my_free(accumE);
    my_free(E);
    return poly;
}

// Get the quotient images...
// X_N is the main variable.
// dd is the difference of d1, d2.
// S keeps the image of quotients.
// opt=0, using UniQuo.
// opt=1, using PseuQuo.
void getQuotients(sfixn N, sfixn dd, sfixn Qsz, sfixn *Q, sfixn* Edims1,
    sfixn *E1, sfixn *Edims2, sfixn*E2, MONTP_OPT2_AS_GENE * pPtr, int opt )
{
    int i, tmp1, tmp2, m1 = 0, m2 = 0, n1 = 1, n2 = 1;
    int offsetE1 = 0, offsetE2 = 0, offsetQ = 0;
    sfixn d1, d2, *f1, *f2, dds = dd + 1;
    sfixn *Qdims;

    signal(SIGINT, catch_intr);

    for(i=1; i<=N; i++) {
        n1 *= Edims1[i]; m1 += Edims1[i]; 
        n2 *= Edims2[i]; m2 += Edims2[i];
    }

    // suppose using the same cube size of evaluation.
    multi_mat_transpose (N, n1, N, Edims1, E1);
    multi_mat_transpose (N, n2, N, Edims2, E2);
    tmp1 = Edims1[1]; Edims1[1] = Edims1[N]; Edims1[N] = tmp1;
    tmp2 = Edims2[1]; Edims2[1] = Edims2[N]; Edims2[N] = tmp2;

    for(i=0; i < n1; i += Edims1[1]) {
        if ((Interrupted==1) || (PrimeError==1)) { return; }
        f1 = E1 + offsetE1;
        f2 = E2 + offsetE2;
        d1 = shrinkDegUni(Edims1[1] - 1, f1);
        d2 = shrinkDegUni(Edims2[1] - 1, f2);

        if (opt == 0) {
            UniQuo(dd, Q + offsetQ, d1, f1, d2, f2, pPtr);
        } else {
            UniPseuQuo(dd, Q+offsetQ, d1, f1, d2, f2, pPtr);
        }

        offsetE1 += Edims1[1];
        offsetE2 += Edims2[1];
        offsetQ += dds;
    }

    multi_mat_transpose(N, n1, N, Edims1, E1);
    multi_mat_transpose(N, n2, N, Edims2, E2);
    tmp1 = Edims1[1]; Edims1[1] = Edims1[N]; Edims1[N] = tmp1;
    tmp2 = Edims2[1]; Edims2[1] = Edims2[N]; Edims2[N] = tmp2;

    Qdims = (sfixn *) my_calloc(N + 1, sizeof(sfixn));
    for(i = 1; i < N; i++) { Qdims[i] = Edims1[i]; }
    Qdims[N] = dds;

    // permate the cube to normal ordered cube
    // such that it can be interpolated right away.
    permuteSlice1toN(N, Qsz, Qdims, Q);
    my_free(Qdims);
}

// the output S is permuted from dim#N to dim#1.
// So each time to take images from S to interpolate.
// the data has to be permuated form dim1#1 to dim#N first.
// then call fast interplation.

/**  
 *  Example, three dimension case (N = 3, M = 2):
 *
 *  Given P = p0(x, y) + p1(x, y) * z and Q = q0(x, y) + q1(x, y) * z
 *  in K[x, y][z]. We assume the degree bound in x and y are both 4.
 *  For the DFT implementation, the evaluation grid is the following:
 *
 *  (1,    1), (wx^2,     1), (wx,     1), (wx^3,     1)
 *  (1, wy^2), (wx^2,  wy^2), (wx,  wy^2), (wx^3,  wy^2)
 *  (1,   wy), (wx^2,    wy), (wx,    wy), (wx^3,    wy)
 *  (1, wy^3), (wx^2,  wy^3), (wx,  wy^3), (wx^3,  wy^3)
 * 
 *  by taking the bit-reversal ordering in each direction. 
 *
 *  Hence E1 consists of data (as a 4 x 4 x 2 matrix)
 *
 *  p0(1,    1), p0(wx^2,     1), p0(wx,     1), p0(wx^3,     1)
 *  p0(1, wy^2), p0(wx^2,  wy^2), p0(wx,  wy^2), p0(wx^3,  wy^2)
 *  p0(1,   wy), p0(wx^2,    wy), p0(wx,    wy), p0(wx^3,    wy)
 *  p0(1, wy^3), p0(wx^2,  wy^3), p0(wx,  wy^3), p0(wx^3,  wy^3)
 *  p1(1,    1), p1(wx^2,     1), p1(wx,     1), p1(wx^3,     1)
 *  p1(1, wy^2), p1(wx^2,  wy^2), p1(wx,  wy^2), p1(wx^3,  wy^2)
 *  p1(1,   wy), p1(wx^2,    wy), p1(wx,    wy), p1(wx^3,    wy)
 *  p1(1, wy^3), p1(wx^2,  wy^3), p1(wx,  wy^3), p1(wx^3,  wy^3)
 *
 *  We first transpose the data to (2 x 4 x 4 matrix)
 *
 *  PQ(1,    1), PQ(wx^2,     1), PQ(wx,     1), PQ(wx^3,     1)
 *  PQ(1, wy^2), PQ(wx^2,  wy^2), PQ(wx,  wy^2), PQ(wx^3,  wy^2)
 *  PQ(1,   wy), PQ(wx^2,    wy), PQ(wx,    wy), PQ(wx^3,    wy)
 *  PQ(1, wy^3), PQ(wx^2,  wy^3), PQ(wx,  wy^3), PQ(wx^3,  wy^3)
 * 
 *  where PQ(u, v) = [P(u, v), Q(u, v)].  
 *
 *  Then we run subresultant algorithm on each pair, getting
 *
 *  RES(1,    1), RES(wx^2,     1), RES(wx,     1), RES(wx^3,     1)
 *  RES(1, wy^2), RES(wx^2,  wy^2), RES(wx,  wy^2), RES(wx^3,  wy^2)
 *  RES(1,   wy), RES(wx^2,    wy), RES(wx,    wy), RES(wx^3,    wy)
 *  RES(1, wy^3), RES(wx^2,  wy^3), RES(wx,  wy^3), RES(wx^3,  wy^3)
 * 
 *  where RES(u, v) is the subresultant chain of P(u, v) and Q(u, v) in z.
 *
 *  Then we transform both E1 and E2 back to the orignial layout (useless?)
 *
 *  p0(1,    1), p0(wx^2,     1), p0(wx,     1), p0(wx^3,     1)
 *  p0(1, wy^2), p0(wx^2,  wy^2), p0(wx,  wy^2), p0(wx^3,  wy^2)
 *  p0(1,   wy), p0(wx^2,    wy), p0(wx,    wy), p0(wx^3,    wy)
 *  p0(1, wy^3), p0(wx^2,  wy^3), p0(wx,  wy^3), p0(wx^3,  wy^3)
 *  p1(1,    1), p1(wx^2,     1), p1(wx,     1), p1(wx^3,     1)
 *  p1(1, wy^2), p1(wx^2,  wy^2), p1(wx,  wy^2), p1(wx^3,  wy^2)
 *  p1(1,   wy), p1(wx^2,    wy), p1(wx,    wy), p1(wx^3,    wy)
 *  p1(1, wy^3), p1(wx^2,  wy^3), p1(wx,  wy^3), p1(wx^3,  wy^3)
 *
 *  If we gather the first element in each subresultant chain (Subres[0]),
 *  then we have the evaluation of S0 at the grid:
 *  
 *  S0(1,    1), S0(wx^2,     1), S0(wx,     1), S0(wx^3,     1)
 *  S0(1, wy^2), S0(wx^2,  wy^2), S0(wx,  wy^2), S0(wx^3,  wy^2)
 *  S0(1,   wy), S0(wx^2,    wy), S0(wx,    wy), S0(wx^3,    wy)
 *  S0(1, wy^3), S0(wx^2,  wy^3), S0(wx,  wy^3), S0(wx^3,  wy^3)
 *
 */
void getSubResultantChains(sfixn N, sfixn w, sfixn Ssz, sfixn *S, sfixn* Edims1, 
    sfixn *E1, sfixn *Edims2, sfixn*E2, MONTP_OPT2_AS_GENE * pPtr )
{
    int m1 = 0, m2 = 0, n1 = 1, n2 = 1, offsetE1 = 0, offsetE2 = 0, offsetS = 0;
    int i, tmp1, tmp2;
    sfixn d1, d2, *f1, *f2, ww = w * w;

    signal(SIGINT, catch_intr);

    for(i = 1; i <= N; i++) {
        n1 *= Edims1[i]; m1 += Edims1[i]; 
        n2 *= Edims2[i]; m2 += Edims2[i];
    }

    multi_mat_transpose(N, n1, N, Edims1, E1);
    multi_mat_transpose(N, n2, N, Edims2, E2);
    tmp1 = Edims1[1]; Edims1[1] = Edims1[N]; Edims1[N] = tmp1;
    tmp2 = Edims2[1]; Edims2[1] = Edims2[N]; Edims2[N] = tmp2;

    for (i = 0; i < n1; i += Edims1[1]) {
        if ((Interrupted == 1) || (PrimeError == 1)) { return; }
        f1 = E1 + offsetE1;
        f2 = E2 + offsetE2;
        d1 = shrinkDegUni(Edims1[1] - 1, f1);
        d2 = shrinkDegUni(Edims2[1] - 1, f2);
        SubResultantSeq_1_new(w, ww, S + offsetS, d1, f1, d2, f2, pPtr);
        offsetE1 += Edims1[1];
        offsetE2 += Edims2[1];
        offsetS += ww;
    }

    multi_mat_transpose(N, n1, N, Edims1, E1);
    multi_mat_transpose(N, n2, N, Edims2, E2);
    tmp1 = Edims1[1]; Edims1[1] = Edims1[N]; Edims1[N] = tmp1;
    tmp2 = Edims2[1]; Edims2[1] = Edims2[N]; Edims2[N] = tmp2;
}

/**
 * function to transpose data matrices.
 */
void permuteSlice1toN(sfixn N, sfixn slicesz, sfixn *slicedims, sfixn *slice) {

    sfixn tmp;
    // A[w][...] -> A[...][w]
    tmp = slicedims[1]; slicedims[1] = slicedims[N]; slicedims[N] = tmp;
    multi_mat_transpose(N, slicesz, N, slicedims, slice);
    // A[...][w] -> A[w][...]
    tmp = slicedims[1]; slicedims[1] = slicedims[N]; slicedims[N] = tmp;
}

/**
 * @ith, index from 0 to w-1
 * @N  , the number of variables  
 * @w  , the smaller main degree
 * @Ssz, the size of S
 * @S  , data in the SCube
 * @slicesz, the size of the slice to be returned.
 */
sfixn *get_ithSlice_fromSubResultantChains(sfixn ith, sfixn N, sfixn w, 
    sfixn slicesz, sfixn *slicedims, sfixn Ssz, sfixn *S)
{
    sfixn ww = w*w, no = Ssz/ww;
    sfixn *slice, offset1=0, offset2=ith*w;
    int i;
    // we only have 0-th .. (w-1)-th slices in each subresultant sequence.
    assert(ith<w);
    slice = (sfixn *)my_calloc(slicesz, sizeof(sfixn));
    for (i=0; i<no; i++) {
        copyVec_0_to_d(w-1, slice+offset1, S+offset2);
        offset1 += w;
        offset2 += ww;
    }
    permuteSlice1toN(N, slicesz, slicedims, slice);
    return slice;
}

void set_ithSliceZero_fromSubResultantChains(sfixn ith, sfixn N, sfixn w, 
    sfixn slicesz, sfixn *slicedims, sfixn Ssz, sfixn *S) 
{
    sfixn ww = w*w, no = Ssz/ww;
    sfixn offset2 = ith*w;
    int i;
    assert(ith<w);
    for(i=0; i<no; i++){
        cleanVec(w-1, S+offset2);
        offset2 += ww;
    }
}

/**
 * A subresultant with ListRectangle layout is the following piece of data
 *
 *  a00    0    0    0   ===> S[00]   b00    0    0    0   ===> S[10]
 *  a10  a11    0    0   ===> S[01]   b10  b11    0    0   ===> S[11]
 *  a20  a21  a22    0   ===> S[02]   b20  b21  b22    0   ===> S[12]
 *  a30  a31  a32  a33   ===> S[03]   b30  b31  b32  b33   ===> S[13]
 *
 * @ith, row index
 * @dth, column index
 *
 * If ith = 0 and dth = 0, returns [a00, b00].
 * If ith = 2 and dth = 1, returns [a21, b21].
 */

// size of returned buffer is of size Ssz/(w*w)
sfixn *get_ithDthSubSlice_fromSubResultantChains(sfixn ith, sfixn dth, sfixn N,
    sfixn w, sfixn subslicesz, sfixn *subslicedims, sfixn Ssz, sfixn *S) 
{
    int i;
    sfixn ww = w * w, no = Ssz / ww;
    sfixn *subslice, offset2 = ith * w + dth;
    assert(ith < w && dth < w);

    subslice = (sfixn *)my_calloc(subslicesz, sizeof(sfixn));
    for (i = 0; i < no; i++) {
        subslice[i] = S[offset2];
        offset2 += ww;
    }
    permuteSlice1toN(N, subslicesz, subslicedims, subslice);
    return subslice;
}


// return sth>=w, mean didn't find the defective resultant,
// in between the starting point and previous non-defective resultant.
// return  0 < sth < w, we found the defective resultant
// before previous non-defective one.
int tracingNextCandidateSliceDefective(int start, sfixn w, sfixn Ssz, sfixn *S)
{
    // no is the no. of squences.
    sfixn ww=w*w, no=Ssz/ww;
    int i, j=0, offseti=0, offsetj=w*(start+1);

    if (start<0) return 0;
    for (j=start+1; j<w; j++) {

        // If before find the defective resultant,
        // we found the non-defective one, then w+1, means
        // there is no defective resultant found before 
        // previous non-defective one.
        offseti = 0;
        for (i=0; i<no; i++) {
            if (S[offseti+offsetj+j] != 0) return w+1;
            offseti += ww;
        }

        // we found the defective one return it.
        offseti = 0;
        for (i=0; i<no; i++) {
            if (S[offseti+offsetj+start] != 0) return j;
            offseti += ww;
        }
        offsetj += w;
    }
    return j;
}

int tracingNextCandidateSlice(int start, sfixn w, sfixn Ssz, sfixn *S)
{
    // no is the no. of squences.
    sfixn ww=w*w, no=Ssz/ww;
    int i, j=0, offseti=0, offsetj=w*(start+1);

    if (start<0) return 0;
    for (j=start+1; j<w; j++) {
        offseti = 0;
        for(i=0; i<no; i++){
            if (S[offseti+offsetj+j] != 0) return j;
            offseti += ww;
        }
        offsetj += w;
    }
    return j;
}

void printAllSRS(sfixn no, sfixn w, sfixn *AllSRS){
    int i, offset=0;
    sfixn ww=w*w;
    for(i=0; i<no; i++) { 
        printSRS(w, AllSRS+offset);
        offset+=ww;
    }
}

// interpolate a polynomial fromt the sub-resultant chain.
preFFTRep* interpIthSlice(sfixn ith, sfixn N, sfixn m, sfixn w, sfixn slicesz, 
    sfixn *slicedims, sfixn Ssz, sfixn *S,  PTS_TREE* pts_tree, 
    MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn *slice;
    preFFTRep *poly;

    slice = get_ithSlice_fromSubResultantChains(ith, N, w, 
        slicesz, slicedims, Ssz, S);

    poly = fastInterpMulti_test(N, m, slicedims, slice, pts_tree->ptsPtr, 
        pts_tree->trees, pPtr);

    my_free(slice);
    return poly;
}

// interpolate a coefficient in the sub-resultant chain.
preFFTRep* interpIthDthSlice(sfixn ith, sfixn dth, sfixn N, sfixn m, sfixn w, 
    sfixn subslicesz, sfixn *subslicedims, sfixn Ssz, sfixn *S, 
    PTS_TREE* pts_tree, MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn *subslice;
    preFFTRep *poly;
    
    subslice = get_ithDthSubSlice_fromSubResultantChains(ith, dth, N, w, 
        subslicesz, subslicedims, Ssz, S);

    poly = fastInterpMulti_test(N, m, subslicedims, subslice, 
        pts_tree->ptsPtr, pts_tree->trees, pPtr);

    // Memory leak, BUG, to fix.
    my_free(subslice);
    return poly;
}

preFFTRep* interpNextCandidateSliceLCDefective(int *nextiAddr, int start, 
    sfixn N, sfixn m, sfixn w, sfixn subslicesz, sfixn *subslicedims, 
    sfixn Ssz, sfixn *S, PTS_TREE* pts_tree, MONTP_OPT2_AS_GENE * pPtr)
{
    preFFTRep* poly;
    (*nextiAddr)=tracingNextCandidateSliceDefective(start, w, Ssz, S);
    if ((*nextiAddr)>=w) return NULL;
    poly = interpIthDthSlice((*nextiAddr), start, N, m, w, 
        subslicesz, subslicedims, Ssz, S, pts_tree, pPtr);

    return poly;
}

// see the regular gcd algorithm.
preFFTRep* interpNextCandidateSliceLC(int *nextiAddr, int start, sfixn N, 
    sfixn m, sfixn w, sfixn subslicesz, sfixn *subslicedims, 
    sfixn Ssz, sfixn *S, PTS_TREE* pts_tree, MONTP_OPT2_AS_GENE * pPtr) 
{
    preFFTRep* poly;
    // printf("start=%d, w=%d.\n", start, w);
    (*nextiAddr)=tracingNextCandidateSlice(start, w, Ssz, S);
    // printf("(*nextiAddr)=%d.\n", (*nextiAddr));
    if((*nextiAddr)>=w) return NULL;

    poly=interpIthDthSlice((*nextiAddr), (*nextiAddr), N, m, w, 
        subslicesz, subslicedims, Ssz, S, pts_tree, pPtr);

    return poly;
}


// interpolate a polynomial in the sub-resultant chain.
preFFTRep* interpIthSliceDFT(sfixn ith, sfixn N, sfixn m, sfixn w, 
    sfixn slicesz, sfixn *slicees, sfixn *slicedims, 
    sfixn Ssz, sfixn *S, sfixn *rootsPtr, MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn *slice;
    preFFTRep *poly;
    slice = get_ithSlice_fromSubResultantChains(ith, N, w, slicesz, slicedims, Ssz, S);
    poly = fastInvDftMulti_test(N, m, slicees, slicedims, slice, rootsPtr,pPtr);
    my_free(slice);
    return poly;
}

preFFTRep* interpIthDthSliceDFT(sfixn ith, sfixn dth, sfixn N, sfixn m, 
    sfixn w, sfixn subslicesz, sfixn *subslicees, sfixn *subslicedims, 
    sfixn Ssz, sfixn *S, sfixn *rootsPtr, MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn *subslice;
    preFFTRep *poly;

    subslice = get_ithDthSubSlice_fromSubResultantChains(ith, dth, N, 
        w, subslicesz, subslicedims, Ssz, S);

    poly = fastInvDftMulti_test(N, m, subslicees, subslicedims, 
        subslice, rootsPtr, pPtr);
    
    my_free(subslice);
    return poly;
}

preFFTRep* interpNextCandidateSliceLCDFTDefective(int *nextiAddr, int start, 
    sfixn N, sfixn m, sfixn w, sfixn subslicesz, sfixn *subslicees, 
    sfixn *subslicedims, sfixn Ssz, sfixn *S, sfixn *rootsPtr, 
    MONTP_OPT2_AS_GENE * pPtr)
{
    preFFTRep* poly;
    (*nextiAddr)=tracingNextCandidateSliceDefective(start, w, Ssz, S);
    if((*nextiAddr)>=w) return NULL;

    poly=interpIthDthSliceDFT((*nextiAddr), start, N, m, w, subslicesz, 
        subslicees, subslicedims, Ssz, S, rootsPtr, pPtr);

    return poly;
}

preFFTRep* interpNextCandidateSliceLCDFT(int *nextiAddr, int start, sfixn N, 
    sfixn m, sfixn w, sfixn subslicesz, sfixn *subslicees, sfixn *subslicedims,
    sfixn Ssz, sfixn *S, sfixn *rootsPtr, MONTP_OPT2_AS_GENE * pPtr)
{
    preFFTRep* poly;
    (*nextiAddr) = tracingNextCandidateSlice(start, w, Ssz, S);
    if ((*nextiAddr)>=w) return NULL;
    poly = interpIthDthSliceDFT((*nextiAddr), (*nextiAddr), N, m, w, 
        subslicesz, subslicees, subslicedims, Ssz, S, rootsPtr, pPtr);
    return poly;
}

// opt=0 -> using Euclidean division.
// opt=1 -> using Pseudo division.
/**
 * EX_QuoMulti:
 * @f1: A C-Cube polynomial.
 * @f2: A C-Cube polynomial.
 * @N: the number of variables.
 * @pPtr: the information of the prime number.
 * @opt: set 0 to use Euclidean division, set 1 to use Pseduo division.
 *
 * To compute the quotient of 'f11' divided by 'f22' via evaluation approach.
 *  
 * Return value: the quotient of 'f11' divided by 'f22'.
 **/
preFFTRep * EX_QuoMulti(preFFTRep *f11, preFFTRep *f22, sfixn N, 
    MONTP_OPT2_AS_GENE * pPtr, int opt)
{
    int i, bool1, bool2;
    sfixn degA, degB, degQ, *Q, M, *degs, co;
    preFFTRep *resPoly, *tmpPoly, *f1, *f2;
    PTS_TREE *pts_tree;
    sfixn Qsz, *Qdat, *Qdims, d1, d2, dd, *bounds; 
    sfixn *dims1, *dims2, Esz1, *E1, Esz2, *E2;

    assert(N(f11) == N);
    assert(N(f22) == N);
    M = N-1;
    if (EX_IsEqualPoly(f11, f22)) { return CreateConsPoly(1); }

    degs = (sfixn *)my_calloc(N + 1, sizeof(sfixn));
    for (i = 1; i < N; i++) { degs[i]=1; }
    tmpPoly = EX_InitOnePoly(N, degs);

    co = 0;
    for(i = 1; i < N; i++){ co += CUMI(tmpPoly, i); }
    DATI(tmpPoly, co)=1;
  
    my_free(degs);
    f1 = EX_EX_mulPoly(N, tmpPoly, f11, pPtr);
    f2 = EX_EX_mulPoly(N, tmpPoly, f22, pPtr);
    EX_FreeOnePoly(tmpPoly);
    
    if (N == 1){
        degA = shrinkDegUni(BUSZSI(f1, 1), DAT(f1));
        degB = shrinkDegUni(BUSZSI(f2, 1), DAT(f2));
        if (degA < degB) return CreateZeroPoly();

        if (opt == 0) {
            Q = EX_UniQuo(&degQ, degA, DAT(f1), degB, DAT(f2), pPtr);
        } else {
            Q = EX_PQuo_Uni(&degQ, degA, DAT(f1), degB, DAT(f2), pPtr);
        }
        resPoly = CreateUniPoly(degQ, Q);
        my_free(Q);
    } else {
        bool1 = IsAllNumberCoeffs(f1);
        bool2 = IsAllNumberCoeffs(f2);
        if (bool1 && bool2){
            return QuoAsUni(f1, f2, pPtr);
        }
        d1 = shrinkDeg(BUSZSI(f1, N), DAT(f1), CUMI(f1, N));
        d2 = shrinkDeg(BUSZSI(f2, N), DAT(f2), CUMI(f2, N));
        if (d1 < d2) return CreateZeroPoly();
        dd = d1 - d2;
        bounds = (sfixn *)my_calloc(M+1, sizeof(sfixn));
        dims1  = (sfixn *)my_calloc(N+1, sizeof(sfixn));
        dims2  = (sfixn *)my_calloc(N+1, sizeof(sfixn));
        Qdims  = (sfixn *)my_calloc(N+1, sizeof(sfixn));
        dims1[N] = BUSZSI(f1, N) + 1;
        dims2[N] = BUSZSI(f2, N) + 1;
      
        for(i=1; i<=M; i++){
	        bounds[i]= BUSZSI(f1, i) + dd * BUSZSI(f2, i) + 1;
            dims1[i] = bounds[i];
            dims2[i] = bounds[i];
        }
        pts_tree = createGoodPtsForf1f2(N, d1, f1, d2, f2, M, bounds, 
            dims1, dims2, pPtr);

        for (i=1, Esz1 = 1; i <= N; i++) { Esz1 *= dims1[i]; }
        for (i=1, Esz2 = 1; i <= N; i++) { Esz2 *= dims2[i]; }

        E1 = (sfixn *)my_calloc(Esz1, sizeof(sfixn));
        E2 = (sfixn *)my_calloc(Esz2, sizeof(sfixn));
        fastEvalMulti_test(M, dims1, E1, f1, pts_tree->trees, pPtr);
        fastEvalMulti_test(M, dims2, E2, f2, pts_tree->trees, pPtr);

        Qsz = 1;
        for(i = 1; i <= M; i++) { Qdims[i] = bounds[i]; Qsz = Qsz * bounds[i]; }
        Qdims[N] = dd + 1;
        Qsz = Qsz * Qdims[N];
        Qdat = (sfixn *)my_calloc(Qsz, sizeof(sfixn));

        getQuotients(N, dd, Qsz, Qdat, dims1, E1, dims2, E2, pPtr, opt);
        my_free(E1); my_free(E2);
        resPoly = fastInterpMulti_test(N, M, Qdims, Qdat, pts_tree->ptsPtr, 
		     pts_tree->trees, pPtr);

        my_free(bounds); my_free(dims1); my_free(dims2); 
        my_free(Qdims);  my_free(Qdat);
        freePTS_TREE(pts_tree);
    }

    EX_FreeOnePoly(f1);
    EX_FreeOnePoly(f2);
    return resPoly;
}

/**
 * TFT based fast evaluation
 *
 * @M    : the number of variables to be evaluated
 * @es   : Fourier degree in each dimension
 * @ls   : the sizes in each dimension, ls[i] is the TFT size in dimension i, i=1..M
 *         ls[M+i] is one plus the partial degree in the (M+i)-th variable, for 1 <= i <= N-M
 *         ls[i] should be at least one plus the partial degree in the i-th variable.
 * @E    : (Output) the evaluation values of size Esz.
 * @poly : The polynomial to be evaluated.
 *
 * @rootsPtr : an array of primitive roots of unity
 *
 *             1, w1, w1^2, ..., w1^{2^es[1]-1},
 *             1, w2, w2^2, ..., w2^{2^es[2]-1}},
 *             ...
 *             1, wM, wM^2, ..., wM^{2^es[M]-1}
 *
 *             In total, the size of rootsPtr is 2^es[1] + ... + 2^es[M]
 *
 * Note :
 *
 * (1) M <= N, the number of variables in poly
 *
 * (2) es and ls will NOT be tranposed by this procedure. 
 *     And rootsPtr remains unchanged.
 *
 * (3) Esz is ls[1] * ... * ls[M] * ls[M+1] * ... * ls[N]
 *
 */
/**
 * Example :
 *
 * M = 2
 * dgs = {0, 1, 2, 2}
 * ls = {0, 2, 3, 3} //size in each dimension, ls[3] is dgs[3]+1
 * es = {0, 1, 2}
 *
 * Assume that poly = ((0 + 1*a) + (2 + 3*a)*b + (4 + 5*a)*b^2)
 *                    +
 *                    ((6 + 7*a) + (8 + 9*a)*b + (10 + 11*a)*b^2)*c
 *                    +
 *                    ((12 + 13*a) + (14 + 15*a)*b + (16 + 17*a)*b^2)*c^2
 *
 * Then E will be filled by three TFT evaluations. To be more precise,
 *
 * Write poly = F0 + F1*c + F2*c^2 and
 *
 * rootsPtr consists of two lists of roots  W = [1, w] and  U = [1, u, u^2, u^3]
 *
 * Then F0 will given by a vector
 *
 *          F0(1, 1), F0(1, u^2), F0(1, u), F0(w, 1), F0(w, u^2), F0(w, u)
 *
 * Similarly, F1 will given by a vector
 *
 *          F1(1, 1), F1(1, u^2), F1(1, u), F1(w, 1), F1(w, u^2), F1(w, u)
 *
 * F2 will given by a vector
 *
 *          F2(1, 1), F2(1, u^2), F2(1, u), F2(w, 1), F2(w, u^2), F2(w, u)
 *
 * Concatenating the above three vectors into EPtr gives the result, which 
 * represents a univariate polynomial F in c of degree 2.
 *
 * Bit reversal ordering is used for each coefficient.
 *
 * */

void TFTEvalMultiD(sfixn M, sfixn *es, sfixn *ls, sfixn *EPtr, preFFTRep *poly,
    sfixn *rootsPtr, MONTP_OPT2_AS_GENE *pPtr)
{

    sfixn i, m, N = N(poly);
    sfixn Esz;
    sfixn *exp_accum, *dims;

    assert(M<=N);
    signal(SIGINT, catch_intr);

    // exp_accum is similar to poly->accum, but it is computed according to ls.
    // exp_accum[0] = 0;
    // exp_accum[1] = 1;
    // exp_accum[2] = exp_accum[1] * ls[1] 
    // ...
    // exp_accum[i] = exp_accum[i-1] * ls[i-1] 
    // ...
    // exp_accum[M] = exp_accum[M-1] * ls[M-1] 
    // ...
    // exp_accum[N] = exp_accum[N-1] * ls[N-1]
    // exp_accum[N+1] = exp_accum[N] * ls[N]
    exp_accum = (sfixn *)my_calloc(N+2, sizeof(sfixn));
    exp_accum[1] = 1;
    for (i=2; i<=N+1; ++i) { 
        exp_accum[i] = exp_accum[i-1] * ls[i-1]; 
    }
    // m = ls[1]*ls[2]*...*ls[M] is the size of each coefficient of this polynomial.
    // Esz = m * ls[M+1] * ... ls[N] is the size of the polynomial.
    Esz = exp_accum[N+1]; 
    m = exp_accum[M+1];

    // Expand all the coefficients into E, since ls usually gives up-bounds only.
    fromtofftRepMultiD(N, exp_accum, EPtr, poly->accum, poly->cuts, poly->data);
    
    my_free(exp_accum);
    //printf("Initial polynomial : ");
    //printVec(SIZ(poly)-1, DAT(poly));
    //printf("E : ");
    //printVec(Esz-1, EPtr);
    // Do TFT for each coefficient
    
    dims = (sfixn *)my_calloc(M+1, sizeof(sfixn));
    for (i=1; i<=M; ++i) dims[i] = ((sfixn)1<<es[i]);
    
    for (i=0; i<Esz/m; ++i) {
        TFTMultiD(EPtr+m*i, M, es, dims, ls, rootsPtr, pPtr);
    }
    //printf("Evaluated polynomial: ");
    //printVec(Esz-1, EPtr);
    my_free(dims);
}

/**
 * TFT based fast Interplotation
 *
 * @M    : the number of variables to be interpolated
 * @es   : 2^es[i] = dims[i] for i = 1..M
 * @ls   : the sizes in each dimension, ls[i] is the TFT size in dimension i, i=1..M
 *         ls[i] is one plus the partial degree in the i-th variable, M+1<=i<=N.
 * @E    : (Input/Output) the evaluation values of size Esz
 * @poly : The polynomial to be interpolated.
 *
 * @rootsPtr : an array of primitive roots of unity
 *
 *             1, w1, w1^2, ..., w1^{2^es[1]-1},
 *             1, w2, w2^2, ..., w2^{2^es[2]-1}},
 *             ...
 *             1, wM, wM^2, ..., wM^{2^es[M]-1}
 *
 *             In total, the size of rootsPtr is 2^es[1] + ... + 2^es[M]
 *
 * Note :
 *
 * (1) M <= N, the number of variables in poly
 *
 * (2) Esz is ls[1] * ... * ls[M] * ls[M+1] * ... * ls[N]
 *
 **/
/**
 * Example :
 *
 * M = 2
 * dgs = {0, 1, 2, 2}
 * ls = {0, 2, 3, 3} //size in each dimension, ls[3] is dgs[3]+1
 * es = {0, 1, 2}
 *
 * Write F = F0 + F1*c + F2*c^2 and
 *
 * rootsPtr consists of two lists of roots  W = [1, w] and  U = [1, u, u^2, u^3]
 *
 * Then F0 will given by a vector
 *
 *          F0(1, 1), F0(1, u^2), F0(1, u), F0(w, 1), F0(w, u^2), F0(w, u)
 *
 * Similarly, F1 will given by a vector
 *
 *          F1(1, 1), F1(1, u^2), F1(1, u), F1(w, 1), F1(w, u^2), F1(w, u)
 *
 * F2 will given by a vector
 *
 *          F2(1, 1), F2(1, u^2), F2(1, u), F2(w, 1), F2(w, u^2), F2(w, u)
 *
 * Concatenating the above three vectors into EPtr gives the result, 
 * which represents a univariate polynomial F in c of degree 2.
 *
 * Initially E holds the values F0(1, 1), ... F2(w, u), which represents 
 * a univariate polynomial.
 *
 * This function interploates a multivariate polynomial F, 
 * which TFT-evaluates to E on rootsPtr.
 *
 **/
preFFTRep* TFTInterpMultiD(sfixn N, sfixn M, sfixn *es, sfixn *ls, sfixn *E,
    sfixn *rootsPtr, MONTP_OPT2_AS_GENE *pPtr)
{
    preFFTRep *poly;
    sfixn *dgs, *dims;
    sfixn *EE, Esz; 
    sfixn i, m = 1;

    assert(M<=N);
    signal(SIGINT, catch_intr);

    // m = ls[1]*ls[2]*...*ls[M] is the size of each coefficient of polynomial.
    // Esz = m * ls[M+1] * ... ls[N] is the size of the polynomial.
    for (i=1; i<=M; ++i) { m *= ls[i]; }
    Esz = m;
    for (i=M+1; i<=N; ++i) { Esz *= ls[i]; }

    EE = (sfixn *)my_calloc(Esz, sizeof(sfixn));
    // Copy the coefficients to EE.
    copyVec_0_to_d(Esz-1, EE, E);

    // printf("\nEE : ");
    // printVec(Esz-1, EE);
    // Interpolating all the coefficients
    dims = (sfixn *)my_calloc(M+1, sizeof(sfixn));
    for (i=1; i<=M; ++i) dims[i] = ((sfixn)1 << es[i]);
    for (i=0; i<Esz/m; ++i) {
        TFTInvMultiD(EE+m*i, M, es, dims, ls, rootsPtr, pPtr);
    }
    dgs = (sfixn *)my_calloc(N+1, sizeof(sfixn));
    // partial degree in the evaluated variables
    for (i=1; i<=N; ++i) dgs[i] = ls[i] - 1;
    
    // TODO : decision to use shrinkDegUni or shrinkDeg!!

    poly = EX_InitOnePoly(N, dgs);
    copyVec_0_to_d(Esz-1, DAT(poly), EE);

    my_free(dgs);
    my_free(dims);
    my_free(EE);

    return poly;
}

// the output S is permuted from dim#N to dim#1.
// So each time to take images from S to interpolate.
// the data has to be permuated form dim1#1 to dim#N first.
// then call fast interplation.
void getSubResultantChainsTFT(sfixn N, sfixn M, sfixn w, sfixn Ssz, sfixn *S,
     sfixn *ls1, sfixn *E1, sfixn *ls2, sfixn *E2, MONTP_OPT2_AS_GENE *pPtr)
{
    int i, tmp1, tmp2, n1=1, n2=1, offsetE1=0, offsetE2=0, offsetS=0;
    sfixn d1, d2, *f1, *f2, ww=w*w;
    signal(SIGINT, catch_intr);

    for (i=1; i<=N; i++) {
        n1 *= ls1[i]; 
        n2 *= ls2[i]; 
    }

    multi_mat_transpose(N, n1, N, ls1, E1);
    multi_mat_transpose(N, n2, N, ls2, E2);
    tmp1 = ls1[1]; ls1[1] = ls1[N]; ls1[N] = tmp1;
    tmp2 = ls2[1]; ls2[1] = ls2[N]; ls2[N] = tmp2;

    for (i=0; i<n1; i+=ls1[1]) {
        if ((Interrupted==1)||(PrimeError==1)) { return; }

        f1 = E1 + offsetE1;
        f2 = E2 + offsetE2;
        d1 = shrinkDegUni(ls1[1]-1, f1);
        d2 = shrinkDegUni(ls2[1]-1, f2);

        SubResultantSeq_1_new(w, ww, S+offsetS, d1, f1, d2, f2, pPtr);

        offsetE1 += ls1[1];
        offsetE2 += ls2[1];
        offsetS += ww;
    }
    multi_mat_transpose(N, n1, N, ls1, E1);
    multi_mat_transpose(N, n2, N, ls2, E2);
    tmp1 = ls1[1]; ls1[1] = ls1[N]; ls1[N] = tmp1;
    tmp2 = ls2[1]; ls2[1] = ls2[N]; ls2[N] = tmp2;
}

// TFT based methods to interpolate a polynomial from an TFT-SCube
preFFTRep* interpIthSliceTFT(sfixn ith, sfixn N, sfixn m, sfixn w, sfixn slicesz, 
    sfixn *slicees, sfixn *slicedims, sfixn Ssz, sfixn *S, 
    sfixn *rootsPtr, MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn *slice;
    preFFTRep *poly;
    slice = get_ithSlice_fromSubResultantChains(ith, N, w, slicesz, 
        slicedims, Ssz, S);
    poly = TFTInterpMultiD(N, m, slicees, slicedims, slice, rootsPtr, pPtr);
    my_free(slice);
    return poly;
}

preFFTRep* interpIthDthSliceTFT(sfixn ith, sfixn dth, sfixn N, sfixn m, 
    sfixn w, sfixn subslicesz, sfixn *subslicees, sfixn *subslicedims, 
    sfixn Ssz, sfixn *S,  sfixn *rootsPtr, MONTP_OPT2_AS_GENE * pPtr)
{
    sfixn *subslice;
    preFFTRep *poly;
    subslice = get_ithDthSubSlice_fromSubResultantChains(ith, dth, N, w, 
        subslicesz, subslicedims, Ssz, S);
    poly = TFTInterpMultiD(N, m, subslicees, subslicedims, subslice, rootsPtr, pPtr);
    my_free(subslice);
    return poly;
}

preFFTRep* interpNextCandidateSliceLCTFTDefective(int *nextiAddr, int start, 
    sfixn N, sfixn m, sfixn w, sfixn subslicesz, sfixn *subslicees, 
    sfixn *subslicedims, sfixn Ssz, sfixn *S, sfixn *rootsPtr, 
    MONTP_OPT2_AS_GENE * pPtr) 
{
    preFFTRep* poly;

    //assert(start>=0);
    //assert(start<w);

    (*nextiAddr) = tracingNextCandidateSliceDefective(start, w, Ssz, S);
    if ((*nextiAddr)>=w) return NULL;
    poly = interpIthDthSliceTFT( (*nextiAddr), start, N, m, w, subslicesz, 
        subslicees, subslicedims, Ssz, S, rootsPtr, pPtr);

    return poly;
}

preFFTRep* interpNextCandidateSliceLCTFT(int *nextiAddr, int start, sfixn N, 
    sfixn m, sfixn w, sfixn subslicesz, sfixn *subslicees, sfixn *subslicedims, 
    sfixn Ssz, sfixn *S, sfixn *rootsPtr, MONTP_OPT2_AS_GENE * pPtr)
{
    preFFTRep* poly;
    (*nextiAddr) = tracingNextCandidateSlice(start, w, Ssz, S);
    if ((*nextiAddr)>=w) return NULL;
    poly = interpIthDthSliceTFT((*nextiAddr), (*nextiAddr), N, m, w, subslicesz,
        subslicees, subslicedims, Ssz, S, rootsPtr, pPtr);
    return poly;
}

////////////////////////////////////////////////////////////////////////////////

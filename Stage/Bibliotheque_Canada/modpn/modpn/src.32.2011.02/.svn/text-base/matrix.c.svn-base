/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#include "matrix.h"




//=====================================================
// 2-D Matrix Transposition.
//=====================================================

// Input: A matrix Mat with n columns, m rows.
// Output: transposed Mat.
/**
 * mat_transpose:
 * @n: number of rows of M.
 * @m: number of columns of M.
 * @data: data field of A 2-D matrix M.
 *
 * Transpose a 2-D matrix in-place.
 * 
 * Return value: 
 **/
/*
void 
mat_transpose (sfixn n, sfixn m, sfixn * data){
  int i,j;
  sfixn tmpary[n][m];
  sfixn * tmpPtr;

  for(i=0;i<m;i++){
    tmpPtr=data+n*i;
    for(j=0;j<n;j++){
      tmpary[j][i]=tmpPtr[j];
    }
  }

  for(i=0;i<n;i++){
    tmpPtr=data+m*i;
    for(j=0;j<m;j++){
      tmpPtr[j]=tmpary[i][j];
     }
  }  

}

*/




//=====================================================
// N-D Matrix Transpose.
//=====================================================


// Transpose data in the Mat from dimension N to dimension dm.
/**
 * decompose:
 * @N: 'N' is the number of dimensions of Mat.
 * @dm: A index for the dm-th dimension.
 * @dims: The dimensions vector. dims[i] keep the size on dimension-i (dims[0] is not used).
 * @data: The data field.
 * 
 * We transpose matrix M from dimension 'dm' to 1. 
 * Return value: 
 **/

void
decompose(sfixn N, sfixn dm, sfixn * accum, sfixn * dims, sfixn * tmpPtr, sfixn * data){
  register int j;
  int i,ii,jj,ac, ldimsN, ldims1, iildimsN, ildims1;
  if (N==dm){
     ac=accum[N+1]/dims[1];
     ldims1=0;
     for(ldimsN=0; ldimsN<ac; ldimsN+=dims[N] ){
       for(i=0, ii=0; i<dims[1]; i++, ii+=ac){
	    iildimsN=ii+ldimsN;
            ildims1=i+ldims1;
            for(j=0, jj=0; j<dims[dm]; j++, jj+=accum[dm]){
	       tmpPtr[j+iildimsN]=data[jj+ildims1];
	    }
       }
       ldims1+=dims[1];
     }
    return;
  }

  for(i=0; i<accum[N+1]; i+=accum[N]){
    decompose(N-1, dm, accum, dims, tmpPtr+i, data+i);
  }

} 



// dims[] from lowest dimension to highest dimension.
// N is the number of dimensions of data.
// so the size of dims[] is N+1.
// we want to transpose data on dm to dim--1.
// the rest size-2 dimension keep no change.
// output: transpose data, and updated dims accordingly.
// dims[0] is useless.
// dims[1] keeps the # of vectors in dimemsion#1;
// dims array  should to be changed corresponding to the matrix tranposition, 
// BUT not in this functions. I did it in MultiD-FFT/TFT after calling this function.


/**
 * multi_mat_transpose:
 * @N: 'N' is the number of dimensions of Matrix M.
 * @n: The number of entries in M.
 * @dm: The dm-th dimension.
 * @dims: The dimensions vector. dims[i] keep the size on dimension-i (dims[0] is not used).
 * @data: The data field.
 * 
 * We transpose matrix M from dimension 'dm' to 1. 
 * Return value: 
 **/
void 
multi_mat_transpose (sfixn N, sfixn n, sfixn dm, sfixn * dims, sfixn * data){
  register int i;
  sfixn * accum, * tmpPtr;


  accum=(sfixn *) my_calloc(N+2, sizeof(sfixn));

  // accum[1] keeps dim-1's interval which is 1.
  // accum[2] keeps dim-2's interval which is #dim-1;
  // accum[3] keeps dim-3's inverval whcih is #dim-1*#dim-2
  accum[1]=1;
  for(i=2; i<=N+1; i++) accum[i]=dims[i-1]*accum[i-1];
  //n=accum[N]*dims[N];
  tmpPtr=(sfixn * ) my_calloc(n, sizeof(sfixn));

  decompose(N, dm, accum, dims, tmpPtr, data);

  for(i=0; i<n; i++) data[i]=tmpPtr[i]; 

  my_free(accum);
  my_free(tmpPtr);
}





//=============================================================
// To generate a vector of random polynomials in 
// Straight Line Program Encoding.
//=============================================================
// M slots, each SLG with GN nodes, prime=p, N is # of vars.
/**
 * randomPolyVec_SLG:
 * @M: int .
 * @GN: Number of node in a dag.
 * @p: A prime.
 * @N: Number of variables.
 * To generate a vector of random polynomials in 
 *  Straight Line Program Encoding (DAG). The output vector has size 'M'
 *  Each DAG has 'GN' nodes and represents a polynomial has 'N' variables.
 * All coefficients are over Z/pZ
 * Return value: Then Random vector of GAGs.
 **/
POLYVECTOR_SLG *
randomPolyVec_SLG(int M, int GN, sfixn p, int N){
  SLG *slg;
  int i;
  POLYVECTOR_SLG *vec=(POLYVECTOR_SLG *)my_malloc(sizeof(POLYVECTOR_SLG));
  vec->M = M;
  vec->entries = (SLG **)my_malloc(M*sizeof(SLG *));
  for(i=0; i<M; i++){     
     slg=randomSLG(GN, p, N);
     //printf("GN=%d\n", GN);
     ENTRYI_V(vec, i)=removRedun(slg);
     freeSLG(slg);
  }
  return vec;
}




POLYVECTOR_SLG *
example_1_PolyVec_SLG(){
  SLG *slg;
  POLYVECTOR_SLG *vec=(POLYVECTOR_SLG *)my_malloc(sizeof(POLYVECTOR_SLG));
  vec->M = 2;
  vec->entries = (SLG **)my_malloc(2*sizeof(SLG *));
 
  slg=generateSLG_example_1_F1();
  // printf("SLG-1:\n");
  //printSLG_Line(slg);

  ENTRYI_V(vec, 0)=removRedun(slg);

  // printf("SLG-1:\n");
  //printSLG_Line(slg);

  freeSLG(slg);

  slg=generateSLG_example_1_F2();
  //printf("SLG-2:\n");
  //printSLG_Line(slg);
  ENTRYI_V(vec, 1)=removRedun(slg);
  //printf("SLG-2:\n");
  //printSLG_Line(slg);
  freeSLG(slg);
 
  return vec;
}






//=============================================================
// To free the vector of polynomials in 
// Straight Line Program Encoding.
//=============================================================
/**
 * freeVec_SLG:
 * @vec_slg: A vector of DAGs.
 * 
 * Free the DAG vector 'vec_slg'.
 * Return value: 
 **/

void
freeVec_SLG(POLYVECTOR_SLG *vec_slg){
  int i;
  if(vec_slg){
    if(vec_slg->entries){
      for(i=0; i<vec_slg->M; i++) freeSLG((vec_slg->entries)[i]);
      my_free(vec_slg->entries);
    }
    my_free(vec_slg);
  }
}




//============================================================
// To print the matrix of polynomials.
//============================================================

/**
 * printJMatrix:
 * @mat: A Jacobean matrix.
 *
 * Print the Jacobean matrix 'mat'.
 * 
 * Return value: 
 **/
void
printJMatrix(POLYMATRIX * mat){
  int i, j;
  printf("{\n");
  printf("M = %d\n", mat->M);
  printf("N = %d\n", mat->N);
  for(i=0; i<mat->M; i++){
    for(j=0; j<mat->N; j++){
      printf("mat[%d][%d]=", i, j);
      printPoly(ENTRYI_M(mat, i, j));
    }
  }
  printf("}\n");
}



//============================================================
// To free the matrix of polynomials.
//============================================================
/**
 * freeJMatrix:
 * @mat: A C-Cube polynomial matrix.
 * 
 * Free the polynomial matrix 'mat'.
 *
 * Return value: 
 **/
void freeJMatrix(POLYMATRIX * mat){
  int i, MN;
  if(mat){
    MN=(mat->M)*(mat->N);
    if(mat->entries){
       for(i=0; i<MN; i++){
         freePoly((mat->entries)[i]);
         my_free((mat->entries)[i]);
       }
       my_free(mat->entries);
    }
    my_free(mat);
  }
}




// ===================================================================
// To convert a vector of SLG polynomials into
// an one-row matrix of dense polynomials.
// ===================================================================
/**
 * SLG2PolyVecotr:
 * @polyVec_SLG: A vector of DAD polynomials.
 * @ts: A triangular set.
 * @tris: The inverses (reverse-ordered) of 'ts'.
 * @N: Number of variables.
 * @pPtr: Information of the prime.
 *  To convert a vector of DAG polynomials into
 *  an one-row matrix of C-Cube polynomials.
 * Return value: The one-row matrix of C-Cube polynomial 
 **/
POLYMATRIX *
SLG2PolyVecotr(POLYVECTOR_SLG *polyVec_SLG, TriSet *ts, TriRevInvSet * tris, sfixn N, MONTP_OPT2_AS_GENE * pPtr){
  int j;
  POLYMATRIX *mat = (POLYMATRIX *)my_malloc(sizeof(POLYMATRIX));
  mat->M = polyVec_SLG->M;
  mat->N = 1;
  mat->entries = (preFFTRep **)my_malloc((mat->N)*(mat->M)*(sizeof(preFFTRep *)));

  for(j=0; j<mat->M; j++){
    //printf("j=%ld.\n", j);
    //fflush(stdout);
       ENTRYI_M(mat, j, 0) = SLG2POLY(ENTRYI_V(polyVec_SLG ,j) , ts, tris, N, pPtr);
   }

  return mat;
}







// ===================================================================
//create a JMatrix (POLYMATRIX) for a given SLG polynomial vector modulo a TS.
// ===================================================================
/**
 * createJMatrix:
 * @polyVec_SLG: A vector of DAG polynomials.
 * @ts: A triangular set.
 * @tris: The inverses (reverse-ordered) of 'ts'.
 * @N: Number of variables. 
 * @pPtr: The information of the prime.
 *
 * Create a Jacobean matrix for the input system 'polyVec_SLG' modulo a the input Triangular set 'ts'.
 *
 * Return value:  The newly created Jacobean matrix.
 **/
POLYMATRIX *
createJMatrix(POLYVECTOR_SLG *polyVec_SLG, TriSet *ts, TriRevInvSet * tris, sfixn N, MONTP_OPT2_AS_GENE * pPtr){
  int i, j;
  SLG *tmpslg;
  POLYMATRIX *mat = (POLYMATRIX *)my_malloc(sizeof(POLYMATRIX));
  operand *vars=(operand *)my_malloc(N*sizeof(operand));
  for(i=0; i<N; i++){
     new_var_ini(vars[i], i+1);
  }
  mat->M = polyVec_SLG->M;
  mat->N = N;
  mat->entries = (preFFTRep **)my_malloc((mat->N)*(mat->M)*(sizeof(preFFTRep *)));
  for(i=0; i<mat->M; i++){
     for(j=0; j<mat->N; j++){
       tmpslg = getDerivOfG_OLD0(ENTRYI_V(polyVec_SLG ,i), vars[j]);
       //printf("-- -1.0 -- \n");
       //fflush(stdout);
       ENTRYI_M(mat, i, j) = SLG2POLY(tmpslg , ts, tris, N, pPtr);
       //printf("-- -1.1 -- \n");
       //fflush(stdout);
       freeSLG(tmpslg);
     }
  }

  for(i=0; i<N; i++){
     my_free(vars[i]);
  }
  my_free(vars);
  return mat;
}







// ===================================================================
//create a JMatrix(POLYMATRIX) for a given SLG polynomial vector modulo a TS.
// N = #(t,x1,x2,...,xn)
// ===================================================================
/**
 * createJMatrix_ForLifting:
 * @polyVec_SLG: The input system, a vector DAG polynomials.
 * @ts: A triangular set.
 * @tris: The inverses (reverse ordered) of 'ts'.
 * @N: The number of variables.
 * @pPtr: The information of the prime.
 *
 *
 * Return value: The Jocobean matrix. 
 **/
POLYMATRIX *
createJMatrix_ForLifting(POLYVECTOR_SLG *polyVec_SLG, TriSet *ts, TriRevInvSet * tris, sfixn N, MONTP_OPT2_AS_GENE * pPtr){
  int i, j;
  operand *roots=(operand *)my_malloc((N-1)*(N-1)*sizeof(operand));
  POLYMATRIX *mat = (POLYMATRIX *)my_malloc(sizeof(POLYMATRIX));
  operand *vars=(operand *)my_malloc((N-1)*sizeof(operand));
  //SLG **tmpslgAry = (SLG **)my_malloc((N-1)*sizeof(SLG *));
  SLG * tmpslg;
  // create variables for doing derivative.
  for(i=0; i<N-1; i++){ new_var_ini(vars[i], i+1+1);}
  // create a shall space for "mat".
  mat->M = polyVec_SLG->M;
  mat->N = N-1;
  mat->entries = (preFFTRep **)my_malloc((mat->N)*(mat->M)*(sizeof(preFFTRep *)));
  
  tmpslg=createWholeJMatrix_For_Lifting_Hashing(roots, i, mat, polyVec_SLG, vars, ts, tris, N, pPtr);
  //tmpslg=createWholeJMatrix_For_Lifting(roots, i, mat, polyVec_SLG, vars, ts, tris, N, pPtr);

for(i=0; i<mat->M; i++){
    //for(j=0; j<mat->N; j++){tmpslgAry[j]=NULL;}


      
  //printf("after lifting.\n");
  //  fflush(stdout);
      for(j=0; j<mat->N; j++){
	ENTRYI_M(mat, i, j) = SLG2POLY_ROOT(roots[i*(mat->N)+j], tmpslg , ts, tris, N, pPtr); 
      }   
   
      //printf("after POLY_ROOT.\n");
      //printSLG_Line(tmpslg);
      //fflush(stdout); 

  }

  freeSLG(tmpslg);

  
  for(i=0; i<N-1; i++){ my_free(vars[i]);}
  my_free(vars);
  my_free(roots);
  return mat;
}



//=====================================================
// To initialize a JMatrix(POLYMATRIX)
// m row, n column.
// N is the # of variable.
//=====================================================


/**
 * initJMatrix:
 * @N: The number of variables.
 * @m: The number of rows.
 * @n: The number of columns.
 * @ts: The triangular set.
 *
 * Initialize a 'm' by 'n' Jacobean matrix, where the entries are 
 * clean (all coefficients are zeros) C-Cube polynomials
 * modulo 'ts'. 
 *
 * Initialize a Jacobean matrix.
 *
 * Return value: 
 **/
POLYMATRIX *
initJMatrix(sfixn N, int m, int n, TriSet *ts){
  int i, j;
  POLYMATRIX *mat = (POLYMATRIX *)my_malloc(sizeof(POLYMATRIX));
  mat->M = m;
  mat->N = n;
  mat->entries = (preFFTRep **)my_malloc(m*n*(sizeof(preFFTRep *)));
  //printf("N=%d, m=%d, n=%d.\n", N, m, n);
  //fflush(stdout);
  for(i=0; i<m; i++){
     for(j=0; j<n; j++){
       //printf("be:i=%d, j=%d.\n", i, j);
       //fflush(stdout);  
       ENTRYI_M(mat, i, j)=(preFFTRep *)my_malloc(sizeof(preFFTRep));
       //printf("mi:i=%d, j=%d.\n", i, j);
       //fflush(stdout);      
       InitOnePoly(ENTRYI_M(mat, i, j), N, ts->bounds);
       //printf("af:i=%d, j=%d.\n", i, j);
       //fflush(stdout);     
     }
  }

  return mat;
}





//=====================================================
//  Increase the data space of a dense polynomial to 
//  accomodate newly lifted variables.
//=====================================================
POLYMATRIX *
increaseMatrix_ForLifting(sfixn N, POLYMATRIX *mat, TriSet *ts){
  POLYMATRIX * newmat;
    int i, j;
  newmat=initJMatrix(N, mat->M, mat->N, ts);

   for(i=0; i<mat->M; i++){

     for(j=0; j<mat->N; j++){
       fromtofftRep(N, CUM(ENTRYI_M(newmat,i,j)), DAT(ENTRYI_M(newmat,i,j)), CUM(ENTRYI_M(mat,i,j)), BUSZS(ENTRYI_M(mat,i,j)), DAT(ENTRYI_M(mat,i,j)));
     }
   } 
   freeJMatrix(mat);
   return newmat;
}


//=========================================================
// Create a random JMATRIX(POLYMATRIX).
// m row, n column.
// N is the # of variable.
//=========================================================

/**
 * increaseMatrix_ForLifting:
 * @mat: A Jacobean matrix .
 * @pPtr: The information of the prime.
 * 
 * Filling the input 'mat' with random values.
 *
 * Return value: A Jacobena matrix filled with random coefficients
 **/
POLYMATRIX *
randomJMatrix(POLYMATRIX * mat, MONTP_OPT2_AS_GENE * pPtr){
  int i, j;
  for(i=0; i<mat->M; i++){
     for(j=0; j<mat->N; j++){
       randomPoly(ENTRYI_M(mat, i, j), pPtr->P);
     }
  }

  return mat;
}




//=========================================================
// JMATRIX(POLYMATRIX) Multiplication.
// input: mat1, mat2
// output: the product.
//=========================================================

/**
 * mulJMatrix:
 * @N: the number of variables.
 * @mat1: A jacobean matrix.
 * @mat2: A Jacobean matrix.
 * @ts: A triangular set.
 * @tris: The inverses (reverse ordered) of 'ts'. 
 * @pPtr: The information of the prime.
 *
 * Matrix multiplication of 'mat1' and 'mat2'
 * 
 * Return value: The product of 'mat1' and 'mat2'.
 **/
POLYMATRIX *
mulJMatrix(sfixn N, POLYMATRIX * mat1, POLYMATRIX * mat2, 
           TriSet * ts, TriRevInvSet * tris,  MONTP_OPT2_AS_GENE * pPtr) {
    int i, j, k;

    POLYMATRIX *resMat = initJMatrix(N, mat1->M, mat2->N, ts);
    assert(mat1->N == mat2->M);
	//printf("here--->00\n");
    //printJMatrix(resMat);
    //fflush(stdout);
    for (i=0; i<(mat1->M); i++) {
      for (j=0; j<(mat2->N); j++) {
	for (k=0; k<(mat1->N); k++) {
	  //printf("before:i=%d, j=%d, k=%d.\n",i,j,k);
          //fflush(stdout);
            addMulPoly_1(ENTRYI_M(resMat, i, j), ENTRYI_M(mat1, i, k), 
                         ENTRYI_M(mat2, k, j) , N, ts, tris, pPtr);
	    //printf("after:i=%d, j=%d, k=%d.\n",i,j,k);
	    //fflush(stdout);
	 }
       }
    }
    //printf("here--->01\n");
    //fflush(stdout);
    return(resMat);
}



//=========================================================
// JMATRIX(POLYMATRIX) constant 2 matrix Multiplication.
// 
//=========================================================

/**
 * scalarMulJMatrix_1:
 * @r: A integer in Z/pZ.
 * @mat: A Jacobean matrix.
 * @pPtr: The informaiton of the prime p.
 * 
 * Multiply 'mat'  by r and receive the product in 'mat'.
 *
 * Return value: The modified input 'mat'
 **/
POLYMATRIX *
scalarMulJMatrix_1(sfixn r, POLYMATRIX * mat, MONTP_OPT2_AS_GENE * pPtr) {
    int i, j;
    for (i=0; i<(mat->M); i++) {
      for (j=0; j<(mat->N); j++) {
	MultiNumbPolyMul_1(r, ENTRYI_M(mat, i, j),  pPtr);
       }
    }
    return(mat);
}




//=========================================================
// JMATRIX(POLYMATRIX) subtraction.
// 
//=========================================================

/**
 * subJMatrix_1:
 * @N: The number of variables.
 * @mat1: A Jacobean matrix.
 * @mat2: A Jacobean matrix.
 * @pPtr: The informaiton of the prime p.
 *
 * subtract 'mat2'  from 'mat1', and replace 'mat1' by the difference
 * Return value: 'mat1'='mat1'-'mat2'.
 **/
POLYMATRIX *
subJMatrix_1(sfixn N, POLYMATRIX * mat1, POLYMATRIX * mat2, MONTP_OPT2_AS_GENE * pPtr) {
    int i, j;
    for (i=0; i<(mat1->M); i++) {
      for (j=0; j<(mat1->N); j++) {
	subEqDgPoly_1(N, ENTRYI_M(mat1, i, j), ENTRYI_M(mat2, i, j), pPtr->P, 1);
       }
    }
    return(mat1);
}






//=========================================================
// JMATRIX(POLYMATRIX) inversion.
// 
//=========================================================

// If return NULL, then inversion failed...

/**
 * INVJMatrix:
 * @N: The number of variables.
 * @mat: A C-Cube polynomial matrix.
 * @ts: A triangular set.
 * @tris: The inverses (reverse ordered) of 'ts'.
 * @pPtr: The information of the prime number.
 *
 * Return value: The inverse of 'mat' or NULL if the inverse doesn't exist.
 **/
POLYMATRIX *
INVJMatrix(sfixn N, POLYMATRIX * mat, TriSet * ts, TriRevInvSet * tris,  MONTP_OPT2_AS_GENE * pPtr){
  
   int i, j, k, index;
   int n;
   sfixn recip;
   POLYMATRIX *  tmpMat, * resMat;
   preFFTRep * tmpPtr, * invPtr, * tmpPoly; 
   assert( mat->M == mat->N );
   n=mat->M;
   tmpMat = initJMatrix(N, n, 2*n, ts);
   for(i=0; i<n; i++){
     for(j=0; j<n; j++){
       CopyOnePoly(ENTRYI_M(tmpMat, i, j), ENTRYI_M(mat, i, j));
     }
     setPolyOne(ENTRYI_M(tmpMat, i, n+i));
   }

   invPtr=(preFFTRep *)my_malloc(sizeof(preFFTRep));
   tmpPoly=(preFFTRep *)my_malloc(sizeof(preFFTRep));

   // buffer is increaed for using MultiRecip.
   //InitOnePoly(invPtr, N, BUSZS(ELEMI(ts, N)));
   BDSI(ts, N)++;
   InitOnePoly(invPtr, N, BDS(ts));
   BDSI(ts, N)--;


   InitOnePoly(tmpPoly, N, BDS(ts));
   // =================================>
   
   for(i=0; i<n; i++){

        index = -1;
        for(j=i; j<n; j++){
	  if(! zeroPolyp(ENTRYI_M(tmpMat,j,i))){
            cleanVec(SIZ(invPtr)-1, DAT(invPtr));
            recip=MultiRecip(N, invPtr, ENTRYI_M(tmpMat,j,i), ts, tris, pPtr);

/*             if(recip == 0){ */
/* 	      printf("ENTRYI_M(tmpMat,%d,%d)= is NOT invertible!\n",j,i); */
/*               printPoly(ENTRYI_M(tmpMat,j,i)); */
/*               printf("TriSet=\n"); */
/*               printTriSet(ts); */
/*               fflush(stdout); */

/* 	    } */


	    if (recip != -1){
	        index = j;
                break;
	    }
	  }
	}

        if(index == -1) {
	  printf("Matrix is not invertible since no good pivot  !\n");
          //printf("mat =\n");
	  //printJMatrix(mat);
          //fflush(stdout);
	  return NULL;

	}


	for(j=0; j<(2*n); j++){
	  tmpPtr=ENTRYI_M(tmpMat,i,j);
          ENTRYI_M(tmpMat,i,j)=ENTRYI_M(tmpMat,index,j);
          ENTRYI_M(tmpMat,index,j)=tmpPtr;
	}


        for(j=0; j<2*n; j++){
	  EX_mul_Reduced(N,ENTRYI_M(tmpMat,i,j),ENTRYI_M(tmpMat,i,j),invPtr,ts,tris,pPtr);
	}



        for(k=i+1; k<n; k++){
	  CopyOnePoly(tmpPoly, ENTRYI_M(tmpMat,k,i));
          for(j=0;j<2*n;j++){
                ENTRYI_M(tmpMat,k,j)=subMulPoly_1(ENTRYI_M(tmpMat,k,j), 
                   ENTRYI_M(tmpMat,i,j), tmpPoly, N,ts, tris, pPtr);
	  }
	}


   }



    for(i=n-1;i>=0;i--){
       for(j=0;j<=i-1;j++){
         CopyOnePoly(tmpPoly, ENTRYI_M(tmpMat,j,i)); 
         for(k=0;k<2*n;k++){
              ENTRYI_M(tmpMat,j,k)=subMulPoly_1(ENTRYI_M(tmpMat,j,k), 
                   ENTRYI_M(tmpMat,i,k), tmpPoly, N,ts, tris, pPtr);  
	 }
    }
   }

   // =================================>
   resMat = initJMatrix(N, n, n, ts);
   for(i=0; i<n; i++){
     for(j=0; j<n; j++){
       CopyOnePoly(ENTRYI_M(resMat, i, j), ENTRYI_M(tmpMat, i, n+j));
     }
   }
   freePoly(invPtr);
   my_free(invPtr);
   freePoly(tmpPoly);
   my_free(tmpPoly);
   freeJMatrix(tmpMat);
   return resMat;
   

}





/**
 * createJMatrix_PolyForlifting:
 * @ts: A triangular set..
 * @tris: The inverses (reverse ordered) of 'ts'.
 * @N: The number of variables.
 * @pPtr: The informaiton of the prime. 
 *
 * Return value: 
 **/
POLYMATRIX *
createJMatrix_PolyForlifting(TriSet *ts,  TriRevInvSet * tris, sfixn N, MONTP_OPT2_AS_GENE * pPtr){
  int i, j;
  POLYMATRIX *mat = (POLYMATRIX *)my_malloc(sizeof(POLYMATRIX));
  preFFTRep * tmpPtr;
  mat->M = N-1;
  mat->N = N-1;
  mat->entries = (preFFTRep **)my_malloc((mat->N)*(mat->M)*(sizeof(preFFTRep *)));


  //printf("ts->bounds");
  //printVec(N, ts->bounds);
  //printTriSet(ts);
  //fflush(stdout);

  for(i=0; i<mat->M; i++){
     for(j=0; j<mat->N; j++){
       ENTRYI_M(mat, i, j)=(preFFTRep *)my_malloc(sizeof(preFFTRep));
       InitOnePoly(ENTRYI_M(mat, i, j), N, BDS(ts));
       // ENTRYI_M(mat, i, j) = degDirevOfPoly(ELEMI(ts,i+1+1), j+1+1, ts, tris, i+1+1, pPtr);
       tmpPtr=degDirevOfPoly(ELEMI(ts,i+1+1), (mat->N)-j+1, ts, tris, i+1+1, pPtr);


       fromtofftRep(N(tmpPtr), CUM(ENTRYI_M(mat, i, j)), DAT(ENTRYI_M(mat, i, j)), CUM(tmpPtr), BUSZS(tmpPtr), DAT(tmpPtr));
       freePoly(tmpPtr);
       my_free(tmpPtr);
     }
  }
  return mat;
}





//=========================================================
// see the lift algorithm
// 
//=========================================================

/**
 * AddHs2TriSet:
 * @N: Number of variables.
 * @ts: A triangular set.
 * @Hs: Hs in the algorithm.
 * @Ptr: The information of the prime number. 
 *
 * A inner step of the lifting algorithm
 *
 * Return value: A modified triangular set. 
 **/
TriSet *
AddHs2TriSet(sfixn N, TriSet *ts, POLYMATRIX *Hs, MONTP_OPT2_AS_GENE * pPtr){
  int i;
  preFFTRep * tmpPtr;
  for(i=2; i<=N(ts); i++){
    //printf("ELEMI(ts,%d)=", i);
    //printPoly(ELEMI(ts,i));
    //printf("ENTRYI_M(Hs,%d,0))=", i-2);
    //printPoly(ENTRYI_M(Hs,i-2,0));
  
    //printf("BUSZS(ELEMI(ts,i)): ");
    //printVec(N,BUSZS(ELEMI(ts,i)));
    //printf("BUSZS( ENTRYI_M(Hs,i-2,0)): ");
    //printVec(N,BUSZS(ENTRYI_M(Hs,i-2,0)));

   
    tmpPtr=(preFFTRep *)my_malloc(sizeof(preFFTRep));
    InitOnePoly(tmpPtr, i, BUSZS(ELEMI(ts, i)));
    fromtofftRep(i, CUM(tmpPtr), DAT(tmpPtr), CUM(ENTRYI_M(Hs,i-2,0)),  
                 BUSZS(ENTRYI_M(Hs,i-2,0)), DAT(ENTRYI_M(Hs,i-2,0)));

    addEqDgPoly_1(i, ELEMI(ts,i), tmpPtr, pPtr->P);

    freePoly(tmpPtr);
    my_free(tmpPtr);

  }
  return ts;
}



 

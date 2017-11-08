/* Authors: Xin Li <xli96@csd.uwo.ca>, Marc Moreno Maza <moreno@csd.uwo.ca> */
/* Copyright (c) 2009 by Marc Moreno Maza.  All rights reserved             */
#include "generalFuncs.h"
#include<assert.h>

int Interrupted=0;
int PrimeError=0;



#ifdef WINDOWS


void *my_calloc(size_t size, size_t thesizeof){
  int i;
  sfixn *res;
  size_t alignment=128;  
  size_t totsize=size*thesizeof, rem=totsize%alignment;

  if(totsize ==0) totsize=alignment;
  if (rem != 0) totsize=totsize-rem+alignment;

  res = _aligned_malloc(totsize, alignment);
  
  assert(res!=NULL);
  totsize=totsize/sizeof(sfixn); 
  for(i=0; i<totsize; i++) res[i]=0; 

  return (void*)res;

}




void *my_malloc(size_t totsize){
  int i;
  size_t alignment=128;
  size_t rem=totsize%alignment;
  sfixn *res;


  if(totsize ==0) totsize=alignment;
  if (rem != 0) totsize=totsize-rem+alignment;

  res = _aligned_malloc(totsize, alignment);
  
    assert(res!=NULL);

  return (void*)res;

}

void my_free(void *ptr){
  _aligned_free(ptr);
}


#else
void *my_calloc(size_t nb, size_t size){
  //  size_t alignment=128, totsize=nb*size, rem=totsize%alignment;
  // if(rem!=0) totsize=totsize-rem+alignment;
  // if(totsize==0) totsize=alignment;
  //return calloc(2, totsize);
  size_t totsize=nb*size+16;
  //if(totsize==0) totsize=16;
  return calloc(1, totsize);
}


void *my_malloc(size_t size){
  //size_t alignment=128, rem=size%alignment;
  //if(rem!=0) size=size-rem+alignment;
  //if(size==0) size=alignment;
  //return malloc(2*size);
  size=size+16;
  //if(size==0) size=16;
  return malloc(size);
}


void my_free(void *ptr){
  if(ptr!=NULL) free(ptr);
}

#endif







//==========================================================
// Copy vectors.
//==========================================================

// copy elements 1..n.
/**
 * copyVec_1_to_n:
 * @n:
 * @desV:
 * @srcV: 
 * assignment:
 * desV[1..n] = srcV[1..n]. 
 * Return value: 
 **/
void copyVec_1_to_n(int n, sfixn * desV, sfixn * srcV){
  register int i;
  for(i=1; i<=n; i++) desV[i]=srcV[i];  
}



// copy elements 0..d.
/**
 * copyVec_0_to_d:
 * @n:
 * @desV:
 * @srcV: 
 * assignment:
 * desV[0..d] = srcV[0..d]. 
 * Return value: 
 **/
void copyVec_0_to_d(int d, sfixn * desV, sfixn * srcV){
  register int i;
  for(i=0; i<=d; i++) desV[i]=srcV[i];  
}



// return the new deg.
/**
 * shiftBigger_1:
 * @d: d+1 is the size of 'vec'
 * @vec: a vector
 * @m: 
 * To shift entrys in the input vector 'vec' right (towards higher indices) m slots. 
 * 
 * Return value: the size of new vector minus one. 
 **/
sfixn
shiftBigger_1(int d, sfixn * vec, int m){
  register int i;
  if(d==0 && vec[0]==0) return 0;
  for(i=d; i>=0; i--) vec[i+m]=vec[i];
  for(i=0; i<m; i++) vec[i]=0;
  return d+m;  
}


// output: (d, desV).
/**
 * EX_copyVec_0_to_d:
 * @d: d+1 is the size 'srcV'.
 * @srcV: a vector.
 *   make a copy of input vector 'srcV'.
 * Return value: the new copy
 **/
sfixn*  EX_copyVec_0_to_d(int d, sfixn * srcV){
  sfixn *desV;
  desV=(sfixn *)my_malloc((d+1)*sizeof(sfixn));
  copyVec_0_to_d(d, desV, srcV);
  return desV;
}


// 1 -- yes it's a zero vec.
// 0 -- no it's non-empty vec.
/**
 * isZeroVec:
 * @siz: size of 'vec'.
 * @vec: a vector.
 *   to check if vec is a zero vector.
 * Return value: if 'vec' is a zero vector returns 1, otherwise returns 0.
 **/
int isZeroVec(int siz, sfixn *vec){
  int i;
  for(i=0; i<siz; i++) if(vec[i]) return 0;
  return 1;
}


// 1 -- yes.
// 0 -- no.
/**
 * isZeroPolyUni:
 * @d: d+1 is the size of vec.
 * @vec: a vector. 
 * 
 * Return value: if vec is a zero vector returns 1, otherwise resturn 0. 
 **/
int isZeroPolyUni(sfixn d, sfixn *vec){
  int i;
  for(i=d; i>=0; i--) if(vec[i]) return 0;
  return 1;
}



// 1 -- yes 
// 0 -- no 
/**
 * isVecContainsZero:
 * @siz: size of 'vec'.
 * @vec: a vector.
 *   to check if vec contains a zero ceofficient.
 * Return value: if 'vec' contains a zero coefficient returns 1, otherwise returns 0.
 **/
int isVecContainsZero(int siz, sfixn *vec){
  int i;
  for(i=0; i<siz; i++) if(vec[i]==0) return 1;
  return 0;
}




//==========================================================
// print vectors.
//==========================================================
/**
 * printVec:
 * @deg: deg+1 is the size of vector data.
 * @data: a vector. 
 * Print a vector.
 * Return value: 
 **/
void printVec(sfixn deg, sfixn * data){
  register int i;
  printf("[");
  printf("deg=%ld\n", deg);
  fflush(stdout);
  for(i=0;i<deg;i++) printf("%ld, ", data[i]);
  printf("%ld]\n", data[deg]);
}


/**
 * printVecAndIndex:
 * @deg: deg+1 is the size of vector data.
 * @data: a vector. 
 * Print a vector as index and data pairs.
 * Return value: 
 **/
void printVecAndIndex(sfixn deg, sfixn * data){
  register int i;
  printf("[");
  for(i=0;i<deg;i++) printf("(%d, %ld), ", i, data[i]);
  printf("%ld]\n", data[deg]);
}




void printVecFrom1(sfixn n, sfixn * data){
  register int i;
  printf("[");
  for(i=1;i<n;i++) printf("%ld, ", data[i]);
  printf("%ld]\n", data[n]);
}


// 1 means yes, it is a constant vector.
// 0 means NO, it is NOT a constatn vector.

/**
 * isConstVec:
 * @n: size of vector 'data'.
 * @data: a vector.
 * To check if data[1]=data[2]=...=data[n-1]=0.
 * Return value: if trun returns 1, otherwise returns 0.
 **/
int isConstVec(sfixn n, sfixn * data){
  int i;
  for (i=n-1; i>=1; i--) if(data[i]!=0) return 0;
  return 1;
}



void printVec_1_to_n_double(sfixn n, double * data){
  register int i;
  printf("[");
  for(i=1;i<n;i++) printf("%f, ", data[i]);
  printf("%f]\n", data[n]);
}



/**
 * fprintVec:
 * @F: the handle of a FILE.
 * @deg: deg+1 is the size of vector data.
 * @data: a vector. 
 * Print a vector into a file.
 * Return value: 
 **/
void fprintVec(FILE * F, sfixn deg, sfixn * data){
  register int i;
  fprintf(F, "[");
  for(i=0;i<deg;i++) fprintf(F, "%ld, ", data[i]);
  fprintf(F, "%ld]\n", data[deg]);
}


void printVecST(sfixn S, sfixn T, sfixn * data){
  register int i;
  printf("[");
  for(i=S;i<T;i++) printf("%ld, ", data[i]);
  printf("%ld]\n", data[T]);
}


void printVecLong(sfixn deg, longfixnum * data){
  register int i;
  printf("[");
  for(i=0;i<deg;i++) printf("%lld, ", data[i]);
  printf("%lld]\n", data[deg]);
}


void printVecDoubleDoubleBlock(sfixn Xst, sfixn Xed, sfixn Xinterval, sfixn Yst, sfixn Yed,sfixn Yinterval,  double ** data, sfixn deg, sfixn * vec){
  register int i,j;
  printf("d1=i*%ld, i=%ld,...,%ld.\n", Xinterval, Xst, Xed);
  printf("d2=j*%ld, j=%ld,...,%ld.\n", Yinterval, Yst, Yed);
  printf("DivDoors=");
  printVecST(1,deg, vec);

  for(i=Xst; i<=Xed; i++){
    for(j=Yst; j<=Yed; j++) {
      printf("[%f]\t", data[i][j]);
	  }
    printf("\n");
  }
  printf("\n");
}



void fprintVecST(FILE * F, sfixn S, sfixn T, sfixn * data){
  register int i;
  fprintf(F, "[");
  for(i=S;i<T;i++) fprintf(F, "%ld, ", data[i]);
  fprintf(F, "%ld]\n", data[T]);
}




/* void fprintVecDoubleDoubleBlock(FILE * F, sfixn Xst, sfixn Xed, sfixn Xinterval, sfixn Yst, sfixn Yed,sfixn Yinterval,  double ** data, sfixn deg, sfixn * vec){ */
/*   register int i,j; */
/*   fprintf(F, "d1=i*%ld, i=%ld,...,%ld.\n", Xinterval, Xst, Xed); */
/*   fprintf(F, "d2=j*%ld, j=%ld,...,%ld.\n", Yinterval, Yst, Yed); */
/*   fprintf(F, "DivDoors="); */
/*   fprintVecST(F,1,deg, vec); */

/*   for(i=Xst; i<=Xed; i++){ */
/*     for(j=Yst; j<=Yed; j++) { */
/*       fprintf(F, "[%f]\t", data[i][j]); */
/* 	  } */
/*     fprintf(F, "\n"); */
/*   } */
/*   printf("\n"); */
/* } */



/* void fprintVecDoubleDoubleBlockGnuplot(FILE * F, sfixn Xst, sfixn Xed, sfixn Xinterval, sfixn Yst, sfixn Yed,sfixn Yinterval,  double ** data, sfixn deg, sfixn * vec){ */
/*   register int i,j; */
/*   //fprintf(F, "d1=i*%ld, i=%ld,...,%ld.\n", Xinterval, Xst, Xed); */
/*   //fprintf(F, "d2=j*%ld, j=%ld,...,%ld.\n", Yinterval, Yst, Yed); */
/*   //fprintf(F, "DivDoors="); */
/*   //fprintVecST(F,1,deg, vec); */
/*   double zero=0.0; */
/*   for(i=Xst; i<=Xed; i++){ */
/*     for(j=Yst; j<=Yed; j++) { */
/*       if(data[i][j] > zero) fprintf(F, "%d   %d   %f\n", i, j, data[i][j]); */
/* 	  } */
/*   } */
/*   fprintf(F, "\n"); */
/* } */



//==========================================================
// ceil(log_2 m).
//==========================================================
// can't handle number> 2^31.

/**
 * logceiling:
 * @m: an integer.
 * 
 * Return ceiling( log2(m) ): 
 **/
sfixn logceiling(sfixn m){
   sfixn power=0;
   sfixn tmpm=m;
   if(m>>(BASE-1)) {return BASE;}
   while(tmpm){tmpm>>=1; power=power+1;}
   if((m<<1)==(1<<power)) power=power-1;
   return power;
}


//==========================================================
// Random Vector.
//==========================================================
// note: Create random vector with random coefficients.


/**
 * randomVec:
 * @vec: a vector.
 * @s: size of the 'vec'.
 * @p: a prime number 
 * 
 *  filling the vector 'vec' by integers in Z/pZ.
 * Return value: 
 **/
sfixn * randomVec(sfixn * vec, sfixn s, sfixn p)
{	sfixn i;
	srand(getSeed());
	for(i=0; i<s; i++) vec[i]=(rand()%p);
        if(! vec[s-1]) vec[s-1]=1;
	return(vec); 
}




/**
 * EX_RandomUniPolyCoeffsVec:
 * @d: an integer.
 * @p: a prime number 
 * 
 * Return value: a vector of size d+1 filled with random numbers in Z/pZ. 
 **/
sfixn * EX_RandomUniPolyCoeffsVec(sfixn d, sfixn p)
{	sfixn i;
        sfixn *vec=(sfixn *)my_calloc(d+1, sizeof(sfixn));
	for(i=0; i<=d; i++) vec[i]=(rand()%p);
        while(! vec[d]) vec[d]=rand()%p;
	return(vec);
}


/**
 * EX_RandomUniPolyCoeffsVec:
 * @d: an integer.
 * @p: a prime number 
 * 
 * Return value: a vector of size d+1 filled with random numbers in Z/pZ. But the leadinfCoefficient is 1.
 **/
sfixn * EX_RandomUniPolyCoeffsVecMonic(sfixn d, sfixn p)
{	sfixn i;
        sfixn *vec= (sfixn *)my_calloc(d+1, sizeof(sfixn));
	for(i=0; i<=d; i++) vec[i]=(rand()%p);
        vec[d]=1;
	return(vec);
}




/**
 * randomMonicVec:
 * @vec: a vector.
 * @s: an integer.
 * @p: a prime number 
 * 
 * Return value: The vector 'vec' of size s filled with random numbers in Z/pZ. But the leadinfCoefficient is 1.
 **/
sfixn * randomMonicVec(sfixn * vec, sfixn s, sfixn p)
{	sfixn i;
	srand(getSeed());
	for(i=0; i<s; i++) vec[i]=(rand()%p);
        vec[s-1]=1;
	return(vec); 
}





sfixn * randomVecSeed(sfixn forceItBeOne, sfixn * vec, sfixn s, sfixn p, unsigned seed)
{
	sfixn i;
	srand(seed);
	for(i=0; i<s; i++){
          if(forceItBeOne) vec[i]=1; else vec[i]=(rand()%p);
        }
        if(! vec[s-1]) vec[s-1]=1;
	return(vec); 
}



//==========================================================
// All 1 Vectors.
//==========================================================
// note: 

/**
 * allOneVec:
 * @vec: A vector.
 * @s: size of 'vec'. 
 * 
 * filling 1s.
 * Return value: 
 **/
sfixn * allOneVec(sfixn * vec, sfixn s)
{
  sfixn i;
	for(i=0; i<s; i++) vec[i]=1;
	return(vec); 
}


//==========================================================
// clean vectors.
//==========================================================
// note: clean the whole vector.
void cleanVec(sfixn deg, sfixn * cof){
  register sfixn i;
  for(i=0;i<=deg;i++) cof[i]=0;
}

// note: clean the vector from..to.
void cleanVecft(sfixn from, sfixn to, sfixn * cof){
  register sfixn i;
  for(i=from;i<=to;i++) cof[i]=0;
}


// note: clean the whole vector.
void cleanVecINT(sfixn deg, int * cof){
  register int i;
  for(i=0;i<=deg;i++) cof[i]=0;
}

// note: clean the whole vector.
void cleanVecDOUBLE(int deg, double * cof){
  register int i;
  for(i=0;i<=deg;i++) cof[i]=0;
}



//==========================================================
// Print univariate polynomial.
//==========================================================

/**
 * printPolyUni:
 * @deg: degree of the univariate polynomial.
 * @coeffs: coefficients of the univariate polynomial. 
 *  print the univariate polynomial (deg, coeffs).
 * Return value: 
 **/
void
printPolyUni(int deg, sfixn * coeffs)
{
  int i=0;
  printf("%ld",coeffs[0]);
  for(i=1;i<=deg;i++) printf("+%ld*x^%d", coeffs[i],i);
  printf("\n");
}



/**
 * fprintPolyUni:
 * @file: the handler of a FILE.
 * @deg: degree of the univariate polynomial.
 * @coeffs: coefficients of the univariate polynomial. 
 *  print the univariate polynomial (deg, coeffs) into a FILE.
 * Return value: 
 **/
void
fprintPolyUni(FILE *file, int deg, sfixn * coeffs)
{
  int i=0;
  fprintf(file, "%ld",coeffs[0]);
  for(i=1;i<=deg;i++) fprintf(file, "+%ld*x^%d", coeffs[i],i);
  fprintf(file, "\n");
}





/**
 * EX_printPolyUni:
 * @deg: degree of the univariate polynomial.
 * @coeffs: coefficients of the univariate polynomial. 
 * @var: a character.
 *  print the univariate polynomial (deg, coeffs) in variable 'var'.
 * Return value: 
 **/
void
EX_printPolyUni(int deg, sfixn * coeffs, char var)
{
  int i=0;
  printf("%ld",coeffs[0]);
  for(i=1;i<=deg;i++) printf("+%ld*%c^%d", coeffs[i], var, i);
  printf("\n");
}






//==========================================================
// gettime().
//==========================================================



/**
 * gettime:
 * 
 * Return value: return current system time. 
 **/
double 
gettime()
{
     return 1.0;
}


//==========================================================
// getSeed().
//==========================================================

/**
 * getSeed:
 * 
 * Return value: a seed (integer) for generating random numbers. 
 **/
unsigned
getSeed()
{
    return 1;
}


//==========================================================
// Reverse the array in place.
//==========================================================

/**
 * reverse1:
 * @deg: 'deg'+1 is the size of 'vec'.
 * @vec: a integer vector. 
 *   reverse the coefficients in 'vec'.
 * Return value: 
 **/
sfixn *
reverse1(sfixn deg, sfixn * vec){
  register int i;
  sfixn m=(deg+1)/2, tmp;
  for(i=0; i<m; i++){
    tmp=vec[i];
    vec[i]=vec[deg-i];
    vec[deg-i]=tmp;
      }
  return vec;
}

//==========================================================
// Reverse the array vec2 into vec1.
//==========================================================



/**
 * reverseUni:
 * @deg: 'deg'+1 is the size for both 'vec1', and 'vec2'.
 * @vec1: a destination vector.
 * @vec2: a source vector.
 * 
 * copy the data in 'vec2' in a reverse order into 'vec1'.
 * 
 * Return value: 'vec1'
 **/
sfixn *
reverseUni(sfixn deg, sfixn * vec1, sfixn * vec2){
  register int i;
  for(i=0; i<=deg; i++){
    vec1[i]=vec2[deg-i];
      }
  return vec1;
}

//==========================================================
// compare two vector. equal return 1, otherwise return 0.
//==========================================================
/**
 * compareVec:
 * @deg: degree of both 'vec1' and 'vec2'.
 * @vec1: a vector.
 * @vec2: a vector.
 * To compare two input vectors 'vec1' and 'vec2'.
 * if they are equal returns 1 otherwise return 0.
 * Return value: 
 **/
int
compareVec(sfixn deg, sfixn * vec1, sfixn * vec2){
  register int i;
  for(i=0; i<=deg; i++){
    if(vec1[i]!=vec2[i]) {return(0);};
      }
  return 1;
}



//=================================================================
// reverse poly in f[x1,...,xn][y].
//=================================================================

// The Input multivariate polynomial f[x1,...,xn][y] encoded in a linear array.
// deg is the degree of f in y.
// sizOfCoef is the size of coefficient of f in y.
// degree >=0, sizOfCoef >0;
sfixn *
reverseMulti_1(sfixn deg, sfixn sizOfCoef, sfixn * vec){
  register int i, j;
  sfixn * tmpCoef=(sfixn * )my_malloc(sizOfCoef*sizeof(sfixn));
  sfixn * tmp1Ptr=vec, * tmp2Ptr=vec+sizOfCoef*deg;
  for(i=0; i<((deg+1)/2); i++){
    for(j=0; j<sizOfCoef; j++) tmpCoef[j]=tmp1Ptr[j];
    for(j=0; j<sizOfCoef; j++) tmp1Ptr[j]=tmp2Ptr[j];
    for(j=0; j<sizOfCoef; j++) tmp2Ptr[j]=tmpCoef[j];
    tmp1Ptr+=sizOfCoef;
    tmp2Ptr-=sizOfCoef;
  }
  my_free(tmpCoef);
  return vec;
}



/**
 * reverseMulti:
 * @deg: ('deg'+1)*'sizOfCoef' is the size for both 'outVec', and 'inVec'.
 * @sizOfCoef: An integer number. 
 * @outVec: A destination vector.
 * @inVec: A source vector.
 * 
 * Copy 'deg'+1 data blocks from 'outVec' in a reverse order into 'outVec'.
 * each data blocks have size 'sizOfCoef'.
 * Return value: 'outVec'
 **/
sfixn *
reverseMulti(sfixn deg, sfixn sizOfCoef, sfixn * outVec, sfixn * inVec){
  register sfixn i, j;
  sfixn end=deg*sizOfCoef;
  for(i=0; i<=end; i+=sizOfCoef){
    for(j=0; j<sizOfCoef; j++) outVec[end-i+j]=inVec[i+j];
     }
  return outVec;
}






/* //================================================================= */
/* // Ex-Gcd for numbers. */
/* //================================================================= */
/* static inline */
/* void egcd (sfixn x, sfixn y, sfixn *ao, sfixn *bo, sfixn *vo)  { */
/*   sfixn tmp; */
/*   sfixn A,B,C,D,u,v,q; */

/*   u = y; v = x;  */
/*   A=1; B=0;  */
/*   C=0; D=1; */

/*   do { */
/*     q = u / v; */
/*     tmp = u; */
/*     u = v; */
/*     v = tmp - q*v; */
/*     tmp = A; */
/*     A = B; */
/*     B = tmp - q*B; */
/*     tmp = C; */
/*     C = D; */
/*     D = tmp - q*D; */
/*   } while (v != 0); */
/*   *ao=A;  */
/*   *bo=C;  */
/*   *vo=v; */
/* } */




/* //================================================================= */
/* // Modular inverse for numbers. */
/* //================================================================= */
/* sfixn inverseMod(sfixn n, sfixn PPP){ */
/*   sfixn a, b, v; */
/*   egcd(n, PPP, &a, &b, &v); */
/*   if (b < 0) */
/*     b += PPP; */
/*   return b % PPP; */
/* } */



//=================================================================
// Print error message.
//================================================================= */
/* void aborting(const char * str){ */
/*   printf(str); */
/*   printf("\n"); */
/* } */


//=================================================================
// Print exception message.
//=================================================================
void printException(int menu){

 switch (menu)              
    {
    case 0:
        printf("Degrees in TriSet must be >=2!\n");        
        break;
    case 1:
        printf("Inversion Failed !\n");        
        break;
    case 2:
        printf("GCD Failed!\n");        
        break;
    default: printf("Unknown Exeption! \n");
    }


}


void
freeVecVec(sfixn m, sfixn **ptsPtr){
  int i;

  if(ptsPtr)
    for(i=1; i<=m; i++){
      if(ptsPtr[i]){
       my_free(ptsPtr[i]);
       ptsPtr[i]=NULL;
      }
    
    }
my_free(ptsPtr);
}





/* interrupt signal handler: sets global variable when user hits Ctrl-C */
void CDECL catch_intr( int signo )
{
    Interrupted = 1;
    signal(SIGINT,catch_intr);
#ifdef _MSC_VER
    signal(SIGBREAK,catch_intr);
#endif
}










/* // ========================================================================= */



/* // For Parallel Version */



/* // ========================================================================= */



/* // For Parallel Version */



/* // ========================================================================= */



/* // For Parallel Version */



/* // ========================================================================= */



/* // For Parallel Version */



/* // ========================================================================= */






/* //================================================================= */
/* // Scheduling "size" tasks onto "n" processors as evenly as possible. */
/* // size = q * n + r   */
/* //================================================================= */
/* void qr_seprator(int *nloop, int *q, int *r, int size, int n){ */
/*   double qf; */
/*   int acu=0; */
/*   *nloop=0; */
/*   qf=(double)size/(double)n; */
/*   *q=(int)(ceil(qf)); */
/*   while(acu<(size-(*q))){ */
/*     acu+=(*q); */
/*     (*nloop)++; */
/*   } */
/*   *r=size-acu; */
/* } */



/* // q=size / (n*co); */
/* // r=size % (n*co); */
/* void qr_seprator2(int *nloop, int *q, int *r, int size, int n, int co){ */
/*   double qf; */
/*   int acu=0, qb; */
/*   *nloop=0; */
/*   qf=(double)size/(double)(n*co); */
/*   *q=(int)(ceil(qf)); */
/*   qb=(*q) * co; */
/*   while(acu<(size-qb)){ */
/*     acu+=qb; */
/*     (*nloop)++; */
/*   } */
/*   *r=size-acu; */
/* } */





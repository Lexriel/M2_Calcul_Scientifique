#include "taylor_shift_conf.h"
#include "taylor_shift_kernel.h"
#include "inlines.h"

// fast multiplication of two polynomials, created by Sardar Haque, I modify just a line to use it in my code
__global__ void listPlainMulGpu_and_right_shift_GPU(sfixn *Mgpu1, sfixn *Mgpu2 , sfixn length_poly, sfixn poly_on_layer, sfixn threadsForAmul, sfixn mulInThreadBlock, sfixn p, double pinv)
{

	__shared__ sfixn sM[2*Tmul];
	/*
	sM is the shared memory where the all the coefficients and intermediate multiplications results
	are stored. For each multiplication it reserve 4*length_poly -1 spaces.
	mulID is the multiplication ID. It refers to the poly in Mgpu2 on which it will work.
	mulID must be less than (poly_on_layer/2).
	*/    
	sfixn mulID= ((threadIdx.x/threadsForAmul) + blockIdx.x*mulInThreadBlock);

	if (mulID < (poly_on_layer/2) && threadIdx.x < threadsForAmul*mulInThreadBlock)
	{
		/*
		The next 10 lines of code copy the polynomials in Mgpu1 from global memory to shared memory.
		Each thread is responsible of copying one coefficient.
		A thread will copy a coefficient from Mgpu1[( mulID* length_poly*2)...( mulID* length_poly*2) + length_poly*2 -1] 
		j+u gives the right index of the coefficient in Mgpu1.

		In sM, the coefficients are stored at the lower part.
		t will find the right (4*length_poly-1) spaced slot for it.
		s gives the start index of its right slot.
		s+u gives right position for the index.
		*/

		sfixn j = ( mulID* length_poly*2);
		sfixn q = ( mulID*(2*length_poly));  // modified, clean the -1
	    	sfixn t = (threadIdx.x/threadsForAmul);
		sfixn u = threadIdx.x % threadsForAmul;

		sfixn s = t*(4*length_poly-1);
		sfixn k = s + length_poly;
		sfixn l = k + length_poly;
		sfixn c = l+u;
		sfixn a, b, i;

		sM[s+u] = Mgpu1[j + u];
		__syncthreads();

		if(u != (2*length_poly-1) )
		{
		/*
		In the multiplication space, the half of the leading coefficients 
		are computed differently than the last half. Here the computation of 
		first half are shown. the last half is shown in else statement.
		In both cases sM[c] is the cofficient on which this thread will work on.
		sM[a] is the coefficient of one poly.
		sM[b] is the coefficient of the other poly.
		*/
			if(u < length_poly)
			{
				a = s;
				b = k + u;   
    				sM[c] =  mul_mod(sM[a],sM[b],p,pinv);
				++a; --b;

				for(i = 0; i < u; ++i, ++a, --b)
				sM[c] =  add_mod(mul_mod(sM[a],sM[b],p,pinv),sM[c] ,p);
				Mgpu2[q+u+1] = sM[c]; //+1 added
			}

			else
			{
				b = l - 1;
				a = (u - length_poly) + 1 + s;
				sM[c] =  mul_mod(sM[a],sM[b],p,pinv);
			     	++a; --b;

				sfixn tempU = u;
				u = (2*length_poly-2) - u;

				for(i = 0; i < u; ++i, ++a, --b)
					sM[c] =  add_mod(mul_mod(sM[a],sM[b],p,pinv),sM[c] ,p); 
				Mgpu2[q+tempU+1] = sM[c];  //+1 added
			}
		}

		else
			Mgpu2[q] = 0; // added for put 0 at position

	}
}


// create array identity (initialization of the array Fact)
__global__ void identity_GPU(sfixn *T, sfixn n)
{
  sfixn k = blockIdx.x * blockDim.x + threadIdx.x;
  sfixn boolean = (sfixn) (k == 0);

  if (k < n+1)
    T[k] = k + boolean;
}


// create all the elements of Factorial (%p)
__global__ void create_factorial_GPU(sfixn *Fact, sfixn n, sfixn e, sfixn p, double pinv) // warning : n+1 is the size of Fact but we will just full the n last element, not the first one
{
  sfixn k = blockIdx.x * blockDim.x + threadIdx.x;
  sfixn i, j, part, pos, base;
  sfixn L = 1;
  sfixn B = 2;

  if (k < n/2)
  {
    // step 1
    Fact[2*k+1] = mul_mod(Fact[2*k], Fact[2*k+1], p, pinv);
    __syncthreads();

    // next steps
    for (i=1; i<e; i++)
    {
      B *= 2;
      L *= 2;
      part = k / L;
      pos = k % L;
      __syncthreads();
      j = L + part*B + pos;
      __syncthreads();
      base = Fact[L + part*B - 1];
      __syncthreads();
      Fact[j] = mul_mod(base, Fact[j], p, pinv);
      __syncthreads();
    }
  }    
}


__global__ void create_factorial_step0_GPU(sfixn *Fact, sfixn n, sfixn e, sfixn p, double pinv) // warning : n+1 is the size of Fact but we will just full the n last element, not the first one
{
  sfixn k = blockIdx.x * blockDim.x + threadIdx.x;

  if (k < n/2)
  {
    // step 1
    Fact[2*k+1] = mul_mod(Fact[2*k], Fact[2*k+1], p, pinv);
  }    
}


__global__ void create_factorial_stepi_GPU(sfixn *Fact, sfixn n, sfixn e, sfixn p, double pinv, sfixn L) // warning : n+1 is the size of Fact but we will just full the n last element, not the first one
{
  sfixn k = blockIdx.x * blockDim.x + threadIdx.x;
  sfixn j, part, pos, base;
  sfixn B = 2 * L;

  if (k < n/2)
  {

    // next steps
      part = k / L;
      pos = k % L;
      j = L + part*B + pos;
      base = Fact[L + part*B - 1];
      Fact[j] = mul_mod(base, Fact[j], p, pinv);
  }    
}


// create an array of the inverse numbers in Z/pZ
__global__ void inverse_p_GPU(sfixn *T, sfixn p, double pinv)
{
  sfixn i;
  sfixn k = blockIdx.x * blockDim.x + threadIdx.x;

  if (k < p)
  {
    if (k > 1)
      for (i=2; i<p; i++)
        {
          if (mul_mod(k, i, p, pinv) == 1)
          {
            T[k] = i;
            i = p;     // to stop the loop
          }
        }

    else if (k == 1)
      T[1] = 1;
    else // (k == 0)
      T[0] = 0;
  }
}


// create the inverse of a number in Z/pZ
__device__ sfixn inverse_GPU(sfixn k, sfixn p, double pinv)
{
  sfixn i, res;

  if (k > 1)
    for (i=2; i<p; i++)
      {
        if (mul_mod(k, i, p, pinv) == 1)
        {
          res = i;
          i = p;     // to stop the loop
        }
      }

  else if (k == 1)
    res = 1;
  else // (k == 0)
    res = 0;

  return res;
}


// creates an array of the Newton's Binomials until n modulo p (! size of the array = n+1)
__device__ sfixn create_binomial_GPU(sfixn *Factorial, sfixn *Inverse_p, sfixn n, sfixn p, double pinv, sfixn id)
{
  sfixn l = n - id;
  sfixn temp = mul_mod(Factorial[id], Factorial[l], p, pinv);

  return mul_mod(Factorial[n], Inverse_p[temp], p, pinv);
}


// create the Newton's Binomial coefficient "n choose id" modulo p
// return "n choose id" = n! / [id!(n-id)!] mod p
__device__ sfixn create_binomial2_GPU(sfixn *Factorial, sfixn n, sfixn p, double pinv, sfixn id)
{
  sfixn l = n - id;
  sfixn prod = mul_mod(Factorial[id], Factorial[l], p, pinv);

  return quo_mod(Factorial[n], prod, p, pinv);
}


// create the array of the coefficients of (x+1)^k for k in (1,2^(e-1))
__global__ void develop_xshift_GPU(sfixn *T, sfixn n, sfixn *Factorial, sfixn p, double pinv)
{
  sfixn k = blockIdx.x * blockDim.x + threadIdx.x;
  sfixn m;
  sfixn pow2 = 1;

  if (k < n)
  {
//    if (k > 1)
    {
      m = (k+1)/2; //k/2

      while (m != 0)
      {
        m /= 2;
        pow2 *= 2;
      }

      T[k] = create_binomial2_GPU(Factorial, pow2, p, pinv, k+1 - pow2);
    }
  }
}


// create the product of two arrays representing polynomials
__device__ void conv_prod_GPU(sfixn *res, sfixn *T1, sfixn *T2, sfixn m, sfixn p, sfixn local_n)
{
  sfixn i, j;
  sfixn K = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (K < m)
  {
    for (j=0; j<K; j++)
    {
      i = K - j;           // K = i+j
      if ((i < local_n+1) && (j < local_n))   // if i < local_n + 1 then T1[i] != 0, else T1[i] = 0 so useless computations 
        res[K] = (res[K] + T1[i]*T2[j]) % p;
    }

    for (j=K+1; j<m; j++)
    {
      i = K + m - j;
      if ((i < local_n+1) && (j < local_n))
        res[K] = (res[K] + T1[i]*T2[j]) % p;
    }
  }
}


// addition of two arrays
__global__ void add_arrays_GPU(sfixn *res, sfixn *T1, sfixn *T2, sfixn size, sfixn p)
{
  sfixn i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i<size)
    res[i] = add_mod(T1[i], T2[i], p);
}


// creates an array of zeros
__global__ void Zeros_GPU(sfixn *T, sfixn n)
{
  sfixn i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
    T[i] = 0;
}


// initialize Polynomial_shift
__global__ void init_polynomial_shift_GPU(sfixn *Polynomial, sfixn *Polynomial_shift, sfixn n, sfixn p)
{
  sfixn i = blockIdx.x * blockDim.x + threadIdx.x;
  sfixn j = 2*i;
  if (i < n/2)
  {
//    if (i % 2 == 0)
      Polynomial_shift[j] = add_mod(Polynomial[j], Polynomial[j+1], p);
//    else // (i % 2 == 1)
      Polynomial_shift[j+1] = Polynomial[j+1];
  }

  /* EXAMPLE for n=8 :
     after this procedure, Polynomial_shift = [f0+f1, f1, f2+f3, f3, f4+f5, f5, f6+f7, f7] */
}


// transfer at each step the polynomials which need to be multiplicated 
__global__ void transfert_array_GPU(sfixn *Mgpu, sfixn *Polynomial_shift, sfixn *Monomial_shift, sfixn n, sfixn local_n, sfixn p, double pinv)
{
  sfixn i = blockIdx.x * blockDim.x + threadIdx.x;
//  sfixn B = 2*local_n;
  sfixn pos, part, bool1, bool2;

//  __shared__ sfixn sM[NB_THREADS];

  /*                   EXAMPLE

  --------------------------------------------------------

     ARRAY Polynomial_shift_device[i-1] considered
       _________ _________ _________ _________
      |         |         |         |         |
      |         |    X    |         |    Y    |
      |_________|_________|_________|_________|
        part=0    part=1    part=2    part=3
     
     local_n = size of a part

  --------------------------------------------------------

     ARRAY Mgpu[i] considered
       ___________________ ___________________
      |                   |                   |
      |   X      (x+1)^m  |   Y      (x+1)^m  |
      |___________________|___________________|
              PART=0               PART=1
     
     B = 2 * local_n = size of a PART
     m = local_n

We want to fill the array Mgpu[i] like this : the polynomials
which need to be multiplicated by (x+1)^m are of odd part and
we store them at the beginning of each PART of Mgpu[i]. The end
of each part doesn't really contain (x+1)^m as we need arrays 
to be multiplicated, so we avoid the multiplication by 1.
Thus the end of each PART contains exactly :

 [(x+1)^m - 1] / x = m + ... + x^(m-1)    {m elements}          */


  if (i < n)
  {
    part = i / local_n;
    pos  = i % local_n;      // i = part * local_n + pos
//    PART = part / 2;
    bool2 = part % 2;  // = 0 or 1 
    bool1 = 1 - bool2; // = 1 or 0, bool1 and bool2 are contraries

//    sM[threadIdx.x] = Monomial_shift[local_n + pos];
//    sM[threadIdx.x] = create_binomial2_GPU(Factorial, local_n, p, pinv, pos+1);

    // What we want to do
/*      if (part % 2 == 0)
          Mgpu[PART * B + local_n + pos] = Monomial_shift[local_n + pos];// + 1];

        else // (part % 2 == 1)
          Mgpu[PART * B + pos] = Polynomial_shift[i];  */


    // What we do (faster)
//    Mgpu[PART * B + local_n * bool1 + pos] = bool1 * Monomial_shift[local_n + pos] + bool2 * Polynomial_shift[i];
      Mgpu[i] = bool1 * Polynomial_shift[local_n+i] + bool2 * Monomial_shift[local_n+pos];//sM[threadIdx.x];
  }
}


__global__ void right_shift_GPU(sfixn *T, sfixn n)
{
  sfixn i = blockIdx.x * blockDim.x + threadIdx.x;
  sfixn a;

  if (i < n)
  {
    a = T[i];
    __syncthreads();

    if (i < n-1)
      T[i+1] = a;
    else
      T[0] = 0;
  }
}


// add parts of three arrays between them
__global__ void semi_add_GPU(sfixn *NewPol, sfixn *PrevPol1, sfixn *PrevPol2, sfixn n, sfixn local_n, sfixn p)
{
  sfixn i = blockIdx.x * blockDim.x + threadIdx.x;
  sfixn part = i / local_n;
  sfixn pos = i % local_n;
  sfixn j = 2 * local_n * part + pos;
  sfixn res;

  if (i < n/2)
  {
/*    if (part % 2 == 0)
      NewPol[i] = add_mod(NewPol[i], PrevPol[i], p);*/
      res = add_mod(PrevPol1[j], PrevPol2[j], p);
      NewPol[j] = add_mod(NewPol[j], res, p);

  }
}


/* ================================================================================

                                    PARALLELIZE !!!!!!!

   ================================================================================ */

// Horner's method to compute g(x) = f(x+1) (equivalent to Shaw & Traub's method for a=1)
/* void horner_shift_GPU(int *Polynomial, int *Polynomial_shift, int n, int p)
{
  int i;
  int *temp;
  temp = (int*) calloc (n, sizeof(int));

  Polynomial_shift[0] = Polynomial[n-1];

  for (i=1; i<n; i++)
  {
    memcpy(temp+1, Polynomial_shift, i*sizeof(int));
    add_arrays(Polynomial_shift, Polynomial_shift, temp, n, p);
    Polynomial_shift[0] = (Polynomial_shift[0] + Polynomial[n-1-i]) % p;
  }
  
  free(temp);
} */

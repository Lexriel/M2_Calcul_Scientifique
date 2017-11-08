# ifndef _TAYLOR_SHIFT_KERNEL_
# define _TAYLOR_SHIFT_KERNEL_

// fast multiplication of two polynomials, created by Sardar Haque, I modify just a line to use it in my code
__global__ void listPlainMulGpu_and_right_shift_GPU(sfixn *Mgpu1, sfixn *Mgpu2 , sfixn length_poly, sfixn poly_on_layer, sfixn threadsForAmul, sfixn mulInThreadBlock, sfixn p, double pinv);


// create array identity (initialization of the array Fact)
__global__ void identity_GPU(sfixn *T, sfixn size);


// create all the elements of Factorial (%p)
__global__ void create_factorial_GPU(sfixn *Fact, sfixn n, sfixn e, sfixn p, double pinv); // warning : n+1 is the size of Fact but we will just full the n last element, not the first one
__global__ void create_factorial_step0_GPU(sfixn *Fact, sfixn n, sfixn e, sfixn p, double pinv);
__global__ void create_factorial_stepi_GPU(sfixn *Fact, sfixn n, sfixn e, sfixn p, double pinv, sfixn L);

// create an array of the inverse numbers in Z/pZ
__global__ void inverse_p_GPU(sfixn *T, sfixn p, double pinv);


// create the inverse of a number in Z/pZ
__device__ sfixn inverse_GPU(sfixn k, sfixn p, double pinv);


// creates an array of the Newton's Binomials until n modulo p (! size of the array = n+1)
__device__ sfixn create_binomial_GPU(sfixn *Factorial, sfixn *Inverse_p, sfixn n, sfixn p, double pinv, sfixn id);


// create the Newton's Binomial coefficient "n choose id" modulo p
__device__ sfixn create_binomial2_GPU(sfixn *Factorial, sfixn n, sfixn p, double pinv, sfixn id);



// create the array of the coefficients of (x+1)^k for several k
__global__ void develop_xshift_GPU(sfixn *T, sfixn n, sfixn *Factorial, sfixn p, double pinv);



// create the product of two arrays representing polynomials
__device__ void conv_prod_GPU(sfixn *res, sfixn *T1, sfixn *T2, sfixn m, sfixn p, sfixn local_n);


// addition of two arrays
__global__ void add_arrays_GPU(sfixn *res, sfixn *T1, sfixn *T2, sfixn size, sfixn p);


// creates an array of zeros
__global__ void Zeros_GPU(sfixn *T, sfixn n);



// initialize Polynomial_shift
__global__ void init_polynomial_shift_GPU(sfixn *Polynomial, sfixn *Polynomial_shift, sfixn n, sfixn p);
/* EXAMPLE for n=8 :
   after this procedure, Polynomial_shift = [f0+f1, f1, f2+f3, f3, f4+f5, f5, f6+f7, f7] */


// transfer at each step the polynomials which need to be multiplicated 
__global__ void transfert_array_GPU(sfixn *Mgpu, sfixn *Polynomial_shift, sfixn *Monomial_shift, sfixn n, sfixn local_n, sfixn p, double pinv);

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


__global__ void right_shift_GPU(sfixn *T, sfixn n);


__global__ void semi_add_GPU(sfixn *NewPol, sfixn *PrevPol1, sfixn *PrevPol2, sfixn n, sfixn local_n, sfixn p);

# endif // _TAYLOR_SHIFT_KERNEL_

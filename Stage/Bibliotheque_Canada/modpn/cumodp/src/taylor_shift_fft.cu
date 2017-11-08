#include "taylor_shift_conf.h"
#include "taylor_shift_cpu.h"
#include "taylor_shift_kernel.h"
#include "taylor_shift.h"
#include "taylor_shift_fft.h"
#include "inlines.h"



__global__ void mult_adjust_GPU(sfixn *Polynomial_shift, sfixn *fft, sfixn n, sfixn local_n, sfixn winv, sfixn p, double pinv)
{

  /*                   EXAMPLE


  --------------------------------------------------------

     ARRAY fft considered
       _______________ _______________ _______________ _______________
      |               |               |               |               |
      |  real coeffs  |    useless    |  real coeffs  |    useless    |
      |_______________|_______________|_______________|_______________|
           PART=0                       PART=2


     B = 2 * local_n =  size of a PART

  --------------------------------------------------------

     ARRAY Polynomial_shift_device considered
       _______________________________
      |               |               |
      |  real coeffs  |  real coeffs  |
      |_______________|_______________|
            part=0         part=1

     B = 2 * local_n = size of a part

                                                              */

  sfixn i = blockIdx.x * blockDim.x + threadIdx.x;
  sfixn B = 2*local_n;

  if (i < n)
    {
      sfixn part = i / B;
      sfixn pos  = i % B;
      sfixn bool1 = (sfixn) (pos != 0);

//       if (i % B == 0)
// 	Polynomial_shift[i] = 0;
//       else
	Polynomial_shift[i] = bool1 * mul_mod(winv, fft[2*B*part + pos-1], p, pinv);

//       a = Polynomial_shift[i];
//       __syncthreads();
//       Polynomial_shift[(i+1)%n] = a;
      //      Polynomial_shift[i] = bool1 * winv * fft[2*B*part + pos - 1];
    }
}


// transfer at each step the polynomials which need to be multiplicated
__global__ void transfert_array_fft_GPU(sfixn *fft, sfixn *Mgpu, sfixn n, sfixn local_n)
{
  sfixn i = blockIdx.x * blockDim.x + threadIdx.x;
//  sfixn B = 2*local_n;
  sfixn part, pos, bool1, bool2;
  part = i / local_n;
  pos  = i % local_n;
  bool2 = part % 2;
  bool1 = 1 - bool2;

  if (i<2*n)
  {
//    if (part % 2 == 0)
      fft[i] = bool1 * Mgpu[(part/2)*local_n + pos];
//    else
//      fft[i] = 0;
  }
}

__global__ void full_monomial(sfixn *Mgpu, sfixn *Monomial_shift, sfixn n, sfixn local_n)
{
  sfixn i = blockIdx.x * blockDim.x + threadIdx.x;
  sfixn part = i / local_n;
  sfixn pos  = i % local_n;

  if (i < n)
  {
    if (part % 2 == 1)
      Mgpu[i] = Monomial_shift[pos];
  }
}


// transfer at each step the polynomials which need to be multiplicated
/*__global__ void transfert_array_fft_GPU(sfixn *fft, sfixn *Polynomial_shift, sfixn *Monomial_shift, sfixn n, sfixn local_n)
{
  sfixn i = blockIdx.x * blockDim.x + threadIdx.x;
  sfixn B = 4*local_n;
  sfixn PART, pos_PART, part, pos_part, bool1, bool2;

  //                   EXAMPLE

//  Let's consider i = B * PART + pos_PART.
//  PART = i / B;
//  pos_PART = i % B;

//  Let's consider pos_PART = local_n * part + pos_part.
//  part = pos_PART / local_n;
//  pos_part = pos_PART % local_n;


//  --------------------------------------------------------

//     ARRAY Polynomial_shift_device[i-1] considered
//       _________ _________ _________ _________
//      |         |         |         |         |
//      |         |    X    |         |    Y    |
//      |_________|_________|_________|_________|
//        part=0    part=1    part=2    part=3

//     local_n = size of a part

//  --------------------------------------------------------

//     ARRAY fft[i] considered
//       ______________________________________ ______________________________________
//      |                                      |                                      |
//      |   X      0      (x+1)^m       0      |   Y      0      (x+1)^m       0      |
//      |______________________________________|______________________________________|
//                       PART=0                                  PART=1

//     B = 4 * local_n = size of a PART
//     m = local_n

// We want to fill the array fft[i] like this : the polynomials
// which need to be multiplicated by (x+1)^m are of odd part and
// we store them at the beginning of each PART of fft[i]. The end
// of each part doesn't really contain (x+1)^m as we need arrays
// to be multiplicated, so we avoid the multiplication by 1.
// Thus the 3rd part of each PART contains exactly :

// [(x+1)^m - 1] / x = m + ... + x^(m-1)    {m elements}


  if (i < 2*n)
  {
  PART = i / B;
  pos_PART = i % B;
  part = pos_PART / local_n;
  pos_part = pos_PART % local_n;

  bool1 = (sfixn) (part % 4 == 0);
  bool2 = (sfixn) (part % 4 == 2);
//  bool3 = (1-bool1) * (1-bool2); // = (sfixn) (part % 4 > 1);

    // What we want to do
//      if (part % 2 == 1)
//        fft[i] = 0;

//      else if (part % 4 == 0)
//        fft[i] = Polynomial_shift[local_n + 2*local_n*PART + pos_part];

//      else // (part % 4 == 2)
//        fft[i] = Monomial_shift[local_n + pos_part];


    // What we do (faster)
//    fft[i] = bool1 * Polynomial_shift[local_n + 2 * local_n * PART +  pos_part] + bool2 * Monomial_shift[local_n + pos_part];// + bool3 * 0;
//      Mgpu[i] = bool1 * Monomial_shift[local_n+pos] + bool2 * Polynomial_shift[i];
  }
}*/

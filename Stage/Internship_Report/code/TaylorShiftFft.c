#include "taylor_shift_conf.h"
#include "taylor_shift_cpu.h"
#include "taylor_shift_kernel.h"
#include "taylor_shift.h"
#include "taylor_shift_fft.h"
#include "inlines.h"

__global__ void mult_adjust_GPU(sfixn *Polynomial_shift, sfixn *fft, 
                sfixn n, sfixn local_n, sfixn winv, sfixn p, double pinv)
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

      Polynomial_shift[i] = bool1 * mul_mod(winv, fft[2*B*part + pos-1], \
                                              p, pinv);
    }
}

// transfer at each step the polynomials which need to be multiplicated
__global__ void transfert_array_fft_GPU(sfixn *fft, sfixn *Mgpu, sfixn n, \
                                        sfixn local_n)
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

__global__ void full_monomial(sfixn *Mgpu, sfixn *Monomial_shift, \
sfixn n, sfixn local_n)
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

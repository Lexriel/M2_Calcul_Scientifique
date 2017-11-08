#include "taylor_shift_conf.h"
#include "inlines.h"
#include "taylor_shift_cpu.h"
#include "taylor_shift_kernel.h"
#include "taylor_shift.h"
#include "taylor_shift_fft.h"
#include "list_pointwise_mul.h"
#include "list_stockham.h"

/* Important to notice :

  n  : number of coefficients of the polynomial considered
 n-1 : degree of the polynomial considered
  p  : prime number, it must be greater than n
*/


// Taylor_shift procedure
void taylor_shift_GPU(sfixn n, sfixn e, char *file, sfixn p, double pinv)
{

  // declaration of variables
  sfixn i, nb_blocks, local_n;
  sfixn *Factorial_device;
  sfixn *Polynomial, *Polynomial_device;
  sfixn *Monomial_shift_device;
  sfixn *temp;
  sfixn *Mgpu;
  sfixn *Polynomial_shift_device[2];
  float cpu_time, gpu_time, outerTime;
  cudaEvent_t start, stop;     /* Initial and final time */

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // beginning parameters
  local_n = 2;
  stock_file_in_array(file, n, Polynomial);

//  printf("  * pinv = %0.20lf\n", pinv);

  // TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cpu_time, start, stop);
  cudaEventDestroy(stop);
  cpu_time /= 1000.0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Create the array Factorial
  cudaMalloc( (void **) &Factorial_device, (n+1) * sizeof(sfixn) );
  nb_blocks = number_of_blocks(n+1);
  cudaThreadSynchronize();
  identity_GPU<<<nb_blocks, NB_THREADS>>>(Factorial_device, n+1);
  cudaThreadSynchronize();
  nb_blocks = number_of_blocks(n/2);
  create_factorial_step0_GPU<<<nb_blocks, NB_THREADS>>>(Factorial_device+1,\
  n, e, p, pinv);
  cudaThreadSynchronize();
  sfixn L = 1;
  for (i=1; i<e; i++)
  {
    L *= 2;
    create_factorial_stepi_GPU<<<nb_blocks, NB_THREADS>>>(Factorial_device+1,\
    n, e, p, pinv, L);
    cudaThreadSynchronize();
  }

  // Create the array of the (x+1)^i
  cudaMalloc( (void **) &Monomial_shift_device , n * sizeof(sfixn) ); // n+1
  cudaThreadSynchronize();
  nb_blocks = number_of_blocks(n);
  develop_xshift_GPU<<<nb_blocks, NB_THREADS>>>(Monomial_shift_device, n, \
  Factorial_device, p, pinv);
  cudaThreadSynchronize();
  cudaFree(Factorial_device);

  /* ************************************************************

                     1st step : initialization 

     ************************************************************ */

  cudaMalloc( (void **) &Polynomial_device, n * sizeof(sfixn) );
  cudaMemcpy( Polynomial_device, Polynomial, n*sizeof(sfixn), \
  cudaMemcpyHostToDevice );
  free(Polynomial);
  cudaMalloc( (void **) &Polynomial_shift_device[0], n * sizeof(sfixn) );
  cudaThreadSynchronize();

  // initialize polynomial_shift
  nb_blocks = number_of_blocks(n/2);
  init_polynomial_shift_GPU<<<nb_blocks, NB_THREADS>>>(Polynomial_device, \
  Polynomial_shift_device[0], n, p);
  cudaThreadSynchronize();

  /* ************************************************************

                          next steps (i<10)

     ************************************************************ */

  sfixn polyOnLayerCurrent = n/2;
  sfixn mulInThreadBlock;
  cudaMalloc((void **)&Mgpu, n * sizeof(sfixn));
  sfixn I = 9;
  if (e < 9)
    I = e;
  cudaMalloc( (void **) &Polynomial_shift_device[1], n * sizeof(sfixn) );
  cudaThreadSynchronize();

  // LOOP
  for (i=1; i<I; i++)
  {
    // transfer the polynomials which will be computed
    nb_blocks = number_of_blocks(n);
    transfert_array_GPU<<<nb_blocks, NB_THREADS>>>(Mgpu, \
    Polynomial_shift_device[(i+1)%2], Monomial_shift_device, n, local_n, p, pinv);
    cudaThreadSynchronize();

    // Compute the product of the polynomials in Mgpu ('P2 * Bin' with Bin
    // the array of binomials) and store them in Polynomial_shift_device[i%2]
    // shifted at the right for the multiplication by x so do
    // [( (x+1)^i - 1 ) / x ] * P2(x+1), then multiply it by x so we have 
    //[(x+1)^i - 1] * P2(x+1)
    mulInThreadBlock = (sfixn) floor((double) NB_THREADS / (double) (2*local_n));
    nb_blocks = (sfixn) ceil(((double) polyOnLayerCurrent/(double)mulInThreadBlock)*0.5);
    listPlainMulGpu_and_right_shift_GPU<<<nb_blocks, NB_THREADS>>>(Mgpu,  \
    Polynomial_shift_device[i%2], local_n, polyOnLayerCurrent, 2*local_n, \
    mulInThreadBlock, p, pinv);
    cudaThreadSynchronize();

    // add [(x+1)^i - 1] * P2(x+1) with P2(x+1) then we get (x+1)^i * P2(x+1) \
       then do P1(x+1) + (x+1)^i*P2(x+1)
    nb_blocks = number_of_blocks(n/2);
    semi_add_GPU<<<nb_blocks, NB_THREADS>>>(Polynomial_shift_device[i%2], Mgpu, \
    Polynomial_shift_device[(i+1)%2], n, local_n, p);
    cudaThreadSynchronize();

    // for the next step
    polyOnLayerCurrent /= 2;
    local_n *= 2;
  }

  /* ************************************************************

                       next steps : FFT (i >= 10)

     ************************************************************ */

  sfixn J = e;
  if (e < 9)
    J = 9;
  sfixn w;
  sfixn *fft_device;
  cudaMalloc( (void **) &fft_device, 2 * n * sizeof(sfixn) );
  cudaThreadSynchronize();

  // LOOP
  for (i=9; i<J; i++)
  {
    // transfer the polynomials which will be FFTed and Mgpu
    nb_blocks = number_of_blocks(n);
    transfert_array_GPU<<<nb_blocks, NB_THREADS>>>(Mgpu, \
    Polynomial_shift_device[(i+1)%2], Monomial_shift_device, n, local_n, p, pinv);
    cudaThreadSynchronize();
    nb_blocks = number_of_blocks(2*n);
    transfert_array_fft_GPU<<<nb_blocks, NB_THREADS>>>(fft_device, Mgpu, n, \
    local_n);
    cudaThreadSynchronize();

    // Convert the polynomials in the FFT world
    w = primitive_root(i+1, p);
    list_stockham_dev(fft_device, polyOnLayerCurrent, i+1, w, p);
    cudaThreadSynchronize();

    // same operation than for ListPlainMul but in the FFT world
    nb_blocks = number_of_blocks(2*n);
    list_pointwise_mul<<<nb_blocks, NB_THREADS>>>(fft_device, 2*local_n, p, pinv, \
    2*n);
    cudaThreadSynchronize();

    // return to the real world
    w = inv_mod(w, p);
    list_stockham_dev(fft_device, polyOnLayerCurrent, i+1, w, p);
    cudaThreadSynchronize();

    // adjust the real coefficients : we need to multiplicate by the following w
    // to have to correct size
    w = inv_mod(2*local_n, p);
    nb_blocks = number_of_blocks(n);
    mult_adjust_GPU<<<nb_blocks, NB_THREADS>>>(Polynomial_shift_device[i%2], \
    fft_device, n, local_n, w, p, pinv);
    cudaThreadSynchronize();

    // semi_add
    nb_blocks = number_of_blocks(n/2);
    semi_add_GPU<<<nb_blocks, NB_THREADS>>>(Polynomial_shift_device[i%2], Mgpu, \
    Polynomial_shift_device[(i+1)%2], n, local_n, p);
    cudaThreadSynchronize();

    // for the next steps
    polyOnLayerCurrent /= 2;
    local_n *= 2;
  }

  /* ************************************************************

                         end : results

     ************************************************************ */

  // Copy the last array containing the Taylor shift by 1 of the input polynomial
  temp = (sfixn*) malloc(n * sizeof(sfixn));
  cudaMemcpy( temp, Polynomial_shift_device[(e-1)%2], n*sizeof(sfixn), \
  cudaMemcpyDeviceToHost );
  cudaThreadSynchronize();

  // TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpu_time, start, stop);
  cudaEventDestroy(stop);
  gpu_time /= 1000.0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // stockes the array of Newton's coefficients in a file
  char name_file[100];
  sprintf(name_file, "Pol%d.shiftGPU_%d.dat\0", e, p);
  stock_array_in_file(name_file, temp, n);

  // deallocation of the last arrays
  free(temp);
  cudaFree(Monomial_shift_device);
  cudaFree(Mgpu);
  cudaFree(fft_device);
  for (i=0; i<2; i++)
    cudaFree(Polynomial_shift_device[i]);

  // TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&outerTime, start, stop);
  cudaEventDestroy(stop);
  outerTime /= 1000.0;
  cpu_time += outerTime;
}

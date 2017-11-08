#include "taylor_shift_conf.h"
#include "inlines.h"
#include "taylor_shift_cpu.h"
#include "taylor_shift_kernel.h"
#include "taylor_shift.h"
#include "taylor_shift_fft.h"
#include "list_pointwise_mul.h"
#include "list_stockham.h"

/*
          (sfixn *Y, sfixn *primes, sfixn n)

int i, j;
// n = d
  for (i=0; i<n; i++)
  {
    for (j=k+1; i<s; j++)
    {

    }
  }

sfixn *D, *L;
cudaMalloc( (void **) D, s*(s-1)/2 * sizeof(sfixn) );
cudaMalloc( (void **) L, s*(s-1)/2 * sizeof(sfixn) );
*/

__device__ void funcPOS(sfixn i, sfixn s, sfixn *primes, sfixn *out)
{
  sfixn out[0] = 0;
  sfixn out[1] = 0;

  while (out[0]<i)
  {
    out[1]++;
    out[0] += s - out[1];
  }

  if (out[0] != i)
  {
    out[0] -= s - out[1];
    out[1]--;
  }
}


__global__ void createDL(sfixn *D, sfixn *L, sfixn *primes, sfixn s)
{
  sfixn i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i<s*(s-1)/2)
  {
    sfixn k_pos[2];
    funcPOS(i, s, primes, k_pos);
    sfixn k = k_pos[0];
    sfixn pos  = i % k_pos[1];
    sfixn j = k+1+pos;

    D[i] = inv_mod(prime[k], prime[j]);
    L[i] = prime[j] - D[i];
  }
}


__global__ void copyX(sfixn *X, sfixn *Polynomial_shift, sfixn n, sfixn step)
{
  sfixn i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
    X[i * (s-1) + step] = Polynomial_shift[i];
}


__global__ void recombine()
{
    sfixn I = blockIdx.x * blockDim.x + threadIdx.x;
}

void chinese_reminder(sfixn *X, sfixn *D, sfixn *L, sfixn n, sfixn e, sfixn s)
{
  sfixn i, j;
  sfixn temp;
  sfixn *D, *L;
  sfixn *X;
  sfixn *Temp_host, *Temp_device;

  Temp_host = (sfixn *) malloc(n * sizeof(sfixn));

  cudaMalloc( (void **) &X, n*(s-1)   * sizeof(sfixn) );
  cudaMalloc( (void **) &Temp_device, n * sizeof(sfixn) );

  nb_blocks = number_of_blocks(n);
  for (j=0; j<s-1; j++)
  {
    sprintf(filename, "Pol%dshiftGPU_%d.dat\0", e, primes[j]);
    stock_file_in_array(filename, n, Temp_host);
    cudaMemcpy( Temp_device, Temp_host, n*sizeof(sfixn), cudaMemcpyHostToDevice );
    copyX<<<nb_blocks, NB_THREADS>>>(X, Temp_device, n, j);
  }
  free(Temp_host);

  nb_blocks = number_of_blocks(s*(s-1)/2);
  createDL<<<nb_blocks, NB_THREADS>>>(D, L, primes, s);


  cudaMalloc( (void **) &D, s*(s-1)/2 * sizeof(sfixn) );
  cudaMalloc( (void **) &L, s*(s-1)/2 * sizeof(sfixn) );

  for (i=0; i<n; i++)
    for (j=k+1; j<s; j++)
    {
      temp = X[];
    }


}

/* test.cu */

// Libraries :
# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <time.h>
# include <ctime>
# include <math.h>
# include <unistd.h>
# include <iostream>
# include <fstream>
using namespace std;

#define NB_THREADS 512


int number_of_blocks(int n)
{
  int res;
  res = n/NB_THREADS;
  if ( n % NB_THREADS != 0)
    res++;
  return res;
}


__global__ void ker(int *T, int n)
{
  int k = blockIdx.x * blockDim.x + threadIdx.x;

  if (k < n)
    T[k] = k;
}


int main()
{
  int n = 512;
  int e = 9;
  int i, nb_blocks;
  int *Td, *Th;

  for (i=0; i<100; i++)
  {
  printf("n = %d\ne = %d\n", n, e);
  Th = (int*) calloc(n, sizeof(int));
  printf("      --> Th calloc done\n");
  cudaMalloc( (void **) &Td, n * sizeof(int) );
  cudaThreadSynchronize();
  printf("      --> Td cudaMalloc done\n");
  cudaMemcpy( Td, Th, n*sizeof(int), cudaMemcpyHostToDevice );
  cudaThreadSynchronize();
  printf("      --> cudaMemcpy(Td, Th) done\n");
  nb_blocks = number_of_blocks(n);
  ker<<<nb_blocks, NB_THREADS>>>(Td, n);
  printf("      --> ker(Td) done\n");
  cudaMemcpy( Th, Td, n*sizeof(int), cudaMemcpyDeviceToHost );
  cudaThreadSynchronize();
  printf("      --> cudaMemcpy(Th, Td) done\n");
  free(Th);
  cudaFree(Td);
  n *= 2;
  e++;
  }

}

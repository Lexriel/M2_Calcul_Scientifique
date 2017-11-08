// Libraries :
# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <time.h>
# include <math.h>
# include <unistd.h>
# include <iostream>
# include <fstream>
using namespace std;

// display of an array
void display_array(int *T, int size)
{
  int k;
  printf("[ ");
  for (k=0; k<size; k++)
    printf("%d ", T[k]);
  printf("] \n");
}

// procedure
__global__ void array_thread_id(int *T, int n)
{
  int i = threadIdx.x;
  if (i < n)
    T[i] = i;
}

// main
int main(int argc, char* argv[])
{
  int i;
  int e = 13;
  int *power2, *power2_device;
  int *T, *T_device;
  int *temp;
  cudaError_t error;

  // allocations
  power2 = (int*) malloc(e*sizeof(int));
  T = (int*) malloc(e*sizeof(int));
  temp = (int*) malloc(e*sizeof(int));
  cudaMalloc( (void **) &power2_device, e * sizeof(int) );
  cudaThreadSynchronize();
  cudaMalloc( (void **) &T_device, e * sizeof(int) );
  cudaThreadSynchronize();

  // calculation of the power of 2
  power2[0] = 1;
  for (i=1; i<e; i++)
    power2[i] = 2 * power2[i-1];

  // copy power2 on power2_device then power2_device on temp
  error = cudaMemcpy( power2_device, power2, e*sizeof(int), cudaMemcpyHostToDevice );
  cudaThreadSynchronize();
  printf("error status: %s\n", cudaGetErrorString(error));

  error = cudaMemcpy( temp, power2_device, e*sizeof(int), cudaMemcpyDeviceToHost );
  cudaThreadSynchronize();
  printf("error status: %s\n", cudaGetErrorString(error));

  // call procedure to fill T_device then copy it on T
  array_thread_id<<<1,e>>>(T_device, e);
  cudaThreadSynchronize();
  error = cudaMemcpy( T, T_device, e*sizeof(int), cudaMemcpyDeviceToHost );
  cudaThreadSynchronize();
  printf("error status: %s\n", cudaGetErrorString(error));

  // display of the arrays
  printf("\npower2 :\n");
  display_array(power2, e);
  printf("\npower2_device (copied in temp) :\n");
  display_array(temp, e);
  printf("\nT_device (copied in T) :\n");
  display_array(T, e);

  // deallocations
  free(T);
  free(temp);
  free(power2);
  cudaFree(power2_device);
  cudaFree(T_device);

}

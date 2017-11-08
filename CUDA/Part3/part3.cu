/* part3.cu */

#include "part3_conf.h"
#include "part3_cpu.h"
#include "part3_kernel.cu"


/* ====================== Main =========================== */

int main(int argc, char* argv[])
{
  int *A, *A_device, *A_reverse_device, *A_aux;
  int block_number;

  A = (int*) malloc(N*sizeof(int));
  A_aux = (int*) malloc(N*sizeof(int));
  cudaMalloc( (void **) &A_device, N*sizeof(int) );
  cudaMalloc( (void **) &A_reverse_device, N*sizeof(int) );

// Nombre de blocs à utiliser :
  block_number = N/BLOCK_SIZE;
  if ( (N % BLOCK_SIZE) != 0 )
    block_number++;

// On crée A et on le stocke directement dans A_device :
  init_tab(A, N);
  //printf("A = [ "); display_tab(A, N);
  cudaMemcpy( A_device, A, N*sizeof(int), cudaMemcpyHostToDevice );

// On inverse A sur le CPU :
  reverse_array(A, N);
  //printf("A_inv = [ "); display_tab(A, N);

// On inverse A sur le GPU :
  _reverse_array<<<block_number, BLOCK_SIZE>>>(A_device, N, A_reverse_device);
  cudaMemcpy( A_aux, A_reverse_device, N*sizeof(int), cudaMemcpyDeviceToHost );
  //printf("A_aux = [ "); display_tab(A_aux, N);

// test d'égalité de tableaux :
  is_that_equal(A, A_aux, N);

// On inverse A sur le GPU avec la mémoire partagée :
  _reverse_array2<<<block_number, BLOCK_SIZE>>>(A_device, N, A_reverse_device);
  cudaMemcpy( A_aux, A_reverse_device, N*sizeof(int), cudaMemcpyDeviceToHost );
  is_that_equal(A, A_aux, N);

}

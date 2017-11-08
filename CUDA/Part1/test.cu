/* test.cu */

# include <stdio.h>
# include <string.h>
# include "test_conf.h"
# include "test_cpu.h"
# include "test_kernel.cu"

/* ================ Procedure_gpu =================== */
//* hote cpu

void procedure_cpu_gpu(int size)
{
  int *T, *T_device, *T_aux;
  int i, block_number;

  T  = (int*) malloc(size*sizeof(int));
  T_aux = (int*) malloc(size*sizeof(int));

  for (i=0; i<size; i++)
    T[i] = i;

  cudaMalloc( (void**) &T_device, size*sizeof(int) );
  cudaMemcpy( T_device, T, size*sizeof(int), cudaMemcpyHostToDevice );

// appel du kernel !


  cudaMemcpy( T_aux, T_device, size*sizeof(int), cudaMemcpyDeviceToHost );

  printf("T_aux = [ ");
  display_tab(T_aux, size);
  printf("]\n");


  /*kernel_1<<< 1, 1 >>> (T_device);

  cudaMemcpy( T_aux, T_device, size*sizeof(int), cudaMemcpyDeviceToHost );

  printf("T_aux = [ ");
  display_tab(T_aux, size);
  printf("]\n");
*/

  block_number = size/BLOCK_SIZE;
  if ( (size % BLOCK_SIZE) != 0 )
    block_number++;

//10000 elements
//256 threads per block
//39+1 blocks 
//40*256 = 10240 threads

  inc_gpu<<< block_number, BLOCK_SIZE >>> (T_device, size);

  cudaMemcpy( T_aux, T_device, size*sizeof(int), cudaMemcpyDeviceToHost );

  printf("T_aux = [ ");
  display_tab(T_aux, size);
  printf("]\n");

  cudaFree(T_device);
  free(T);
  free(T_aux);
}


/* ===================== Main ======================== */

int main (int argc, char* argv[]) 
{
  int size = atoi(argv[1]);


  // procedure_cpu(size);
  // procedure_cpu2(size);
 procedure_cpu_gpu(size);
}



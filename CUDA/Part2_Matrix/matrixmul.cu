/* matrixmul.cu */

#include "matrixmul_conf.h"
#include "matrixmul_cpu.h"
#include "matrixmul_kernel.cu"


/* ================== BuildMatrix ======================== */

void BuildMatrix(int m, int n, float *T)
{
  int i;
  for (i=0; i<m*n; i++)
    T[i] = (float)rand()/RAND_MAX;
}


/* ====================== Main =========================== */

int main(int argc, char* argv[])
{
  float *A, *B, *C, *C_aux, *A_device, *B_device, *C_device;
  dim3 thread_block (BLOCK_SIZE, BLOCK_SIZE);
  dim3 block (WC/BLOCK_SIZE, WC/BLOCK_SIZE);

// Allocation de mémoire :
  A = (float*) malloc(HA * WA * sizeof(float));
  B = (float*) malloc(HB * WB * sizeof(float));
  C = (float*) malloc(HC * WC * sizeof(float));
  C_aux = (float*) malloc(HC * WC * sizeof(float));
  cudaMalloc( (void**) &A_device, HA * WA * sizeof(float) );
  cudaMalloc( (void**) &B_device, HB * WB * sizeof(float) );
  cudaMalloc( (void**) &C_device, HC * WC * sizeof(float) );
  srand(time(NULL));

// Construction et affichage de A et de B :
  BuildMatrix(HA, WA, A);
  //printf("A = [ "); display_tab(A, HA*WA);
  BuildMatrix(HB, WB, B);
  //printf("B = [ "); display_tab(B, HB*WB);

// Copie de A et de B dans A_device et B_device :
  cudaMemcpy( A_device, A, HA*WA*sizeof(float), cudaMemcpyHostToDevice ); // on copie A dans A_device
  cudaMemcpy( B_device, B, HB*WB*sizeof(float), cudaMemcpyHostToDevice );
  
// Produit matriciel par le CPU :
  matmul(A, B, C);
  //printf("C = [ "); display_tab(C, HC*WC);

// Produit matriciel par le GPU :
  _matmul<<<block, thread_block>>>(A_device, B_device, C_device);
  cudaMemcpy( C_aux, C_device, HC*WC*sizeof(float), cudaMemcpyDeviceToHost );
  //printf("C_aux = [ "); display_tab(C_aux, HC*WC);

// Test sur l'égalité de C et C_aux :
  is_that_equal(C, C_aux, HC*WC);
  
// Déallocation de mémoire :
  free(A);
  free(B);
  free(C);
  free(C_aux);
  cudaFree(A_device);
  cudaFree(B_device);
  cudaFree(C_device);
}



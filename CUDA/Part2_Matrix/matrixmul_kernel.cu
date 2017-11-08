/* ========================================================

                Fonctions propres au GPU 

   ======================================================== */

// Effectue le produit matriciel de A par B pour obtenir C :
__global__ void _matmul(float *A, float *B, float *C)
{
  int i, j, k, l;

  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;

    l = i*WC + j;
    C[l] = 0;
    for (k=0; k<WC; k++)
      C[l] = C[l] + A[i*WC+k]*B[k*WC+j];
}

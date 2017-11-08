/* ========================================================

                Fonctions propres au GPU 

   ======================================================== */



__global__ void _reverse_array(int *A_in, int n, int *A_out)
{
  int i;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  A_out[i] = A_in[n-1-i];
}

/*
nb_blocks = N/BLOCK_SIZE;
nb_threads_per_block = BLOCK_SIZE
nb_threads = N;

UN TABLEAU BLOCK_SIZE par block
N/BLOCK_SIZE blocks 
*/

__global__ void _reverse_array2(int *A_in, int n, int *A_out)
{
  __shared__ int s[BLOCK_SIZE]; //tableau propre à chaque thread
  int i;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  s[threadIdx.x] = A_in[n-1-i];
  __syncthreads();
  A_out[i] = s[threadIdx.x];
}



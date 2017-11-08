__device__ int _logarithm_2(int m)
{
  int count = 0;
  while (m != 1)
  {
    m = m/2;
    count++;
  }
  return count;
}

__global__ void kernel_compute_gpu(int n, int nb_iters, int *T)
{
  int iter;
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n)
    for(iter=0; iter<nb_iters; iter++)
      T[id] = _logarithm_2(T[id]*T[id]+T[id]+iter);
}

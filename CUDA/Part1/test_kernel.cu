/* test_kernel.cu
  it does not contain anything for the moment
device AJOUTER +1 en parallèle à chaque élément du tableau
*/

//kernel !

__global__ void kernel_1(int* T_device)
{
  T_device[0] += 1;
}

__global__ void inc_gpu(int* a, int n)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n)
    a[id]++;
}

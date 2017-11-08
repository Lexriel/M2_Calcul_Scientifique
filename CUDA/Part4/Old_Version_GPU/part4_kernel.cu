__device__ int f_gpu(int k, int n)
{
  int i, M;
  M = 0;
  for (i=0; i<k; i++)
    M = M + (n-i-1);
  return M;
}


__device__ int compute_delta_gpu(int* a, int* b, int* p, int i, int j, int n)
{
  int d; int k;
  
  d = ( a[i*n+i] - a[j*n+j] ) * ( b[ p[j]*n + p[j] ] - b[ p[i]*n + p[i] ] ) +
    ( a[i*n+j] - a[j*n+i] ) * ( b[ p[j]*n + p[i] ] - b[ p[i]*n + p[j] ] );
  
  for (k=0; k<n; k++)
    if (k != i && k != j)
      d = d +( a[k*n+i] - a[k*n+j] ) * ( b[ p[k]*n + p[j] ] - b[ p[k]*n + p[i] ] ) +
	( a[i*n+k] - a[j*n+k] ) * ( b[ p[j]*n + p[k] ] - b[ p[i]*n + p[k] ] );
  
  return d;
}


__global__ void main_gpu(int *voisin_device, int *a_device, int *b_device, int *solution_device, int n)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x; // I = id
  int k;
  int i,j;

  if (id < n*(n-1)/2)
  {
    // définit i et j à partir de I
  k = 0;
  while ( id >= f_gpu(k,n) )
    k++;
  k--;
  
    i = k; 
    j = id - f_gpu(k,n) + k + 1;

    // calcul le décalage d'un voisin et le place dans le tableau voisin_device
    voisin_device[id] = compute_delta_gpu(a_device, b_device, solution_device, i, j, n);
  }
}

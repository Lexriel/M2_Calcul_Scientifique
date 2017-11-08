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
  int I = blockIdx.x * blockDim.x + threadIdx.x;
  int k = 0;
  int temp = 0;
  int prec = 0;
  int i, j;

  if (I < n*(n-1)/2)
{
  while (I >= temp)
    {
      k++;
      prec = temp;
      temp = temp + (n-k);
    }
  k--;

// on prend l'étape précédente, on est allé un cran trop haut
  i = k; 
  j = I - prec + k + 1;

  // calcul le décalage d'un voisin et le place dans le tableau voisin_device
  voisin_device[I] = compute_delta_gpu(a_device, b_device, solution_device, i, j, n);
}
}

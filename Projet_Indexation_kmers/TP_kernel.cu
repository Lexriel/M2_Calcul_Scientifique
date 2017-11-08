// Procédure de recherche de kmers sur le GPU
__global__ void find_kmers_GPU(unsigned long* input, unsigned long* s, unsigned long* s2, unsigned long n, unsigned long NB_BLOCK_MAX)
{
  unsigned long J = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long i;
  unsigned long K = J*NB_BLOCK_MAX;

  if (K < n)
    {
      // additionne les minimum de chaque thread en s[K]
      if (input[K] < s[K])
        s[K] = input[K];
      for(i=1; i<NB_BLOCK_MAX; i++)
        {
          if (input[K+i] < s[K+i])
            s[K] += input[K+i];
          else
            s[K] += s[K+i];
        }
    }
  __syncthreads();


  // créaction d'un tableau s2 simple à sommer
  if (J<n/NB_BLOCK_MAX)
    s2[J] = s[K];

  // somme
  for (i=n/(2*NB_BLOCK_MAX); i>0; i=i/2)
    {
      __syncthreads();
      if (J < i)
        s2[J] += s2[J+i];
    }

  s[0] = s2[0];

}


// Fonction de calcul du code sur le GPU
__device__ unsigned long code_GPU(char *w, unsigned long n)
{
  int wi = 0;
  unsigned long i;
  unsigned long result = 0;
  unsigned long power_4_i = 1; // 4^i

  for(i=0; i<n; i++)
  {
    if (w[i] == 'A')
      wi = 0;
    if (w[i] == 'C')
      wi = 1;
    if (w[i] == 'T')
      wi = 2;
    if (w[i] == 'G')
      wi = 3;
    result = result + wi * power_4_i;
    power_4_i = power_4_i * 4;
    }

  return result;
}


// Procédure de création de l'index de la séquence de référence sur le GPU
__global__ void creation_index_GPU(char *chaine, unsigned long* temp_code, unsigned long size, unsigned long* index, unsigned long nb_kmers, unsigned long k)
{
  int J = blockIdx.x * blockDim.x + threadIdx.x;

  if (J<size-k)
    index[code_GPU(chaine+J, k)]++;

}

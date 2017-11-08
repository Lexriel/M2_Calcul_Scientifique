int logarithm_2(int m)
{
  int count = 0;
  while (m != 1)
  {
    m = m/2;
    count++;
  }
  return count;
}

void compute_cpu(int n,int nb_iters)
{
 int i;
 int* T  = (int*) malloc(n*sizeof(int));
  for(i=0; i<n; i++)
    T[i]  = rand() % (nb_iters-1) + 1;

/*  printf("T = [ ");
  for (i=0; i<n; i++)
    printf("%d ", T[i]);
  printf("]\n"); */

int id, iter;
  for(id=0; id<n; id++)
    for(iter=0; iter<nb_iters; iter++)
      T[id] = logarithm_2(T[id]*T[id]+T[id]+iter);


/*  printf("Avec compute_cpu : T = [ ");
  for (i=0; i<n; i++)
    printf("%d ", T[i]);
  printf("]\n"); */


free(T);


}



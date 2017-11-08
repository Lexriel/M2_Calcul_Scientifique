/* part4.cu */

#include "part4_conf.h"
#include "part4_cpu.h"
#include "part4_kernel.cu"


/* ====================== Main_CPU =========================== */

int main(int argc, char* argv[])
{
  int *a, *a_device, *b, *b_device, *solution, *solution_device, *voisin, *voisin_device, *ij;
  int score, n, temp, i, m, condition, nb_blocks;
  int iter = 0;
  srand(time(NULL));

  if (argc<2)
    {
      printf("Please give a dat file\n");
      exit(1);
    }

  loadInstances(argv[1],n,a,b);

  m = n*(n-1)/2; // taille du tableau voisin
  nb_blocks = m/NB_THREAD; // nombre de blocs
  if ( m % NB_THREAD != 0)
    nb_blocks++;

  solution = (int*) malloc(n*sizeof(int));
  voisin = (int*) malloc(n*(n-1)/2*sizeof(int));
  ij = (int*) malloc(3*sizeof(int));

  cudaMalloc( (void **) &a_device, n*n*sizeof(int) );
  cudaMalloc( (void **) &b_device, n*n*sizeof(int) );
  cudaMalloc( (void **) &solution_device, n*sizeof(int) );
  cudaMalloc( (void **) &voisin_device, m*sizeof(int) );
  
  cudaMemcpy( a_device, a, n*n*sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy( b_device, b, n*n*sizeof(int), cudaMemcpyHostToDevice );

  create(solution,n);
  score = evaluation(a,b,solution,n);

  printf("score début = %d \n", score);
  printf("solution début = ");
  for (i=0; i<n; i++)
    printf("%d ", solution[i]);
  printf("\n\n");


  free(a);
  free(b);


condition = 0;

while ( condition == 0 )
    {

      cudaMemcpy( solution_device, solution, n*sizeof(int), cudaMemcpyHostToDevice );

      main_gpu<<<nb_blocks, NB_THREAD>>>(voisin_device, a_device, b_device, solution_device, n);

      cudaMemcpy( voisin, voisin_device, m*sizeof(int), cudaMemcpyDeviceToHost );
 
//      printf("voisin = ");
//      for (i=0; i<n; i++)
//        printf("%d ", voisin[i]);
//      printf("\n\n");


      min_tab(ij, voisin, m, n, condition); // donne le min du tableau voisin et les éléments i et j à permuter

      temp = solution[ij[0]];
      solution[ij[0]] = solution[ij[1]];
      solution[ij[1]] = temp;

      score = score + ij[2];
      iter++;

//      printf("%d\n", ij[2]);  printf("%d\n", ij[0]);  printf("%d\n", ij[1]);
    }


      printf("Le meilleur score, trouvé en %d itérations, est donnée par :\n", iter);
      printf("z(pi) = %d \n", score);
      printf("pi = [ ");
      for (m=0; m<n; m++)
        printf("%d ", solution[m]);
      printf("] \n\n");

      cudaFree(a_device);
      cudaFree(b_device);
      cudaFree(solution_device);
      cudaFree(voisin_device);
      free(ij);
      free(solution);
      free(voisin);

}

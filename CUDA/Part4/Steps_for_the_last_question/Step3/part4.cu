/* part4.cu */

#include "part4_conf.h"
#include "part4_cpu.h"
#include "part4_kernel.cu"


/* ====================== Main_CPU =========================== */

int main(int argc, char* argv[])
{
  int *a, *a_device, *b, *b_device;
  int *solution, *solution_device, *meilleure_solution, *voisin, *voisin_device, *ij;
  int score, best_score, n, temp, m, condition, nb_blocks, k, nb_solution;
  srand(time(NULL));

  if (argc<3)
    {
      printf("Please give a dat file in argument 1 and the number of iterations in argument 2.\n");
      exit(1);
    }

  loadInstances(argv[1],n,a,b);
  nb_solution = atoi(argv[2]);

  m = n*(n-1)/2;             // taille du tableau voisin
  nb_blocks = m/NB_THREAD;   // nombre de blocs
  if ( m % NB_THREAD != 0)
    nb_blocks++;

// Allocations dynamiques
  solution = (int*) malloc(n*sizeof(int));
  meilleure_solution = (int*) malloc(n*sizeof(int));
  voisin = (int*) malloc(n*(n-1)/2*sizeof(int));
  ij = (int*) malloc(3*sizeof(int));

  cudaMalloc( (void **) &a_device, n*n*sizeof(int) );
  cudaMalloc( (void **) &b_device, n*n*sizeof(int) );
  cudaMalloc( (void **) &solution_device, n*sizeof(int) );
  cudaMalloc( (void **) &voisin_device, m*sizeof(int) );
  
  cudaMemcpy( a_device, a, n*n*sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy( b_device, b, n*n*sizeof(int), cudaMemcpyHostToDevice );


/* ==================================================================
   =                                                                =
   =                          Multistart :                          =
   =                                                                =
   =         on lance le hill-climbing nb_solution fois             =
   =         (rentré en ligne de commande)                          =
   =                                                                =
   ================================================================== */

  for (k=0; k<nb_solution; k++)
    {

      create(solution,n);                   // génère une solution pi
      score = evaluation(a,b,solution,n);   // évalue le score z(pi)

      condition = 0;                        // booléen conditionnel pour la boucle


// Recherche d'une meilleure solution tant qu'un voisin en propose une

      while ( condition == 0 )
        {

          cudaMemcpy( solution_device, solution, n*sizeof(int), cudaMemcpyHostToDevice );

          // le gpu calcule le décalage d'un voisin
          main_gpu<<<nb_blocks, NB_THREAD>>>(voisin_device, a_device, b_device, solution_device, n);

          cudaMemcpy( voisin, voisin_device, m*sizeof(int), cudaMemcpyDeviceToHost );
 
          // le cpu définit les éléments de solution à permuter pour avoir un meilleur score
          min_tab(ij, voisin, m, n, condition);

          // on permute les éléments trouvés
          temp = solution[ij[0]];
          solution[ij[0]] = solution[ij[1]];
          solution[ij[1]] = temp;

          // on calcule le nouveau score
          score = score + ij[2];

        }

      // initialisation de la meilleure solution et du meilleur score à l'étape k=0
      if (k == 0)
        {
          memcpy(meilleure_solution, solution, n*sizeof(int));
          best_score = score;
        }

      // crée la meilleure solution et le meilleur score si il en a un
      if ( (k != 0) && (score < best_score) )
        {
          memcpy(meilleure_solution, solution, n*sizeof(int));
          best_score = score;
        }

    }

  // affichage des résultats finaux
  printf("Le meilleur score trouvé par les Hill-climbing par :\n");
  printf("z(pi) = %d \n", best_score);
  printf("pi = [ ");
  for (m=0; m<n; m++)
    printf("%d ", meilleure_solution[m]);
  printf("] \n\n");

  // désallocation des tableaux
  cudaFree(a_device);
  cudaFree(b_device);
  cudaFree(solution_device);
  cudaFree(voisin_device);
  free(a);
  free(b);
  free(ij);
  free(solution);
  free(voisin);
  free(meilleure_solution);

}

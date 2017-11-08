/* part4.cu */

#include "part4_conf.h"
#include "part4_cpu.h"
#include "part4_kernel.cu"


/* ====================== Main_CPU =========================== */

int main(int argc, char* argv[])
{
  int *a, *a_device, *b, *b_device;
  int *solution, *solution_device, *meilleure_solution;
  int *voisin, *voisin_device, *ij;
  int i, j, k, score, best_score;
  int n, temp, m, condition, nb_blocks, nb_solution;
  int seed = time(NULL);

  // données temporelles
  clock_t initial_time;   /* Initial time in micro-seconds */
  clock_t final_time;     /* Final time in micro-seconds */
  float cpu_time;         /* Total time of the cpu in seconds */
  float gpu_time;         /* Total time of the gpu in seconds */ 

  if (argc < 3)
    {
      printf("Please give a data file in argument 1 and the number of iterations in argument 2.\n");
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
   =                      Multistart CPU :                          =
   =                                                                =
   =         on lance le hill-climbing nb_solution fois             =
   =         (rentré en ligne de commande)                          =
   =                                                                =
   ================================================================== */

  /* ces instructions sont les mêmes que dans pour le multistart du
     gpu donc je me permets de ne pas les commenter.*/

  initial_time = clock();
  srand(seed);


  for (k=0; k<nb_solution; k++)
    {

      create(solution,n);                  
      score = evaluation(a,b,solution,n);   

      condition = 0;


      while ( condition == 0 )
        {

          ij[0] = 0;
          ij[1] = 1;
          ij[2] = 1;

          for (i=0; i<(n-1); i++)
            {
              for (j=i+1; j<n; j++)
                {
                  temp = compute_delta_cpu(a, b, solution, i, j, n);
         
                  if (temp < ij[0])
                    {
                      ij[0] = temp;
                      ij[1] = i;
                      ij[2] = j;
                    }

                }
            }

          if (ij[0] >= 0)
            condition = 1;

    
          temp = solution[ij[1]];
          solution[ij[1]] = solution[ij[2]];
          solution[ij[2]] = temp;

          score = score + ij[0];
        }
 

      if ( (k == 0) || ( (k != 0) && (score < best_score) ) )
        {
          memcpy(meilleure_solution, solution, n*sizeof(int));
          best_score = score;
        }

    }


  final_time = clock();
  cpu_time = (final_time - initial_time)*1e-6;

  // affichage des résultats finaux sur CPU
  printf("Le meilleur score trouvé par les Hill-climbing avec le CPU est :\n");
  printf("z(pi) = %d \n", best_score);
  printf("pi = [ ");
  for (k=0; k<n; k++)
    printf("%d ", meilleure_solution[k]);
  printf("] \n");
  printf("Temps d'exécution CPU : %f s\n\n", cpu_time);



/* ==================================================================
   =                                                                =
   =                      Multistart GPU :                          =
   =                                                                =
   =         on lance le hill-climbing nb_solution fois             =
   =         (rentré en ligne de commande)                          =
   =                                                                =
   ================================================================== */

  initial_time = clock();
  srand(seed);

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

  final_time = clock();
  gpu_time = (final_time - initial_time)*1e-6;


  printf("Le meilleur score trouvé par les Hill-climbing avec le GPU est :\n");
  printf("z(pi) = %d \n", best_score);
  printf("pi = [ ");
  for (k=0; k<n; k++)
    printf("%d ", meilleure_solution[k]);
  printf("] \n");
  printf("Temps d'exécution GPU : %f s\n\n", gpu_time);


/* ======================================================
   =                                                    =
   =          fin de ce merveilleux programme           =
   =                                                    =
   ====================================================== */

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

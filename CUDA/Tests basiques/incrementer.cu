# include <stdio.h>
# include <string.h>

# define BLOCK_SIZE 256

/* ======================================= */
/* ------------ FONCTIONS CPU ------------ */
/* ======================================= */

// affiche tableau
void afficher_tableau(int* T, int size)
{
  int i;

  for (i=0; i<size; i++)
    printf("%d,", T[i]);
  printf("\n");
}

// Cette fonction incrémente chaque case du tableau sur CPU
void incrementer_CPU(int *a, int n)
{
  int i;
  for (i=0; i<n; i++)
    a[i]++;
}

/* ======================================= */
/* ------------ FONCTIONS GPU ------------ */
/* ======================================= */

// Cette fonction incrémente chaque case du tableau sur GPU
__global__ void incrementer_GPU(int* a, int n)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n)
      a[id]++;
}

/* ======================================= */
/* ---------- FONCTIONS CPU-GPU ---------- */
/* ======================================= */

// Cette fonction affiche le tableau après incrémentation sur CPU
void procedure_CPU_GPU(int size)
{
  int *T, *T_device, *T_aux;
  int i, block_number;

  // Allocations mémoire
  T  = (int*) malloc(size*sizeof(int));
  T_aux = (int*) malloc(size*sizeof(int));
  cudaMalloc( (void**) &T_device, size*sizeof(int) );

  // Affectation du tableau T, affichage
  for (i=0; i<size; i++)
    T[i] = i;
  printf("T = ");
  afficher_tableau(T, size);

  // Copie sur le GPU, affichage
  printf("\nOn copie T sur T_device dans le GPU.\n");
  printf("Une recopie de T_device sur T_aux dans le CPU nous permet de retourner le contenu de T_device :\n\n");
  cudaMemcpy( T_device, T, size*sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy( T_aux, T_device, size*sizeof(int), cudaMemcpyDeviceToHost );  
  printf("T_aux = ");
  afficher_tableau(T, size);

  // Incrémentation sur le CPU, affichage
  printf("\nSi on incrémente les valeurs des cases de T sur le CPU :\n");
  incrementer_CPU(T,size);
  printf("T = ");
  afficher_tableau(T, size);

  // Calcul du nombre de blocks nécessaires, affichage
  block_number = size/BLOCK_SIZE;
  if ( (size % BLOCK_SIZE) != 0 )
    block_number++;
  printf("\nNombre de blocks  : %d\n", block_number);
  printf("Taille des blocks : %d threads\n", BLOCK_SIZE);
  block_number = size/BLOCK_SIZE;
  if ( (size % BLOCK_SIZE) != 0 )
    block_number++;

  // Incrémentation sur le GPU, affichage
  printf("\nSi on incrémente les valeurs des cases de T sur le GPU :\n");
  printf("valeur = %d\n", block_number);
  incrementer_GPU<<< block_number, BLOCK_SIZE >>> (T_device, size);
  cudaMemcpy( T_aux, T_device, size*sizeof(int), cudaMemcpyDeviceToHost );
  printf("T_aux = ");
  afficher_tableau(T_aux, size);

  cudaFree(T_device);
  free(T);
  free(T_aux);
}


/* ======================================= */
/* ---------------- MAIN ----------------- */
/* ======================================= */

int main(int argc, char* argv[])
{
  int size;
  if (argc < 2)
    {
      printf("Missing parameter: './incrementer 32' (where 32 is the number of threads used).\n");
      exit(1);
    }

  printf("1/ Sur le CPU, un tableau T est créé.\n");
  printf("2/ T est copié sur le GPU dans T_device.\n");
  printf("3/ Chaque thread incrémente un coefficient de T_device, qui est donc modifié.\n");
  printf("4/ T_device est envoyé sur le CPU dans T_aux.\n");
  printf("Le code s'effectue correctement si CUDA est bien paramétré.\n\n");

  size = atoi(argv[1]);
  procedure_CPU_GPU(size);

  return 0;
}

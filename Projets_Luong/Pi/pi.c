# include <stdio.h>
# include <stdlib.h>
# include <mpi.h>


// Fonction puissance pour des nombres entiers
int power(int a, int b)
{
  int i;
  int res = 1;

  if (b != 0)
    {
      for (i=0; i<b; i++)
	res = res*a;
    }
  return res;
}


// Fonction apparaissant dans la somme
double f(int n)
{
  double sg = -1;
  if (n % 2 == 0)
    sg = 1;
  return 4 * sg / ((double) (2*n+1));
}

/* =========================== Main =========================== */

int main(int argc,char* argv[])
{
  int rank, P, i, precision, nb_termes;
  long long int a, b;
  char c;
  double partial_sum, total_sum, temp;
  FILE *fp;

  MPI_Init(NULL,NULL);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &P);

  partial_sum = 0;
  total_sum = 0;
  precision = atoi(argv[1]) + 1;
  nb_termes = 2*power(10, precision);

// Chaque processeur effectue une partie de la somme de a à b
  a =   (long long int) rank   * nb_termes / P;
  b = (long long int) (rank+1) * nb_termes / P;

  for (i = a; i<b; i++)
    partial_sum = partial_sum + f(i);

// On additionne toutes les sommes partielles pour avoir la somme totale qu'est Pi dans le processeur de rang 0
  MPI_Reduce(&partial_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


/* Le processeur de rang 0 affiche les 10 premières décimales de pi à
   l'écran puis enregistre pi dans le fichier pi.txt. */
  if (rank == 0)
    {
      printf("An approximation of Pi with %d decimals is : %0.12lf\n", precision-1, total_sum);

      fp = fopen("pi.txt", "w");
      fprintf(fp, "An approximation of Pi with %d decimals is given by : \npi = 3.", precision-1);
      temp = 10*(total_sum - 3);
      for (i=1; i<precision; i++)
        {
          c = (int) temp;
          fprintf(fp, "%d", c);
          temp = 10*(temp - c);
        }
      fclose(fp);
    }

  MPI_Finalize();

  return 0;
}

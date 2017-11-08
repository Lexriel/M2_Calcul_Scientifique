# include <stdio.h>
# include <mpi.h>

int main()
{
  int source, cible, nb, recu;
  double debut, fin;
  MPI_Status status;
  MPI_Request request;

  MPI_Init(NULL,NULL);
  debut = MPI_Wtime();
  MPI_Comm_rank(MPI_COMM_WORLD, &source);
  MPI_Comm_size(MPI_COMM_WORLD, &nb);
  cible = (source + 1) % nb;
  MPI_Isend(&source, 1, MPI_INT,cible, 1, MPI_COMM_WORLD, &request);
  MPI_Recv(&recu, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
  fin = MPI_Wtime();
  printf("Le numéro %d a recu la valeur %d. Temps d'execution = %f \n", source, recu, fin-debut);
  MPI_Finalize();
  return 0;
}

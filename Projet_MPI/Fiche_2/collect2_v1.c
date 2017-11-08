# include <stdio.h>
# include <mpi.h>
# include <string.h>
# include <stdlib.h>
# include <time.h>
# define N 20


int main()
{
  int M, i, l, target, nb, rank;
  double begin, end;
  int tab[30], sub_tab[30];
  MPI_Request request;
  MPI_Status status;

  MPI_Init(NULL,NULL);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nb);

  begin = MPI_Wtime();

// M = length of the considered array to send
// l = length of the sub-arrays
    M = N; 
    while ( M % nb != 0)
      M++;
    l = M/nb;


/* ======================== processus of rank 0 ============================ */

if (rank == 0)
  {
    srand(time(NULL));

// Creation of the array tab :
    for (i=0; i<M; i++)
      {
        if (i<N)
          tab[i] = rand()%100;
        else
          tab[i] = 0;
      }

// Display of tab :
  printf("tab[%d] = [ ", M);
  for (i=0; i<M ; i++)
    printf(" %d ", tab[i]);
  printf("]\n\n");

// Sends a sub-array of tab to each process :
  for (target=0; target<nb; target++)
      MPI_Isend(&tab[target*l], l, MPI_INT, target, 2, MPI_COMM_WORLD, &request);
  }

/* ========================================================================= */


// Reception of a sub-array for each process :
  MPI_Recv(sub_tab, l, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);	      


// Display of the elements received by each process :
  printf("Processor %d : sub-array = [ ", rank);
  for (i=0; i<l ; i++)
    printf("%d ", sub_tab[i]);
  printf("]\n");


// Display of the execution time of each process : 
  end = MPI_Wtime();
  printf("Processor %d : execution time = %f.\n\n", rank, end-begin);


  MPI_Finalize();

  return 0;
}

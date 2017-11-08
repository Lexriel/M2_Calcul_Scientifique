# include <stdio.h>
# include <mpi.h>
# include <string.h>
# include <stdlib.h>
# include <time.h>
# define N 20


int main()
{
  int M, i, l, nb, rank;
  double begin, end;
  int tab[30], sub_tab[30];

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


// Process 0 creates the array tab and displays it :
if (rank == 0)
  {
    srand(time(NULL));
    for (i=0; i<M; i++)
      {
        if (i<N)
          tab[i] = rand()%100;
        else
          tab[i] = 0;
      }

  printf("tab[%d] = [ ", M);
  for (i=0; i<M ; i++)
    printf(" %d ", tab[i]);
  printf("]\n\n");
  }


// Process 0 sends tab to each process :
  MPI_Scatter(tab, l, MPI_INT, sub_tab, l, MPI_INT, 0, MPI_COMM_WORLD);


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

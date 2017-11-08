# include <stdio.h>
# include <mpi.h>
# include <string.h>
# include <stdlib.h>
# include <time.h>
# define N 20


int main()
{
  int M, i, l, nb, rank, partial_sum, total_sum;
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
  MPI_Scatter(tab, l, MPI_INT, &sub_tab, l, MPI_INT, 0, MPI_COMM_WORLD);


// Each process computes the sum of the elements of sub_tab it receives :
  partial_sum = 0;
  for (i=0; i<l; i++)
    partial_sum += sub_tab[i];


// Display of the elements received by each process and the partial sums :
  printf("Processor %d : sub-array = [ ", rank);
  for (i=0; i<l ; i++)
    printf("%d ", sub_tab[i]);
  printf("]\n");
  printf("Processor %d : partial_sum = %d.\n\n", rank, partial_sum);

  
// The process 0 receives all the partial sums and sums them :
  MPI_Reduce(&partial_sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);


// The process 0 displays the total sum of the elements of tab and displays the execution time of this program :
  if (rank == 0)
    {
      end = MPI_Wtime();
      printf("Processor 0 : total_sum = %d.\n", total_sum);
      printf("Execution time = %f.\n\n", end-begin);
    }

  MPI_Finalize();

  return 0;
}

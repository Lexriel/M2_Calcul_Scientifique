# include <stdio.h>
# include <mpi.h>
# include <string.h>
# include <stdlib.h>
# include <time.h>
# define N 10


int main()
{
  int i, nb, rank;
  double begin, end;
  int tab[N];
  int* big_tab;


  MPI_Init(NULL,NULL);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nb);
  big_tab = (int*) malloc(nb*N*sizeof(int));

  begin = MPI_Wtime();


// Allows to initialize the rand() function differently for each process :
  srand(time(NULL)+rank);


// Definition of a random array of N elements in each process :
  for (i=0; i<N; i++)
    tab[i] = rand()%100;


// Display of the elements of the array of each process :
  printf("Processor %d : tab = [ ", rank);
  for (i=0; i<N ; i++)
    printf("%d ", tab[i]);
  printf("]\n");


// Sends the array of each process to the process 0 and puts them into the array big_tab successively :
  MPI_Gather(tab, N, MPI_INT, big_tab, N, MPI_INT, 0,MPI_COMM_WORLD);


// Display of the array big_tab and the execution time of this programm :
  if (rank == 0)
    {
      printf("Processor 0 : big_tab = [ ");
      for (i=0; i<N*nb ; i++)
        printf("%d ", big_tab[i]);
      printf("]\n");
      end = MPI_Wtime();
      printf("Execution time = %f.\n\n", end-begin);
    }

  MPI_Finalize();

  return 0;
}

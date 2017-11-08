# include <stdio.h>
# include <mpi.h>
# include <string.h>


int main()
{
  int rank;
  double begin, end;
  char message[1000];

  MPI_Init(NULL,NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  begin = MPI_Wtime();


// Creation  of the message in the process 0 : 
  if (rank == 0)
     strcpy(message, "Bonjour, je vends des aspirateurs pas cher !");


// Sends the message to all the processes, then they receive the message :
  MPI_Bcast(message, 1000, MPI_CHAR, 0, MPI_COMM_WORLD);


// Displays the execution time and the message received by all the processes :
  if (rank != 0)
    {
      end = MPI_Wtime();
      printf(" * Peace message received by the processus %d: \"%s\"\n     Execution time of the processor %d = %f.\n\n", rank, message, rank, end-begin);
    }

  MPI_Finalize();
  return 0;
}

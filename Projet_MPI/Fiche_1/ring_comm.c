# include <mpi.h>
# include <string.h>
# include <stdio.h>
# include <stdlib.h>

int main()
{
  int rank, nb, k;
  double begin, end;
  MPI_Status status;
//  MPI_Request request;

  MPI_Init(NULL,NULL);
  begin = MPI_Wtime();

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nb);


/* ============================ Process of rank 0 ================================ */

  if (rank == 0)
    {
// Define the integer we want to put into circulation between processes :
      printf("Give me an integer and I will put it into circulation between processes, in order of their ranks : ");
      scanf("%d", &k);

// Sends this number to the process '1' :
      MPI_Send(&k, 1, MPI_INT, rank+1, rank, MPI_COMM_WORLD);

// Receives this number from the process 'nb-1' :
      MPI_Recv(&k, 1, MPI_INT, nb-1, nb-1, MPI_COMM_WORLD, &status);

// Displays the number received and the execution time of the process '0' :
      printf("The processor %d has received the number %d from the processor %d.\n", rank, k, nb-1);
      end = MPI_Wtime();
      printf("Execution time = %f.\n", end-begin);
    }

/* =============================================================================== */



/* ============ All the processes of rank different from 0 and nb-1 ============== */

  if ((rank != 0) && (rank != nb-1))
    {
// Receives the number sent by the process 'rank-1' :
      MPI_Recv(&k, 1, MPI_INT, rank-1, rank-1, MPI_COMM_WORLD, &status);
      printf("The processor %d has received the number %d from the processor %d.\n", rank, k, rank-1);

// Sends the number received to the process 'rank+1' :
      MPI_Send(&k, 1, MPI_INT, rank+1, rank, MPI_COMM_WORLD);
      printf("The processor %d has sent the number %d to the processor %d.\n", rank, k, rank+1);
    }

/* =============================================================================== */



/* =========================== Process of rank nb-1 ============================== */

  if (rank == nb-1)
    {
// Receives the number sent by the process 'nb-2' :
      MPI_Recv(&k, 1, MPI_INT, rank-1, rank-1, MPI_COMM_WORLD, &status);
      printf("The processor %d has received the number %d from the processor %d.\n", rank, k, rank-1);

// Sends the number received to the process '0' :
      MPI_Send(&k, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
      printf("The processor %d has sent the number %d to the processor 0.\n", rank, k);
    }

/* =============================================================================== */


  MPI_Finalize();
  return 0;
}

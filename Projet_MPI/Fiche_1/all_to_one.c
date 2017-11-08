# include <stdio.h>
# include <mpi.h>

int main()
{
  int i, rank, M, nb, M_received, sum;
  double begin, end;

  MPI_Status status;
  MPI_Request request;

  MPI_Init(NULL,NULL);
  begin = MPI_Wtime();

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nb);


/* ================ All the processes of rank different from 0 ================== */

  if (rank != 0)
    {
// Sends the number M to the process 0 :
      M = 1000*rank;
      MPI_Isend(&M, 1, MPI_INT, 0, 12, MPI_COMM_WORLD, &request);
      printf("The number %d was sent by %d.\n", M, rank);
    }

/* ============================================================================== */



/* =========================== Process of rank 0 ================================ */

  else
    {
/* Receives the number M from each process of rank different of 0, and compute
   their sum : */
      sum = 0;
      for (i=1; i<nb; i++)
      { 
        MPI_Recv(&M_received, 1, MPI_INT, i, 12, MPI_COMM_WORLD, &status);
        sum = sum + M_received;	      
      }

// Displays the sum of the numbers sended :
      printf("The sum of the numbers sended, computed by the processor 0, is: %d.\n", sum);
    }

/* ============================================================================== */


// Displays the execution time of each process :
  end = MPI_Wtime();
  printf("Execution time of the processor %d = %f.\n", rank, end-begin);

  MPI_Finalize();
  return 0;
}

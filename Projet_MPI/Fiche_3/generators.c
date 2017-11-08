# include <mpi.h>
# include <stdio.h>
# include <stdlib.h>

# define N 16  // matrix size

int main()
{
  int M, i, j, k, l, P, rank;
  int **Sub_Mat;
  MPI_Status status;
  MPI_Request request;

  MPI_Init(NULL,NULL);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &P);


// M = number of lines of the matrix such that M%P == 0. 
// l = number of lines in each sub-matrix
    M = N; 
    while ( M % P != 0)
      M++;
    l = M/P;


// Allocation of memory for Sub_Mat :
  Sub_Mat = (int**) malloc(l*sizeof(int*));
  for (i=0; i<l; i++)
    Sub_Mat[i] = (int*) malloc(M*sizeof(int));


// Definition of the sub-matrix :
for (j=0; j<rank*l; j++)
  Sub_Mat[0][j] = 0;

for (j=rank*l; j<N; j++) // Creation of the first line.
{
   if (j%2 == 0)
     Sub_Mat[0][j] = 1;
   else
     Sub_Mat[0][j] = 2;
}

for (i=1; i<l; i++) // Creation of the other lines by a recursive process.
{
  for (j=0; j<N ; j++)
    Sub_Mat[i][j] = Sub_Mat[i-1][j];

  Sub_Mat[i][rank*l-1+i] = 0; // We flip the first non-zero value of the line into a zero.
}


/* ====================== Display of the whole matrix ========================== */

  if (rank == 0)
    {
      k = 1;
      printf("Sub_Matrix of process %d :\n", rank);

      for (i=0; i<l; i++)
        {
          for (j=0; j<N; j++)
   	    printf("%d ", Sub_Mat[i][j]);

          printf("\n");
        }
      MPI_Isend(&k, 1, MPI_INT, rank+1, rank, MPI_COMM_WORLD, &request);
    }


/* ============ All the processes of rank different from 0 and P-1 ============== */

  if ((rank != 0) && (rank != P-1))
    {
      MPI_Recv(&k, 1, MPI_INT, rank-1, rank-1, MPI_COMM_WORLD, &status);
      printf("Sub_Matrix of process %d :\n", rank);

      for (i=0; i<l; i++)
        {
          for (j=0; j<N; j++)
   	    printf("%d ", Sub_Mat[i][j]);

          printf("\n"); 
        } 

      MPI_Send(&k, 1, MPI_INT, rank+1, rank, MPI_COMM_WORLD);

    }


/* =========================== Process of rank P-1 ============================== */

  if (rank == P-1)
    {
// Receives the number sent by the process 'P-2' :
      MPI_Recv(&k, 1, MPI_INT, rank-1, rank-1, MPI_COMM_WORLD, &status);
      printf("Sub_Matrix of process %d :\n", rank);

      for (i=0; i<l; i++)
        {
          for (j=0; j<N; j++)
   	    printf("%d ", Sub_Mat[i][j]);

          printf("\n");
        }
    }

/* =============================================================================== */


  MPI_Finalize();

  return 0;
}

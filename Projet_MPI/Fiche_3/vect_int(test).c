# include <mpi.h>
# include <stdio.h>
# include <stdlib.h>
# define M 4

// This file is just a test to see what it give if we consider
// a consider an array of integers.

void Build_Vector_1(int K, int rank, int *Vector)
{
  int line;
  for (line=0; line<K; line++)
//     Vector[line] = (rank + line)*2. ;
    Vector[line] = line + M*rank;
}


int main()
{
  int i, j, k, rank, P;
  double begin, end;
  int Vector[M];
  int *Big_Vector;

  MPI_Status status;

  MPI_Init(NULL,NULL);
  begin = MPI_Wtime();

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &P);

  Big_Vector = (int *) calloc(P*M,sizeof(int));


// Each process builds its own vector :
  Build_Vector_1(M, rank, Vector);

  for (i=0; i<M; i++)
    Big_Vector[rank*M+i] = Vector[i];
  printf("Big_Vector (process %d) : [ ", rank);
    for (i=0; i<M*P; i++)
      printf("%d ", Big_Vector[i]);
    printf("]\n");
    
    
    
    for (i=0; i<P; i++)
      {
	if (rank == i)
	  for (k=0; k<P; k++)
	    {
              if (k != rank)
                MPI_Send(Vector, M, MPI_INT, k, rank, MPI_COMM_WORLD);
	      printf("Process %d : Vector = [ ", rank);
	      for (j=0; j<M; j++)
		printf("%d ", Vector[j]);
	      printf("]\n");
	    }

	else // (rank != i)
            MPI_Recv(&Big_Vector[i*M], M, MPI_INT, i, i, MPI_COMM_WORLD, &status);
      }
    
    
    
    
    // Display of Big_Vector by an arbitraly process
    if (rank == 1)
      {
	end = MPI_Wtime();
	printf("Process %d : execution time = %f.\n", rank, end-begin);
	printf("Big_Vector (in lines) : [ ");
	for (i=0; i<M*P; i++)
	  printf("%d ", Big_Vector[i]);
	printf("]\n");
      }
    
    MPI_Finalize();
    return 0;
}

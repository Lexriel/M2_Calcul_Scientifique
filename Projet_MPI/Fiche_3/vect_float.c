# include <mpi.h>
# include <stdio.h>
# include <stdlib.h>
# define M 4

void Build_Vector_1(int K, int rank, float *Vector)
{
  int line;
  for (line=0; line<K; line++)
    Vector[line] = (rank + line)*2.;
}


int main()
{
  int i, j, k, rank, P;
  double begin, end;
  float Vector[M];
  float *Big_Vector;
  
  MPI_Status status;
  
  MPI_Init(NULL,NULL);
  begin = MPI_Wtime();
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  
  Big_Vector = (float *) calloc(P*M,sizeof(float));
  
  
// Each process builds its own vector :
  Build_Vector_1(M, rank, Vector);
  

// We stock this vector in the Big_Vector (instead of sending to itself, that is useless
// in this case where we have to use only MPI_Send and MPI_Recv primitives) :
  for (i=0; i<M; i++)
    Big_Vector[rank*M+i] = Vector[i];
  /* printf("Big_Vector (process %d) : [ ", rank);
    for (i=0; i<M*P; i++)
      printf("%f ", Big_Vector[i]);
    printf("]\n"); */
    
    
/* We begin to consider sending process one after one.
   In each consideration, there is one send and one receive in P iterations.
   When one process has sent its vector to everyone and that everyone has received it, 
   we can consider the following process to do the same thing. */ 
  for (i=0; i<P; i++)
    {
      if (rank == i)
	for (k=0; k<P; k++)
	  {
	    if (k != rank)
	      MPI_Send(Vector, M, MPI_FLOAT, k, rank, MPI_COMM_WORLD);
	        printf("Process %d : Vector = [ ", rank);
		for (j=0; j<M; j++)
		printf("%f ", Vector[j]);
		printf("]\n");
	  }
      else // (rank != i)
	MPI_Recv(&Big_Vector[i*M], M, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
    }
  
    
// Display of Big_Vector by an arbitraly process
  if (rank == P-1)
    {
      end = MPI_Wtime();
      printf("Process %d : execution time = %f.\n", rank, end-begin);
      printf("Big_Vector (in lines) : [ ");
      for (i=0; i<M*P; i++)
	printf("%f ", Big_Vector[i]);
      printf("]\n");
    }
  
  MPI_Finalize();
  return 0;
}

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
  int i, k, rank, P;
  double begin, end;
  float Vector[M];
  float *Big_Vector;
  MPI_Status status;
  
  MPI_Init(NULL,NULL);
  begin = MPI_Wtime();
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  
  Big_Vector = (float *) calloc(P*M, sizeof(float));
  
// Each process builds its own vector :
  Build_Vector_1(M, rank, Vector); 

// Sends the array of each process to the process k and puts them into the array res successively :
  for(k=0; k<P; k++)
    MPI_Gather(Vector, M, MPI_FLOAT, Big_Vector, M, MPI_DOUBLE, k, MPI_COMM_WORLD);
  
// Display of Big_Vector by an arbitraly process :
  if (rank == P-1)
    {
      printf("\nIn the process of rank %d :\nBig_Vector (in lines) : [ ", rank);
      for (i=0; i<M*P; i++)
	printf("%f ", Big_Vector[i]);
      printf("]\n\n");
      end = MPI_Wtime();
      printf("Execution time = %f.\n\n", end-begin);
    }

  free(Big_Vector);
  
  MPI_Finalize();
  return 0;
}

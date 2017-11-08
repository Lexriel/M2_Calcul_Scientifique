# include <stdio.h>
# include <mpi.h>
# include <stdlib.h>

// Procedure which build a sub-matrix of A :
void Build_Matrix(int N, int P, int rank, double **Matrix)
{
  int h = N/P;
  int line, col, local_line;
  for (line=h*rank; line<h*(rank+1); line++)
    {
      local_line = line - h*rank;
      for (col=0; col<N; col++)
	{
	  if (col == line)
	    Matrix[local_line][col] = 1.0;
	  else if (col == line+5)
	    Matrix[local_line][col] = -5.0;
	  else if (col == line+6)
	    Matrix[local_line][col] = 5.0;
	  else
	    Matrix[local_line][col] = 0.0;
	}
    }
}

// Procedure which build the vector x :
void Build_Vector_2(int N, double *Vector)
{
  int line;
  for (line=0; line<N; line++)
    Vector[line] = -1.0;
}


int main()
{
  int i, j, k, M, h, rank, P, iter;
  double begin, end;
  double *x, *y, *res;
  double **Sub_Mat;
  int N = 15;
  
  MPI_Status status;
  //  MPI_Request request;
  
  MPI_Init(NULL,NULL);
  begin = MPI_Wtime();
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  
  // M = number of lines of the matrix A such that M%P == 0. 
  // h = number of lines in each sub-matrix
  M = N; 
  while ( M % P != 0)
    M++;
  h = M/P;
  
  // Allocation of memory for vectors and matrix we use :
  x = (double*) malloc(M*sizeof(double));
  y = (double*) malloc(h*sizeof(double));
  res = (double*) malloc(M*sizeof(double));
  
  Sub_Mat = (double**) malloc(h*sizeof(double*));
  for (i=0; i<h; i++)
    Sub_Mat[i] = (double*) malloc(M*sizeof(double));
  
  // Each process creates its own vector x and sub-matrix of A :
  Build_Vector_2(M, x);
  Build_Vector_2(M, res); // so x == res.
  Build_Matrix(M, P, rank, Sub_Mat);

// We iterate twice the computation of A*res+x (the first computation makes A*x+x (as res=x), and the next one makes A*res+x = A*(A*x+x)+x)
  
  for (iter=0; iter<2; iter++)
    {
      // Partial construction of y = A*res + x :
      for (i=0; i<h; i++)
	{
	  y[i] = 0;
	  for (j=0; j<M; j++)
	    y[i] = y[i] + Sub_Mat[i][j]*res[j];
	  y[i] = y[i] + x[i];
	}
      
      // Each process puts its partial solution y into the global solution res :
      for (i=0; i<h; i++)
	res[rank*h+i] = y[i];
      
      // Each process sends to every process of different rank its peace of res : 
      for (i=0; i<P; i++)
	{
	  if (rank == i)
	    {
	      for (k=0; k<P; k++)
		{
		  if (k != rank)
		    MPI_Send(y, h, MPI_DOUBLE, k, rank, MPI_COMM_WORLD);
		}
	    }
	  
	  else // (rank != i)
	    MPI_Recv(&res[i*h], h, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
	}
      
      
      // Display of res by the last process
      if (rank == P-1)
	{
	  
          if (iter == 0)
            {
	      printf("The result res = A*x+x is given (in lines) by :\nres : [ ");
	      for (i=0; i<M; i++)
		printf("%f ", res[i]);
	      printf("]\n\n");
            }            
	  
	  else //(iter == 1)
		 {
		   printf("The result res = A*(A*x+x)+x is given (in lines) by :\nres : [ ");
		   for (i=0; i<M; i++)
		     printf("%f ", res[i]);
		   printf("]\n\n");
		   
		   end = MPI_Wtime();
		   printf("Execution time = %f.\n\n", end-begin);
		 }
	}
    }

// Deallocation of the arrays :  
  free(x);
  free(y);
  free(res);
  for (i=0; i<h; i++)
    free(Sub_Mat[i]);
  free(Sub_Mat);
  
  MPI_Finalize();
  return 0;
}

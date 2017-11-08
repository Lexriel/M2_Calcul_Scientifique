# include <stdio.h>
# include <stdlib.h>
# include <mpi.h>

// definition of the generic term of the sum to compute pi //


double pow(double a, double b)
{
  int i;
  double result=1.0;
  
  if (b !=0)
    {
      for (i=0; i<b; i++)
	result = result*a;
    }
  return result;
}



double f(int k)
{
  return 4*pow(-1,k)/ (double) (2*k+1);
}


int main(int argc,char* argv[])
{
  int rank,P,i,nb_decimal,nb_termes;
  int start, stop;
  double partial_sum, total_sum;

 /* MPI_Status status;
  MPI_Request request;*/

  MPI_Init(NULL,NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &P);

  nb_decimal = atoi(argv[1])+1;
  nb_termes = 2*pow(10, nb_decimal);
  partial_sum=0;
  total_sum=0;

  // Let's compute selon les P processeurs //

  start = rank * ((float)nb_termes/(float)P);
  stop = (rank+1) * ((float)nb_termes/(float)P);
  
  for (i = start; i < stop; i++)   
    partial_sum = partial_sum + f(i);

  // Reduction with sum //

  MPI_Reduce(&partial_sum, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0)
   printf ("Approx of Pi with %d decimals is : %f\n", nb_decimal, total_sum);

  MPI_Finalize();
  return 0;
}

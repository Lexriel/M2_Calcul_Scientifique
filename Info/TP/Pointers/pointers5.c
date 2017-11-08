# include <stdlib.h>
# include <stdio.h>
# define N 10

int main()
{
  int n1, n2, n3, i, j, k;
  int **A, **B, **C;
  srand((unsigned int) time(NULL));
  printf("You want to do the product of two matrixes A and B of size (n1,n2) and (n2,n3) respectively; A and B contening random integers.\n");
  printf("Give me n1:\n");
  scanf("%d", &n1);
  printf("Give me n2:\n");
  scanf("%d", &n2);
  printf("Give me n3:\n");
  scanf("%d", &n3);
 
  A=(int **) malloc(n1*sizeof(int*));
  for (i=0; i<n1; i++)
    A[i]=(int*) malloc(n2*sizeof(int));

  B=(int **) malloc(n2*sizeof(int*));
  for(j=0; j<n2; j++)
    B[j]=(int*) malloc(n3*sizeof(int));

  C=(int **) malloc(n1*sizeof(int*));
  for(i=0; i<n1; i++)
    C[i]=(int*) malloc(n3*sizeof(int*));

  /* Let's define A and B. */

  for (i=0; i<n1; i++)
    for (j=0; j<n2; j++)
      A[i][j]=rand()%N;


  for (j=0; j<n2; j++)
    for (k=0; k<n3; k++)
      B[j][k]=rand()%N;
 
  /* Let's compute the matrix product between A and B. */

  for (i=0; i<n1; i++)
    for(k=0; k<n3; k++)
      {
	C[i][k] = 0;
	for (j=0; j<n2; j++)
	  {
	  C[i][k] = C[i][k] + A[i][j] * B[j][k];  
	  }
      }  

  /* Give A, B and C. */

  printf("The matrix A is:\n");
  for (i=0; i<n1; i++)
    {
      for (j=0; j<n2; j++)
	printf("%d  ", A[i][j]);
      printf("\n");
    }
  printf("\n");

  printf("The matrix B is:\n");
  for (j=0; j<n2; j++)
    {
      for (k=0; k<n3; k++)
	printf("%d  ", B[j][k]);
      printf("\n");
    }
  printf("\n");

  printf("The matrix C is:\n");
  for (i=0; i<n1; i++)
    {
      for (k=0; k<n3; k++)
	printf("%d  ", C[i][k]);
      printf("\n");
    }
  printf("\n");

  return 0;

}

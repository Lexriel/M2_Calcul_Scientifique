# include <stdlib.h>
# include <stdio.h>
# define N 100

int main()
{
  int n, m, i;

  int *T1, *T2;

  printf("Give me the size of your table:\n");
  scanf("%d", &n);
 
  T1=(int *) malloc(n*sizeof(int));
  T2=(int *) malloc((n+1)*sizeof(int));

  for (i=0; i<n; i++)
    {
      T1[i]=rand()%N;
      T2[i]=T1[i];
    }
  
  printf("\n Give me the %d st value you want for the table T2:\n", n+1);
  scanf("%d", &m);
  T2[n]=m%N;
  free(T1);
  T1=T2;

  for(i=0; i<=n; i++)
    printf("T1[%d]=%d \n", i+1, T2[i]);
}

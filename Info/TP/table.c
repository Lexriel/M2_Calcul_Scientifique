# include <stdio.h>
# include <stdlib.h>
# define N 5

void mytabprint(int tab[], int size)
{
  int i;
  for (i=0; i<size; i++)
    printf("%d ",tab[i]);
  putchar('\n');
}

int main()
{
  
  int a, i, j, z;
  int tab[N];
  
  for (a=0; a<N; a++)
    tab[a]=rand()%100;
  
  printf("Initial table \n ");
  mytabprint(tab, N);
  
  for (i=1; i<N; i++)
    {
      
      for (j=0; j<i; j++)
	{
	  
	  if (tab[i]<tab[j])
	    {
	      z=tab[j];
	      tab[j]=tab[i];
	      tab[i]=z;
	    }
	}
      
    }
  printf("Sorted table \n ");
  mytabprint( tab, N);
  return 0;
}

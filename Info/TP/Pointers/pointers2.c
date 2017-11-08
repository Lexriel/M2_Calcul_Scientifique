# include <stdio.h>
# include <stdlib.h>
# include "pointer2.h"
# include "pointers1.h"

char* concatenate(char *A, char *B)
{
  int m, n, i;
  char *C;

  m=nb_characters(A);
  n=nb_characters(B);
  C=(char *) malloc((m+n+1)*sizeof(char));
/* The +1 is for the \0 character at the end. */
  for(i=0; i<m ; i++)
    *(C+i)=*(A+i);

  for (i=m; i<m+n; i++)
    *(C+i)=*(B+i-m);

  return C;
}


int main()
{
  printf("%s", concatenate("anti", "banana"));
  return 0;
}

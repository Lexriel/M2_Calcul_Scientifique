# include <stdio.h>
# include <stdlib.h>
# include "pointers1.h"


int nb_characters(char *chain)
{
  int i, n;
  
  for (i=0; *(chain +i) != '\0'; i++)
    ;
  
  n=i;
  return n;
}


/*                                  TEST
 ==============================================================================
int main()
{
  printf("%d", nb_characters("this is a test"));
  return 0;
} 
===============================================================================
                                                                             */

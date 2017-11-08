# include "french_tarot1.h"
# include <stdio.h>
# include <stdlib.h>

int tarot(card A[2], color required_suit)
{
  if ((A[0].col != required_suit) && (A[1].col != required_suit))
    {
      printf("Error");
      exit(EXIT_FAILURE);
    }
  
  if (A[0].col == A[1].col)
    {
    if (A[0].val > A[1].val)
      return 0;
    else
      return 1;
    }

  if (A[0].col != A[1].col)
    {
      if (A[0].col == 5)
	return 0;
      if (A[1].col == 5)
	return 1;
    
      else
	{
	if (A[0].col == required_suit)
	  return 0;
	else
	  return 1;
	}
    }
  exit(EXIT_FAILURE);
}

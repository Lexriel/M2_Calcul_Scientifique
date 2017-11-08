# include "french_tarot1.h"

int is_it_an_oudler(card c)
{
  if (c.col == 5)
    if (c.val == 1 || c.val == 21)
      return 1; 

  if (c.col == 6)
    return 1;
  /* This is an oudler. */

  else
    return 0; 
  /* This is not an oudler. */
}


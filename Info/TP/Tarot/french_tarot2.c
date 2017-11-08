# include <stdio.h>
# include "french_tarot1.h"


int main()
{
  card c={trump,15};
  int a=is_it_an_oudler(c);
  if (a==1)
    printf("This is an oudler.");
  else
    printf("This is not an oudler.");
  return 0;
}


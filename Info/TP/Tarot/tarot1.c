# include <stdio.h>
# include "french_tarot1.h"
# include "tarot0.h"

int main()
{
  card B[4], C[2], W ;              /* W: winner */
  int r;                            /* r: required suit */
  int n, i, k;                      /* k: number of the winner player */


  for (i=0; i<4; i++)
    {
      printf("Color of the player %d 's card (1 to 6): \n", i+1);
      scanf("%d",&n);
      B[i].col=n;
      printf("Value of the player %d 's card (1 to 21): \n", i+1);
      scanf("%d",&n);
      B[i].val=n;
    }

  printf("\n");
  printf("Give me the required suit.\n");
  scanf("%d", &r);
  printf("\n\n");
  
  C[0]=B[0];
  C[1]=B[1];
  k=1;
      
  for (i=1; i<=3; i++)
    {
      n=tarot(C[2], r);
      
      if (n==0)
	W=C[0];
      else
	W=C[1]; k=i+1;
      
      C[0]=W;
      C[1]=B[i+1];
      r=C[0].col; 
    }

  printf("The card has a color %d and a value %d .\n", B[0].col, B[0].val);
  printf("The player %d wins the trick.\n\n", k);
  return 0;
}

# include "power.h"

float power ( float x, int k)
  {
    int i ;
    float y=1 ;

    if (k==0)
    return 1;
   
    else
      {
        for (i=1; i<=k; i++)
        y=y*x ;
      }

  return y;
  
  }

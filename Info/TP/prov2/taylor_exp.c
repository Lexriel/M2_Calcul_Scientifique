# include "taylor_exp.h"

float taylor_exp(float x, int n)
{
  float z=1;
  int i;
  for (i=1; i<=n; i++)
    z=z+power(x,i)/fact(i) ;
  return z;
}



# include "fact.h"

int fact(int n)
{
  return n<=1 ? 1 : n*fact(n-1);
}

long double fact2 (int n)
{
  return n<=1 ? 1 : n*fact2(n-1);
}
/*  int i;
  long double res=1;
  for (i=2; i<=n; i++)
    res*=i;
  return res;
  }*/

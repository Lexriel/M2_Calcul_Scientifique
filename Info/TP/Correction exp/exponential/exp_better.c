# include "exp_better.h"

long double exp_better(long double x, int n){
  long double res=1,c=x;
  int i;
  for (i=2; i<=n+1; i++)
    {
      res+=c;
      c*=(x/i);
    }
  return res;
}

float exp_better2(float x, int n){
  float res=1,c=x;
  int i;
  for (i=2; i<=n+1; i++)
    {
      res+=c;
      c*=(x/i);
    }
  return res;
}

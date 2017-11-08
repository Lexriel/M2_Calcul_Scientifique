# include "exp_basic.h"

float exp_basic(float x, int n){
  float res=1;
  int i;
  for (i=1; i<=n; i++)
    res+=power(x,i)/fact(i);
  return res;
}

long double exp_basic2(long double x, int n){
  long double res=1;
  int i;
  for (i=1; i<=n; i++)
    res+=power2(x,i)/fact2(i);
  return res;
}

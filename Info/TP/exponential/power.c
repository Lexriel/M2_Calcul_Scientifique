# include "power.h"

float power(float x, int n)
{
  float res=1;
  while (n>=1)
    { 
      res*=x;
      n--;
    }
  return res;
}

double power3(double x, int n)
{
  double res=1;
  while (n>=1)
    { 
      res*=x;
      n--;
    }
  return res;
}

long double power2(long double x, int n)
{
  long double res=1;
  while (n>=1)
    { 
      res*=x;
      n--;
    }
  return res;
}

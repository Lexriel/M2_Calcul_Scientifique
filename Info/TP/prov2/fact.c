# include "fact.h"


int fact (int n)
{
  int f=1, k ;
  
  if (n!=0)
    {
      for (k=1; k<=n; k++)
	f=f*k ;
    }
  
  return f;
}


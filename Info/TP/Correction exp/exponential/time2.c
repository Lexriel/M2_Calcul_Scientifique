# include "time2.h"

int main()
{
  int i,j=1;
  long double res;
  for (i=0; i<=24; i++)
    {
      initial_time=clock();
      res=exp_better(3,j);
      final_time=clock();
      cpu_time=(final_time - initial_time)*1e-6;
      printf("%d %Lf %f\n",i,res,cpu_time);
      j<<=1;
    }
  exit(EXIT_SUCCESS);
}

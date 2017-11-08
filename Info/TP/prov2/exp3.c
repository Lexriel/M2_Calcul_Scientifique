# include "menu.h"


main()
{
  clock_t initial_time; /* Initial time in micro-seconds */ 
  clock_t final_time; /* Final time in micro-seconds */ 
  float cpu_time; /* Total time in seconds */ 

  int k,i,j;
  float t[L][C], u[L][C], z=1, y;
  
  for (k=0; k<C; k++) 
    {
       t[1][k]=k+1;
    
      initial_time=clock();
      taylor_exp(3,power(2,k+1));
      final_time=clock(); 
      cpu_time=(final_time - initial_time)*1e-6; 

      t[0][k]=cpu_time;
    }
  for  (i=0; i<C; i++)
    {
     for  (j=0; j<L; j++) 
	printf("%f ",t[j][i]);
      printf("\n");
    }

  // modèle économique  
  for (k=0; k<C; k++) 
    {
       u[1][k]=k+1;
    
      initial_time=clock();
      y=3/(k+1)*y;
      z=z+y;
      final_time=clock(); 
      cpu_time=(final_time - initial_time)*1e-6; 

      u[0][k]=cpu_time;
    }
  for  (i=0; i<C; i++)
    {
     for  (j=0; j<L; j++) 
	printf("%f ",u[j][i]);
      printf("\n");
    }
}

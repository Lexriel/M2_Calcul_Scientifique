# include "menu.h"

void menu()
{
  printf("menu :\n");
  printf("0 - exit\n");
  printf("1 - Compute a power of a value\n");
  printf("2 - Compute a factorial\n");
  printf("3 - Compute a taylor expansion of exp\n");
}

int main(void){
  int n,k;
  float x;
  /* one may want to add a loop here */
  menu();
  scanf("%d",&n);
  switch (n)
    {
    case 1:
      {
	printf("Value of x ? ");
	scanf("%f",&x);
	printf("Value of n ? ");
	scanf("%d",&k);
	printf("%f",power(x,k));
	break;
      }
    case 2:
      {
	printf("Value of n ? ");
	scanf("%d",&k);
	printf("%d",fact(k));
	break;
      }
    case 3:
      {
	clock_t initial_time; /* Initial time in micro-seconds */ 
	clock_t final_time; /* Final time in micro-seconds */ 
	float cpu_time; /* Total time in seconds */ 

	printf("Value of x ? ");
	scanf("%f",&x);
	printf("Value of n ? ");
	scanf("%d",&k);

	initial_time=clock();

	printf("taylor(%f,%d)=%f\n\n",x,k,taylor_exp(x,k));

	final_time=clock(); 
	cpu_time=(final_time - initial_time)*1e-6; 
	printf ("This computation was done on %f seconds.\n\n",cpu_time);
	break;
      }
    case 0:
      {
	printf("Good bye\n");
	break;
      }
    default:
      {
	printf("Sorry, this option does not exist\n");
	exit(EXIT_FAILURE);
      }
    }
  exit(EXIT_SUCCESS);
}

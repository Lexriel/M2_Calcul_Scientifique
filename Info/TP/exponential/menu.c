# include "menu.h"

void menu()
{
  printf("menu :\n");
  printf("0 - Exit\n");
  printf("1 - Compute a power of a value using float\n");
  printf("2 - Compute a power of a value using double\n");
  printf("3 - Compute a power of a value using long double\n");
  printf("4 - Compute a factorial using int\n");
  printf("5 - Compute a factorial using long double\n");
  printf("6 - Compute an approximation of an exponential : naive algorithm with float and int\n");
  printf("7 - Compute an approximation of an exponential : naive algorithm with long double\n");
  printf("8 - Same than 7 with running time\n");
  printf("9 - Compute an approximation of an exponential : better algorithm with float and int\n");
  printf("10 - Compute an approximation of an exponential : better algorithm with long double\n");
  printf("11 - Same than 10 with running time\n");
}

int main(void){
  int n,k,go=1;
  float x,res2;
  double y;
  long double z,res;
  while (go) 
    {
      menu();
      scanf("%d",&n);
      switch (n)
	{
	case 0:
	  exit(EXIT_SUCCESS);
	case 1:
	  {
	    printf("Value of x ? ");
	    scanf("%f",&x);
	    printf("Value of n ? ");
	    scanf("%d",&k);
	    printf("The power is %f\n",power(x,k));
	    break;
	  }
	case 2:
	  {
	    printf("Value of x ? ");
	    scanf("%lf",&y);
	    printf("Value of n ? ");
	    scanf("%d",&k);
	    printf("The power is %f\n",power3(y,k));
	    break;
	  }
	case 3:
	  {
	    printf("Value of x ? ");
	    scanf("%Lf",&z);
	    printf("Value of n ? ");
	    scanf("%d",&k);
	    printf("The power is %Lf\n",power2(z,k));
	    break;
	  }
	case 4:
	  {
	    printf("Value of n ? ");
	    scanf("%d",&k);
	    printf("The factorial is %d\n",fact(k));
	    break;
	  }
	case 5:
	  {
	    printf("Value of n ? ");
	    scanf("%d",&k);
	    printf("The factorial is %Lf\n",fact2(k));
	    break;
	  }
	case 6:
	  {
	    printf("Value of x ? ");
	    scanf("%f",&x);
	    printf("Order of approximation ? ");
	    scanf("%d",&k);
	    printf("The value of the approximation is %f\n",exp_basic(x,k));
	    break;
	  }
	case 7:
	  {
	    printf("Value of x ? ");
	    scanf("%Lf",&z);
	    printf("Order of approximation ? ");
	    scanf("%d",&k);
	    printf("The value of the approximation is %Lf\n",exp_basic2(z,k));
	    break;
	  }
	case 8:
	  {
	    printf("Value of x ? ");
	    scanf("%Lf",&z);
	    printf("Order of approximation ? ");
	    scanf("%d",&k);
	    initial_time=clock();
	    res2=exp_basic2(z,k);
	    final_time=clock();
	    cpu_time=(final_time - initial_time)*1e-6;
	    printf("The value of the approximation is %f\n",res2);
	    printf ("It took %f seconds to compute it\n",cpu_time);
	    break;
	  }
	case 9:
	  {
	    printf("Value of x ? ");
	    scanf("%f",&x);
	    printf("Order of approximation ? ");
	    scanf("%d",&k);
	    printf("The value of the approximation is %f\n",exp_better2(x,k));
	    break;
	  }
	case 10:
	  {
	    printf("Value of x ? ");
	    scanf("%Lf",&z);
	    printf("Order of approximation ? ");
	    scanf("%d",&k);
	    printf("The value of the approximation is %Lf\n",exp_better(z,k));
	    break;
	  }
	case 11:
	  {
	    printf("Value of x ? ");
	    scanf("%Lf",&z);
	    printf("Order of approximation ? ");
	    scanf("%d",&k);
	    initial_time=clock();
	    res=exp_better2(z,k);
	    final_time=clock();
	    cpu_time=(final_time - initial_time)*1e-6;
	    printf("The value of the approximation is %Lf\n",res);
	    printf ("It took %f seconds to compute it\n",cpu_time);
	    break;
	  }
	default:
	  printf("Sorry, this option does not exist\n");
	}
    }
  exit(EXIT_SUCCESS);
}

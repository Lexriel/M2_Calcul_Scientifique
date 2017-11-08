# include <stdlib.h>
# include <stdio.h>

int main()
{
  long int tab[20000], n ;
  long long int z0=0, z1=0, z2=0, z3=0 ;
  int i ;
  float y=0, z ;

  for (n=0; n<20000; n+=4)
    {
      tab[n]=rand()%1112465224;
      z0=z0+tab[n];

      tab[n+1]=rand()%6;
      z1=z1+tab[n+1];

      tab[n+2]=rand()%5001;
      z2=z2+tab[n+2];
      
      tab[n+3]=rand()%1000001;
      z3=z3+tab[n+3];
    }
  //computing directly the sum of the numbers of the table
  for(i=0; i<20000; i++)
    y=y+tab[n];
  
  
  // computing the sum of the sum of each case
  z=z1+z2;
  z=z+z3;
  z=z+z0;

  printf("The sum of all numbers in tab[20000] is %f.\n", y);
  printf("The sum of the sum of numbers in each case in tab[20000] is %f.\n\n", z);

}

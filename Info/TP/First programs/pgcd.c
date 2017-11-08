# include <stdio.h>
# include <math.h>

main()
{
  int a, b, r1, r2, r;
  
  printf("Give the two numbers you want to compute their GCD.\n");
  scanf("%d %d", &a, &b);
  
  if (a<b)
    r1=b, r2=a ;
  else
    r1=a, r2=b ;
  
  while (r2>0)
    {
      r=r1%r2 ;
      r1=r2 ;
      r2=r ;      
    }
  
  printf("GCD(%d,%d)=%d \n\n", a, b, r1); 
}

# include <stdio.h>
# include <math.h>

main()
{
  int n, i, k ;
  double p, a ;
  float x ;

  printf("Give the x-value you want to calculate your polynomial P.\n") ;
    scanf("%f", &x) ;

    printf("Give me the degree of your polynomial P(X).\n") ;
  scanf("%d", &n) ;

  printf("Give me the coefficient a_%d :\n", n);
  scanf("%lf", &a) ;
  p=a ;

  for (i=1; i<=n; i++)
    {
      k=n-i ;
      printf("Give me the coefficient a_%d :\n", k) ;
      scanf("%lf", &a) ;
      p=p*x+a ;
    }

  printf("P(%f)=%f \n\n", x, p) ;

}

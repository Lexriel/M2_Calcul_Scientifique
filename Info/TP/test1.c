# include <stdio.h>
# include <stdlib.h>
# define produit(A,B) ((A)*(B))
# define carre(x) ((x)*(x)))
int main()
{
  float x,y;
  int n;
  x=produit(5,6);
  y=carre(3);
  printf("cela doit valoir 30: x=%f\n", x);
  printf("3^2 fait 9: y=%f\n", y);
  return 0;
}

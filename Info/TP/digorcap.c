# include <stdio.h>
# include <stdlib.h>
# define ISDIG(c) (((c)>='0')&&((c)<='9')))
# define ISSMALL(c) (((c)>='a')&&((c)<='z')))
# define ISCAP(c) (((c)>='A')&&((c)<='Z')))
# define DIGORCAP(c) (ISCAP(c)||ISDIG(c))

/* int main()
{
  char c,d;
  c='0'+'A';
  d=115;
  printf("%c %c\n",c,d);
  return 0;
}

*/

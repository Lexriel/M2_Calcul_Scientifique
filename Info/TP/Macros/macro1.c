# include <stdio.h>
# define TRUE (!FALSE)
# define FALSE 0

# define MAX(a,b) (a)>(b)?(a):(b)

int main()
{
  int a=5,b=13;
  printf("%d\n",MAX(a,b));
  printf("%d\n",MAX(a++,b++));
  printf("a %d b %d\n",a,b);
  printf("%d\n",MAX(++a,++b));
  printf("a %d b %d\n",a,b);
  return 0;
}

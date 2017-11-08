# include <stdio.h>

# define ISDIG(c) (((c)>='0')&&((c)<='9'))

int main() 
{ 
  unsigned int res=0;
  char c; 
  while (c=getchar(), ISDIG()) 
    {
      res*=10; 
      res+=c-'0'; 
    } 
  printf("%d\n",res); 
  return 0; 
}

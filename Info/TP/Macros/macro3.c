# include <stdio.h> 

# define ISDIG(c) (((c)>='0')&&((c)<='9')) 

int main() 
{ 
  unsigned int res=0; 
  int c; 
  while (ISDIG(c=getchar())) 
    { 
      res*=10; 
      res+=c-'0'; 
    } 
    printf("%d",res); 
    return 0; 
}



# include <stdio.h>
# include <stdlib.h>
# include "library1.h"

void findnumber(int n)
{
  FILE * fp;
  pointer_book* library;
  library = (pointer_book*) calloc(10000, sizeof(pointer_book));
  fp=fopen("biglib.txt","r");
  puts(n);
}

int main()
{
  findnumber(12);
}

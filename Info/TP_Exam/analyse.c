# include <stdio.h>
# include <stdlib.h>
# include <string.h>

int main()
{
  FILE* fp;
  char c;
  fp=fopen("texte.txt", "r");
  while ((c=fgetc(fp)) != EOF)
    printf("%c",c);
  fclose(fp);
  return 0;
}

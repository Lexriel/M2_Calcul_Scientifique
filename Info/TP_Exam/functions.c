# include "pp.h"
# include "functions.h"

int indent(FILE* fp)
{
  int i;
  for (i=0; i<cnt; i++)
    fputs("  ", fp);
  return 0;
}


int brace1_command(FILE* fp)
{
  indent(fp);
  fputs("{\n", fp);
  cnt++;
  indent(fp);
  return BEGINNING_LINE;
}


int brace2_command(FILE* fp)
{
  cnt--;
  if (cnt<0)
    {
      printf("Be careful, one or some braces are not closed.\n");
      exit(1);
    }
  indent(fp);
  fputs("}\n", fp);
  indent(fp);
  return BEGINNING_LINE;
}

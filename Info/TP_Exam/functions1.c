# include "pp.h"

/* This function makes an indentation of a line (2 spaces). */
int indent(int m)
{
  int i;
  for (i=0; i<m; m++)
    printf("  ");
  return 0;
}


/* This function opens the file called 'file1' in reading mode,
   and a new one in writting mode. */
int open_file(char* file1, FILE* fp1, FILE* fp2)
{
  char* file2;
  file2=(char*) malloc(N*sizeof(char));
  file2=strcat("_","file1");
  printf("Give the name of the file you want to apply the pretty printer:\n");
  scanf("%s", file1);
  fp1=fopen("file1", "r");
  if (fp1 == NULL)
    {
      printf("Impossible to open the file %s.\n", file1);
      exit(1);
    }
  fp2=fopen("file2", "w");
  if (fp2 == NULL)
    {
      printf("Impossible to create the second file.\n");
      exit(1);
    }
  return 0;
}


/* This function closes two files. */
int close_file(FILE* fp1, FILE* fp2)
{
  fclose(fp1);
  fclose(fp2);
  return 0;
}


/* This command indents the line, writes the '{', then we pass a line, add 1 to cnt, indent again and we are now in the beginning of a new line. */
void brace1_command(int j, FILE* fp)
{
  indent(j);
  putc('{', fp);
  putc('\n', fp);
  j++;
  indent(j);
  w=BEGINNING_LINE;
}


/* This command indent with the same indentation than the previous '{' (that's why we decrement before) write the '}', then we pass a new line and indent again. We are now in the beginning of a new line. */
void brace2_command(int j, FILE* fp)
{
  cnt--;
  if (j<0)
    {
      fprintf(stderr, "Be careful, one or some braces are not closed.\n");
      exit(EXIT_FAILURE);
    }
  indent(j);
  putc('}', fp);
  putc('\n', fp);
  indent(j);
  w=BEGINNING_LINE;
}

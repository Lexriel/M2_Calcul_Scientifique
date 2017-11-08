# include "pp.h"
# include "functions.h"


int main()
{
  char *file1, file2[N]="_";
  int c;
  FILE *fp1=NULL;
  FILE *fp2=NULL;
  cnt=0;
  com=0;
  w=BEGINNING_LINE;
  file1=(char*) malloc((N+1)*sizeof(char));

  printf("In what file you want to apply the pretty printer ?\n");
  scanf("%s", file1);

  strcat(file2, file1);
  fp1=fopen(file1, "r");
  fp2=fopen(file2, "w");
  c=fgetc(fp1);
  
  while (c != EOF)
    {
      switch (w)
	{


	case BEGINNING_LINE:
	  switch (c)
	    {
      	    case '/':
	      w=BEGINNING_COMMENT;
	      break;
	    case ' ':
	      break;
	    case '\n':
	      break;
	    case '\t':
	      break;
	    case '{':
	      putc('\n', fp2);
	      w=brace1_command(fp2);
	      break;
	    case '}':
	      putc('\n', fp2);
	      w=brace2_command(fp2);
	      break;
	    case '#':
	      putc(c, fp2);
	      w=MACRO;
	      break;
	    case '"':
	      putc(c, fp2);
	      w=CHAIN;
	      break;
	    default:
	      putc(c, fp2);
	      w=MIDDLE_LINE;
	      break;
	    }
	  break;
	  
	  
	case MIDDLE_LINE:
	  switch (c)
	    {
	    case '"':
	      putc(c, fp2);
	      w=CHAIN;
	      break;
	    case '/':
	      w= BEGINNING_COMMENT;
	      break;
	    case '\n':
	      putc(c, fp2);
	      indent(fp2);
	      w=BEGINNING_LINE;
	      break;
	    case '{':
	      putc('\n', fp2);
	      w=brace1_command(fp2);
	      break;
	    case '}':
	      putc('\n', fp2);
	      w=brace2_command(fp2);
	      break;
	    default:
	      putc(c, fp2);
	      break;
	    }
	  break;

	case BEGINNING_COMMENT:
	  switch (c)
	    {
	    case '\n':
	      fputs("/\n", fp2);
	      indent(fp2);
	      w=BEGINNING_LINE;
	      break;
	    case '{':
	      fputs("/\n", fp2);
	      w=brace1_command(fp2);
	      break;
	    case '}':
	      fputs("/\n", fp2);
	      w=brace2_command(fp2);
	      break;
	    case '/':
	      putc('*', fp2);
              w=COMMENT;
	      break;
	    case '*':
	      putc('\n', fp2);
	      indent(fp2);
	      fputs("/*", fp2);
	      com++;
	      w=PLUS_COMMENT;
	      break;
	    case '"':
	      putc('/', fp2);
	      putc('"', fp2);
	      w=CHAIN;
	      break;
	    default:
	      putc('/', fp2);
	      putc(c, fp2);
	      w=MIDDLE_LINE;
	      break;
	    }
	  break;

	case COMMENT:
	  switch (c)
	    {
	    case '\n':
	      fputs("*/\n", fp2);
	      indent(fp2);
	      fputs("/*", fp2);
	      w=BEGINNING_SPACES_COMMENT;
	      break;
	    case '*':
	      w=END_COMMENT;
	      break;
	    default:
	      putc(c, fp2);
	      break;
	    }
	  break;

	case PLUS_COMMENT:
	  switch (c)
	    {
	    case '\n':
	      fputs("*/\n", fp2);
	      indent(fp2);
	      w=BEGINNING_LINE;
	      break;
	    default:
	      putc(c, fp2);
	      break;
	    }
	  break;

	case BEGINNING_SPACES_COMMENT:
	switch (c)
	  {
	  case ' ':
	    break;
	  case '\t':
	    break;
	  case '\n':
	    break;
	  default : 
	    putc(c, fp2);
	    w=COMMENT;
	    break;
	  }
	break;

	case END_COMMENT:
	  switch (c) 
	    {
	    case '*':
	      putc('*', fp2);
	      break;
	    case '/':
	      fputs("*/\n", fp2);
	      indent(fp2);
	      com--;
	      w=BEGINNING_LINE;
	      break;
	    default:
	      putc('*', fp2);
	      putc(c, fp2);
	      w=COMMENT;
	      break;
	    }
	  break;
      
	case CHAIN:
	  switch (c)
	    {
	    case '"':
	      putc(c, fp2);
	      w=MIDDLE_LINE;
	      break;
	    default:
	      putc(c, fp2);
	      break;
	    }
	  break;

	case MACRO:
	  switch (c)
	    {
	    case '\n':
	      putc('\n', fp2);
	      indent(fp2);
	      w=BEGINNING_LINE;
	      break;
	    case '/':
	      w=BEGINNING_COMMENT;
	      break;
	    default:
	      putc(c, fp2);
	      break;
	    }

	} /* end braces of switch (w) */
c=fgetc(fp1);
    } /* end brace of while */


/* Test of coherence at the end. */  
  if ((com != 0) && (cnt != 0))
    {
      fprintf(stderr, "Be careful, one or some comments are not closed and one or some braces too.\n");
      exit(1);
    }
  if ((com == 0) && (cnt != 0))
    {
      fprintf(stderr, "Be careful, one or some braces are not closed.\n");
      exit(1);
    }
  if ((com != 0) && (cnt == 0))
    {
      fprintf(stderr, "Be careful, one or some comments are not closed.\n");
      exit(1);
    }



  /* close the 2 files we were using. */
  fclose(fp1);
  fclose(fp2);
  free(file1);

  return 0;
}   
      

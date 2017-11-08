# include "pp.h"



int main()
{
  char* file1;
  char c;
  cnt=0;
  com=0;
  w=BEGINNING_LINE;
  FILE* fp1;
  FILE* fp2;
  open_file(file1, fp1, fp2);
  

  while ((c=fgetc(fp1)) != EOF)
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
	      brace1_command(cnt, fp2);
	      break;
	    case '}':
	      putc('\n', fp2);
	      brace2_command(cnt, fp2);
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
	      indent(cnt);
	      w=BEGINNING_LINE;
	      break;
	    case '{':
	      putc('\n', fp2);
	      brace1_command(cnt, fp2);
	      break;
	    case '}':
	      putc('\n', fp2);
	      brace2_command(cnt, fp2);
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
	      putc('/', fp2);
	      putc('\n', fp2);
	      indent(cnt);
	      w=BEGINNING_LINE;
	      break;
	    case '{':
	      putc('/', fp2);
	      putc('\n', fp2);
	      brace1_command(cnt, fp2);
	      break;
	    case '}':
	      putc('/', fp2);
	      putc('\n', fp2);
	      brace2_command(cnt, fp2);
	      break;
	    case '/':
	      putc(c, fp2);
	      break;
	    case '*':
	      putc('\n', fp2);
	      indent(cnt);
	      putc('/', fp2);
	      putc('*', fp2);
	      com++;
	      w=COMMENT;
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
	      putc('*', fp2);
	      putc('/', fp2);
	      putc('\n', fp2);
	      indent(cnt);
	      putc('/', fp2);
	      putc('*', fp2);
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
	      putc('*', fp2);
	      putc('/', fp2);
	      putc('\n', fp2);
	      indent(cnt);
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
	      indent(cnt);
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
    } /* end brace of while */
	  
 error_message: fprintf(fp2,"");
  if ((com != 0) && (cnt != 0))
    {
      fprintf(stderr, "Be careful, one or some comments are not closed and one or some braces too.\n");
      exit(EXIT_FAILURE);
    }
  if ((com == 0) && (cnt != 0))
    {
      fprintf(stderr, "Be careful, one or some braces are not closed.\n");
      exit(EXIT_FAILURE);
    }
  if ((com != 0) && (cnt == 0))
    {
      fprintf(stderr, "Be careful, one or some comments are not closed.\n");
      exit(EXIT_FAILURE);
    }

  close_file(fp1, fp2);
  return 0;
  exit(EXIT_SUCCESS);
}   
      

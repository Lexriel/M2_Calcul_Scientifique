# ifndef PP_H
# define PP_H

# include <stdio.h>
# include <stdlib.h>
# include <string.h>

# define N 100

  /* cnt is a number which counts +1 for '{' and -1 for '}'. We will use it to know how many spaces we need to indent lines.
     com is a number which counts +1 for the biginning of a comment, and -1 for the end of a comment.
     We define these two numbers as global variables. */
int cnt, com;

/* We define a special type to know in what kind of situation we are in the document we want to modify with the pp programm, and so, to do different things in each situations. These situations are detailled as elements of where_am_i: */
enum Where_am_i {BEGINNING_LINE, MIDDLE_LINE, BEGINNING_COMMENT, BEGINNING_SPACES_COMMENT, COMMENT, END_COMMENT, CHAIN, MACRO} ;
typedef enum Where_am_i where_am_i;
where_am_i w;

extern int indent(int);
extern void brace1_command(int, FILE*);
extern void brace2_command(int, FILE*);
extern int open_file(char*, FILE*, FILE*);
extern int close_file(FILE*, FILE*);



# endif /* PP_H */

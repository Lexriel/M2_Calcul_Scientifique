# ifndef PP_H
# define PP_H

# include <stdio.h>
# include <stdlib.h>
# include <string.h>

# define N 100


int cnt, com;


enum Where_am_i {BEGINNING_LINE, MIDDLE_LINE, BEGINNING_COMMENT, BEGINNING_SPACES_COMMENT, PLUS_COMMENT, COMMENT, END_COMMENT, CHAIN, MACRO} ;
typedef enum Where_am_i where_am_i;
where_am_i w;


# endif /* PP_H */

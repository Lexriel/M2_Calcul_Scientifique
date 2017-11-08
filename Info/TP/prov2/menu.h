# ifndef MENU_H 
# define MENU_H 

/* we include the standard libraries we will use */ 
#include <stdio.h> /* for printf and scanf */ 
#include <stdlib.h> /* for exit(), EXIT_SUCCESS and EXIT_FAILURE */ 
# include <time.h>

/* we include the other files related to this one */ 
# include "fact.h" 
# include "power.h" 
# include "taylor_exp.h"

/* one can define some constants */ 
# define L 2 
# define C 13 

/* we declare the functions */ 
extern void menu(); 

# endif /* MENU_H */

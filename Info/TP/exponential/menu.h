/* The first two lines and the last one are made so that the preprocessor will read this file only once */
# ifndef MENU_H
# define MENU_H

/* we include the standard libraries we will use */
# include <stdio.h> /* for printf and scanf */
# include <stdlib.h> /* for exit(), EXIT_SUCCESS and EXIT_FAILURE */
# include <time.h> /* to manage the running times */

/* we include the other files related to this one */
# include "fact.h"
# include "power.h"
# include "exp_basic.h"
# include "exp_better.h"

/* one can define some constants */

/* one define global variables */
clock_t initial_time; /* Initial time in micro-seconds */
clock_t final_time; /* Final time in micro-seconds */
float cpu_time; /* Total time in seconds */

/* we declare the functions */
extern void menu(); 

# endif /* MENU_H */

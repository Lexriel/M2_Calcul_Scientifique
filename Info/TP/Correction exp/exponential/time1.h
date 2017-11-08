# ifndef TIME1_H
# define TIME1_H

# include <stdio.h> /* for printf and scanf */
# include <stdlib.h> /* for exit(), EXIT_SUCCESS and EXIT_FAILURE */
# include <time.h> /* to manage the running times */

# include "exp_basic.h"

clock_t initial_time; /* Initial time in micro-seconds */
clock_t final_time; /* Final time in micro-seconds */
float cpu_time; /* Total time in seconds */

# endif /* TIME1_H */

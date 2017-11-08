# ifndef DATE_H
# define DATE_H
# include <stdio.h>
# include <time.h>
# include <stdlib.h>
# include <string.h>

// I present the structure of tm, which gives time data, used in the program 'one_to_all.c'.

/* ======================= Structure of tm ======================= 

{   int tm_sec;        seconds (0,59) 
    int tm_min;        minutes (0,59)
    int tm_hour;       hours since midnight (0,23)
    int tm_mday;       day of the month (0,31)
    int tm_mon;        month since January (0,11)
    int tm_year;       years happened since 1900
    int tm_wday;       day since Sunday (0,6)
    int tm_tm_yday;    day since the 1st January (0,365)
    int tm_isdst;      jet lag
};
   =============================================================== */

/* These strings will be used in order to give the date in letters. */ 
const char * day[] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};

const char * month[] = {"January", "February", "Marth", "April", "May", "June", "July", "August", "September", "October", "November", "December"};

# endif /* DATE_H */

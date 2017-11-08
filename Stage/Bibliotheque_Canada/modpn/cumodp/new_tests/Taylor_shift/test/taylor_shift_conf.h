#ifndef _TAYLOR_SHIFT_CONF_H_
#define _TAYLOR_SHIFT_CONF_H_

// Libraries :
# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <time.h>
# include <ctime>
# include <math.h>
# include <unistd.h>
# include <iostream>
# include <fstream>
using namespace std;


// Number of threads per block (size of a block) :
#define NB_THREADS 512

//#define MAX_LEVEL 25

const int Tmul = 512;
#define BASE_1 31

typedef int sfixn;

// Debuging flags
#define DEBUG 0

#endif // _TAYLOR_SHIFT_CONF_H_

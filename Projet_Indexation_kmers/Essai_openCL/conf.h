# ifndef CONF_H
# define CONF_H

// Librairies
# include <fcntl.h>
# include <math.h>
# include <unistd.h>
# include <sys/types.h>
# include <sys/stat.h>
# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <time.h>
# include <iostream>
# include <fstream>
using namespace std;

# ifdef OPENCL_H
# include <OpenCL/opencl.h> // Apple
# else
# include <CL/cl.h>  // ATI, nvidia
# endif

// Programmes pour OpenCL
# include "handle.c"
# include "setup.c"
# include "time_tools.c"

# define DATA_SIZE (256*1024)

# endif /* CONF_H */

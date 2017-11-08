#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// Libraries :
# include <stdio.h>
# include <stdlib.h>
# include <time.h>
# include <sys/time.h>
# include <unistd.h>

// Thread block size :
#define BLOCK_SIZE 18 // on modifie ce paramètre là pour tirer des conclusions

// Matrix dimensions :
#define WA BLOCK_SIZE*15 // Matrix A width
#define HA WA // Matrix A height
#define WB WA // Matrix B width
#define HB WA // Matrix B height
#define WC WB // Matrix C width
#define HC HA // Matrix C height

#endif // _MATRIXMUL_H_

#ifndef _TAYLOR_SHIFT_CPU_
#define _TAYLOR_SHIFT_CPU_
#include "taylor_shift_conf.h"


// error message if there is a lack of arguments to make the program
  void error_message(sfixn m);


// computes the nomber of blocks
  sfixn number_of_blocks(sfixn n);


// stocks a file in an array
  void stock_file_in_array(char* filename, sfixn n, sfixn* & a);


// stockes the array of Newton's coefficients in a file
  void stock_array_in_file(const char *name_file, sfixn *T, sfixn size);


// computes the number of lines of a file
  sfixn size_file(char* filename);


// display of an array
  void display_array(sfixn *T, sfixn size);


// addition of two arrays
  void add_arrays(sfixn *res, sfixn *T1, sfixn *T2, sfixn size, sfixn p);


// Horner's method to compute g(x) = f(x+1) (equivalent to Shaw & Traub's method for a=1)
  void horner_shift_CPU(sfixn *Polynomial, sfixn *Polynomial_shift, sfixn n, sfixn p);


#endif // _TAYLOR_SHIFT_CPU_

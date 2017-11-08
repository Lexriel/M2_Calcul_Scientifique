# ifndef _TAYLOR_SHIFT_CPU_
# define _TAYLOR_SHIFT_CPU_

// error message if there is a lack of arguments to make the program
void error_message(int m);


// computes the nomber of blocks
  int number_of_blocks(int n);


// stocks a file in an array
  void stock_file_in_array(char* filename, int n, int* & a);


// stockes the array of Newton's coefficients in a file
  void stock_array_in_file(const char *name_file, int *T, int size);


// computes the number of lines of a file
  int size_file(char* filename);


// display of an array
  void display_array(int *T, int size);


// addition of two arrays
  void add_arrays(int *res, int *T1, int *T2, int size, int p);


// Horner's method to compute g(x) = f(x+1) (equivalent to Shaw & Traub's method for a=1)
  void horner_shift_CPU(int *Polynomial, int *Polynomial_shift, int n, int p);

#endif // _TAYLOR_SHIFT_CPU_

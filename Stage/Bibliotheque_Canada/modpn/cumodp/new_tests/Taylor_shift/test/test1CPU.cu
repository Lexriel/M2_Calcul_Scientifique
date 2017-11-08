# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <time.h>
# include <math.h>
# include <unistd.h>
# include <iostream>
# include <fstream>
using namespace std;


/* Important to notice :

  n  : number of coefficients of the polynomial considered
 n-1 : degree of the polynomial considered
  p  : prime number, it must be greater than n
*/




// error message if there is a lack of arguments to make the program
void error_message(int m)
{
  if (m < 3)
    {
      printf("********** ERROR, not enough arguments ! **********\nThe program works with the following parameters:\n\n");
      printf("1st parameter : file containing coefficients of the polynomial you want to consider.\n");
      printf("2nd parameter : prime number p.\n");
      
      exit(1);
    }
}


// function modulo (faster than using %p)
int double_mul_mod(int a, int b, int p, double pinv)
{
  int q = (int) ((((double) a) * ((double) b)) * pinv);
  int res = a * b - q * p;

  return (res < 0) ? (-res) : res;
}


// creates an array of the sequence of the factorials until n modulo p (! size of the array = n+1)
void create_factorial(int *Factorial, int n, int p, double pinv)
{
  int k;
  Factorial[0] = 1;
  Factorial[1] = 1;
  for (k=2; k<n+1; k++)
    Factorial[k] = double_mul_mod(k, Factorial[k-1], p, pinv);
}


// creates an array of the Newton's Binomials until n modulo p (! size of the array = n+1)
void create_binomial_CPU(int *Binomial, int *Factorial, int *Inverse_p, int n, int p, double pinv)
{
  int k, l;
  int temp;
  for (k=0; k<n+1; k++)     // we create together two parts of the array Binomial
  {
    l = n-k;
    if (k>l)                // and finally this loop has just n/2 steps
      break;
    temp = double_mul_mod(Factorial[k], Factorial[l], p, pinv);
    Binomial[k] = double_mul_mod(Factorial[n], Inverse_p[temp], p, pinv);
    Binomial[l] = Binomial[k];
  }
}


// stocks a file in an array
void stock_file_in_array(char* filename, int n, int* & a)
{
  ifstream data_file;
  int i;
  data_file.open(filename);

  if (! data_file.is_open())
    {
      printf("\n Error while reading the file %s. Please check if it exists !\n", filename);
      exit(1);
    }

  a = (int*) malloc (n*sizeof(int));

  for (i=0; i<n; i++)
    data_file >> a[i];

  data_file.close();
}


// stockes the array of Newton's coefficients in a file
void stock_array_in_file(const char *name_file, int *T, int size)
{
  int i;
  FILE* file = NULL;

  file = fopen(name_file, "w+");
  if (file == NULL)
  {
    printf("error when opening the file !\n");
    exit(1);
  }
 
  // writting the file  
  fprintf(file, "%d", T[0]);
  for (i=1; i<size; i++)
    fprintf(file, "\n%d", T[i]);
  fclose(file);
}


// computes the number of lines of a file
int size_file(char* filename)
{
  int size = 0;
  ifstream in(filename);
  std::string line;

  while(std::getline(in, line))
    size++;
  in.close();

  return size;
}


// display of an array
void display_array(int *T, int size)
{
  int k;
  printf("[ ");
  for (k=0; k<size; k++)
    printf("%d ", T[k]);
  printf("] \n");
}


// create an array of the inverse numbers in Z/pZ
void inverse_p(int *T, int p, double pinv)
{
  int i,j;
  T[0] = 0;
  T[1] = 1;
  for (i=2; i<p; i++)
    for (j=2; j<p; j++)
      if (double_mul_mod(i, j, p, pinv) == 1)
      {
        T[i] = j;
        T[j] = i;
        break;
      }
}


// create the array of the coefficients of (x+1)^k for several k
void develop_xshift(int *T, int e, int *Factorial, int *Inverse_p, int p, double pinv, int n)
{
  int i, j;
  int *bin;
  int power2_i = 2;
  T[0] = 1;              // for (x+1)^0
  T[1] = 1;              // for (x+1)^1
  T[n] = 1;

  for (i=2; i<e+1; i++)  // for (x+1)^i with i in (2,e)
  {
    bin = (int*) malloc((power2_i+1)*sizeof(int));
    create_binomial_CPU(bin, Factorial, Inverse_p, power2_i, p, pinv);
    for (j=0; j<power2_i; j++)
      T[power2_i+j] = bin[j];
    free(bin);
    power2_i *= 2;
  }
}


// create the product of two arrays representing polynomials
void conv_prod(int *res, int *T1, int *T2, int m, int p)//, double pinv)
{
  int i, j, k;//, temp;
  
  for (i=0; i<m; i++)
    for (j=0; j<m; j++)
    {
      k = (i+j) % m;
//      temp = double_mul_mod(T1[i], T2[j], p, pinv);
      res[k] = (res[k] + T1[i]*T2[j]) % p;
  }
}


// addition of two arrays
void add_arrays(int *res, int *T1, int *T2, int size, int p)
{
  int i;
  for (i=0; i<size; i++)
    res[i] = (T1[i] + T2[i]) % p;
}


// creates Polynomial_shift(x) = Polynomial(x+1)
void create_polynomial_shift_CPU(int *Polynomial, int *T, int *Monomial_shift, int n, int p, double pinv, int local_n)
{
  int i, j;
  int *Temp1, *Temp2, *Temp3, *res;

  if (local_n != 1)  
  {
    create_polynomial_shift_CPU(Polynomial, T, Monomial_shift, n, p, pinv, local_n/2);

    if (local_n != n)
    {
      Temp1 = (int*) calloc(2*local_n, sizeof(int));
      Temp2 = (int*) calloc(2*local_n, sizeof(int));
      Temp3 = (int*) calloc(2*local_n, sizeof(int));
      res   = (int*) malloc(2*local_n * sizeof(int));

      memcpy(Temp3, Monomial_shift + local_n, (local_n+1) * sizeof(int));

      for (j=0; j<n; j+=2*local_n)
      {
        memcpy(Temp1, T + j, local_n*sizeof(int));
        memcpy(Temp2, T + local_n + j, local_n*sizeof(int));
        for (i=0; i<2*local_n; i++)
          res[i] = 0;
        conv_prod(res, Temp3, Temp2, 2*local_n, p);
  printf("conv_prod :\n");
  //display_array(res, 2*local_n);
        add_arrays(res, res, Temp1, 2*local_n, p);
        memcpy(T + j, res, 2*local_n*sizeof(int));
      }

      free(Temp1);
      free(Temp2);
      free(Temp3);
      free(res);
    }
  }

  else // (local_n == 1)
  {
    for (i=0; i<n; i+=2)
    {
      T[i] = Polynomial[i] + Polynomial[i+1];
      T[i+1] = Polynomial[i+1];
    }
  }

printf("\nlocal_n = %d\n", local_n);
//printf("T : \n");
//display_array(T, n);

}


// Horner's method to compute g(x) = f(x+1) (equivalent to Shaw & Traub's method for a=1)
void horner_shift_CPU(int *Polynomial, int *Polynomial_shift, int n, int p)
{
  int i;
//  int k = 0;
  int *temp;
/*  int percent[10];

  for (i=1; i<11; i++)
    percent[i-1] =  (i*n) / 10;*/

  temp = (int*) calloc (n, sizeof(int));

  Polynomial_shift[0] = Polynomial[n-1];

  for (i=1; i<n/*percent[0]*/; i++)
  {
    memcpy(temp+1, Polynomial_shift, i*sizeof(int));
    add_arrays(Polynomial_shift, Polynomial_shift, temp, n, p);
    Polynomial_shift[0] = (Polynomial_shift[0] + Polynomial[n-1-i]) % p;
  }
//  printf("\n%3d %% of horner_shift done.\n", 10);

/*  for (k=0; k<9; k++)
  {
    for (i=percent[k]; i<percent[k+1]; i++)
    {
      memcpy(temp+1, Polynomial_shift, i*sizeof(int));
      add_arrays(Polynomial_shift, Polynomial_shift, temp, n, p);
      Polynomial_shift[0] = (Polynomial_shift[0] + Polynomial[n-1-i]) % p;
    }
    printf("%3d %% of horner_shift done.\n", 10*(k+2));
  }*/

  free(temp);
}


// MAIN
int main(int argc, char* argv[])
{
  // temporal data
//  clock_t initial_time;   /* Initial time in micro-seconds */
//  clock_t final_time;     /* Final time in micro-seconds */
  float cpu_time;         /* Total time of the cpu in seconds */
  // TIME
  cudaEvent_t start, stop;     /* Initial and final time */
  cudaEventCreate(&start); 
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // TIME
//  initial_time = clock();  // for the CPU


  //  int m, i, j, k, size;
  int n, p, e;
//  int *Factorial, *Binomial, *Inverse_p;
  int *Polynomial, *Polynomial_shift2;//, *Polynomial_shift1;
//  int *Monomial_shift;
//  int *temp;

//  const char newton[12] = "newton.dat\0";
  double pinv;


  // beginning parameters
  error_message(argc);
  p = atoi(argv[2]);
  pinv = (double) 1/p;
  n = size_file(argv[1]);
  e = (int) log2((double) n);
  stock_file_in_array(argv[1], n, Polynomial);


  printf("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\
                     TAYLOR_SHIFT ON CPU\n\n\
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");
  printf("\nPARAMETERS :\n------------\n\n");
  printf("  * n = %d\n", n);
  printf("  * e = %d\n", e);
  printf("  * p = %d\n", p);
  printf("  * pinv = %0.20lf\n", pinv);


  // allocation of memory
/*  Factorial         = (int*) malloc((n+1)*sizeof(int));
  Binomial          = (int*) malloc((n+1)*sizeof(int));
  Inverse_p         = (int*) malloc(p*sizeof(int));
  Monomial_shift    = (int*) malloc((n+1)*sizeof(int));
  Polynomial_shift1 = (int*) calloc(n,sizeof(int));*/
  Polynomial_shift2 = (int*) calloc(n,sizeof(int));
//  temp              = (int*) malloc(n*sizeof(int));

  // instructions
/*  create_factorial(Factorial, n, p, pinv);
  printf("\n      --> create_factorial done\n");
  inverse_p(Inverse_p, p, pinv);
  printf("\n      --> inverse_p done\n");
  create_binomial_CPU(Binomial, Factorial, Inverse_p, n, p, pinv);
  printf("\n      --> create_binomial done\n");
  develop_xshift(Monomial_shift, e, Factorial, Inverse_p, p, pinv, n);
  printf("\n      --> develop_xshift done\n");
  stock_array_in_file(newton, Monomial_shift, n+1);
  printf("\n      --> stock_array_in_file done\n");
  memcpy(temp, Polynomial, n*sizeof(int));
  create_polynomial_shift_CPU(Polynomial, Polynomial_shift1, Monomial_shift, n, p, pinv, n);
  printf("\n      --> create_polynomial_shift done\n");*/
  horner_shift_CPU(Polynomial, Polynomial_shift2, n, p);
  printf("\n   --> horner_shift done\n");


  // save in files the polynomial_shift
  char name_file[30];
/*  sprintf(name_file, "Pol%d.shiftDNC.dat\0", e);
  stock_array_in_file(name_file, Polynomial_shift1, n);*/
  sprintf(name_file, "Pol%d.shiftHOR.dat\0", e);
  stock_array_in_file(name_file, Polynomial_shift2, n);


  // tests
/*  printf("\n\nTEST values and arrays\n======================\n\n");
  printf("\nFactorial :\n");
  display_array(Factorial, n+1);
  printf("\nBinomial :\n");
  display_array(Binomial, n+1);*/
  printf("\nPolynomial :\n");
//  display_array(Polynomial, n);
/*  printf("\nInverse_p :\n");
  display_array(Inverse_p, p);
  printf("\nMonomial_shift :\n");
  display_array(Monomial_shift, n+1);

  printf("\n\nTEST taylor shift methods\n=========================\n\n");
  printf("Polynomial_shift (Divide & Conquer method) :\n");
  display_array(Polynomial_shift1, n);*/
  printf("\nPolynomial_shift (Horner's method) :\n");
//  display_array(Polynomial_shift2, n);

/*  printf("\n\nTEST convolution_product\n========================\n\n");
  int T1[4]  = {1,2,0,3};
  int T2[4]  = {2,-1,-1,0};
  int res[4] = {0,0,0,0};
  conv_prod(res, T1, T2, 4, p);
  printf("conv_prod(T1*T2) :\n");
  display_array(res, 4);*/
//  printf("\n\n");


  // free memory
/*  free(Factorial);
  free(Binomial);*/
  free(Polynomial);
//  free(Polynomial_shift1);
  free(Polynomial_shift2);
/*  free(Inverse_p);
  free(Monomial_shift);
  free(temp);*/


  // TIME
//  final_time = clock();  // for the CPU
//  cpu_time = (float) (final_time - initial_time) / CLOCKS_PER_SEC;
//  printf("\n  * cpu_time = %.3f s\n\n", cpu_time);
  // TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cpu_time, start, stop);
  cudaEventDestroy(stop);
  cpu_time /= 1000.0;
  printf("\n  * cpu_time = %.6f s\n\n", cpu_time);

  return 0;
}

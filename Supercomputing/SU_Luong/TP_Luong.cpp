# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <time.h>
# define SIZE 10


/* A usefull procedure which displays an array at the screen,
   I will use it in the exercises 2, 3 and 4, whereas it is
   asked only in the exercice 4.                              */

void display_array(int* T, int n)
{
  int i;
  printf("[ ");
  for (i=0; i<n; i++)
    printf("%d ", T[i]);
  printf("]\n");
}

/* ====================================================

            Exercice 1 : Functions and procedure

   ==================================================== */

// 1.1 : Divisions

int divide(int a, int b)
{
  return a/b;
}

float divide_not_truncated(int a, int b)
{
  float res;
  res =  ((float) a) / ((float) b);
  return res;
}


// 1.2 : Parameters and passing modes

int power(int a, int b)
{
  int i, res;
  res = 1;
  if (b != 0)
    for (i=0; i<b; i++)
      res = res*a;
    return res;
}

void power2(int a, int b, int& result)
{
  int i, res;
  result = 1;
  if (b != 0)    for (i=0; i<b; i++)
      result = result*a;
}

void power3(int a, int b, int* result)
{
  int i, res;
  *result = 1;
  if (b != 0)
    for (i=0; i<b; i++)
      *result = *result*a;
}

// 1.3 Random number generator

int random_number0(int max)
{
  int res;
 
  res = rand() % max;
  return res;
}

int random_number(int min, int max)
{
  int res;
  res = rand() % (max-min);
  res = min + res;
  return res;
}

// Procedure of the exercice 1 in order to use functions and procedures
// we create.

void exercice1()
{
  int c, e, f, g;
  float d;
  int result;

  c = divide(12,7);
  printf("\nWith the function divide : the truncated result of 12/7 is : %d.\n", c);
  
  d = divide_not_truncated(12,7);
  printf("With the function divide_not_truncated : the approximate result of 12/7 is : %f.\n\n", d);

  e = power(2, 3);
  printf("With the function power : 2^3 = %d\n", e);
 
  power2(2, 3, result);
  printf("With the procedure power2 : 2^3 = %d\n", result);

  power3(2, 3, &result);
  printf("With the procedure power3 : 2^3 = %d\n\n", result);

  f = random_number0(100);
  printf("With the function random_number0, which gives a random number between 0 and 100 here, we get the number : %d.\n", f);

  g = random_number(10, 20);
  printf("With the function random_number, which gives a random number between 10 and 20 here, we get the number : %d.\n\n", g);
}

/* ====================================================

            Exercice 2 : Allocation of arrays

   ==================================================== */

// 2.1 : Static arrays (in the procedure exercice2)

// 2.2 : Dynamic arrays

void init_array(int* T, int n)
{
  int i;
  for (i=0; i<n; i++)
    T[i] = random_number0(100);
}

void compute_power(int* T, int n, int a)
{
  int i;
  for (i=0; i<n; i++)
    T[i] = power(T[i], a);
}

// Procedure of the exercice 2 in order to use functions and procedures
// we create.

void exercice2(int n)
{
  int T[5];
  int i, N, a, b;
  int T2[SIZE], T7[SIZE];
  int *T3, *T4, *T5, *T6;


  printf("\nWe create a random array of 5 integers : T = ");
  for (i=0; i<5; i++)
    T[i] = random_number0(100);
  display_array(T, 5); // this procedure displays the array (see exercice 4 for more details)


  printf("We create a random array of %d integers : T2 = ", SIZE);
  for (i=0; i<SIZE; i++)
    T2[i] = random_number0(10);
  display_array(T2, SIZE);
  printf("\n");


  N = random_number0(20);
  T3 = (int*) malloc(N*sizeof(int));
  printf("We create a random array of %d integers, which contains integers between 0 and 100 : T3 = ", N);
  for (i=0; i<N; i++)
    T3[i] = random_number0(100);
  display_array(T3, N);
  printf("\n");
  free(T3);

   
  T4 = (int*) malloc(n*sizeof(int));
  printf("We create an array of %d integers, which contains integers between 0 and 100 : T4 = ", n);
  for (i=0; i<n; i++)
    T4[i] = random_number0(100);
  display_array(T4, n);
  printf("\n");
  free(T4);


  T5 = (int *) malloc(n*sizeof(int));
  init_array(T5, n);
  printf("The procedure init_array creates the array : T5 = ");
  display_array(T5, n);
  printf("\n");

  
  T6 = (int *) malloc(n*sizeof(int));
  memcpy(T6, T5, n*sizeof(int));
  printf("The memcpy procedure copies T5 in T6 : T6 = ");
  display_array(T6, n);
  printf("\n");

  free(T5);
  free(T6);


  compute_power(T2, SIZE, 3);
    printf("The compute_power procedure puts each element of the array T2 to the power 3 here, so now : T2 = ");
  display_array(T2, SIZE);
  printf("\n");

}


/* ====================================================

            Exercice 3 : Manipulation of arrays

   ==================================================== */

// 3.1 : Prefix sum

void prefix_sum(const int* input, int* output, int n)
{
  int i, c;
  output[0] = input[0];
  for (i=1; i<n; i++)
    output[i] = output[i-1] + input[i];
}

// 3.2 : Matrix and array structures
/* (procedure of the exercice 3 in order to use functions and procedures
   we create). */

void exercice3(int m, int n){
  int *T9, *T10;
  int i, j, count;
  int **Mat;
  int *Mat_in_line;

  T9 = (int*) malloc(n*sizeof(int));
  printf("\nWe define an array of %d elements : T9 = ", n);
  for (i=0; i<n; i++)
    T9[i] = random_number0(10);
  display_array(T9, n);
  printf("\n");

  T10 = (int*) malloc(n*sizeof(int));
  prefix_sum(T9, T10, n);
  printf("With the prefix_sum procedure applied on T9, we get : T10 = ", n);
  display_array(T10, n);
  printf("\n");

  Mat = (int**) malloc(m*sizeof(int*));
  for (i=0; i<m; i++)
    Mat[i] = (int*) malloc(n*sizeof(int));

  printf("Summary: you enter in command line m = %d and n = %d.\n", m, n);

  memcpy(Mat[0], T9, n*sizeof(int));
  printf("The memcpy procedure copies T9 in Mat[0] : Mat[0] = ");
  display_array(Mat[0], n);
  printf("\n");

  for (i=1; i<m; i++)
    prefix_sum(Mat[i-1], Mat[i], n);

  printf("We perform %d iterations of the prefix_sum operation and obtain the following matrix Mat (we represent its lines) :\n", m);
  for (i=0; i<m; i++)
    display_array(Mat[i], n);
  printf("\n");


  printf("Now, we store this matrix in a one-dimensional vector Mat_in_line :\nMat_in_line = ");
  Mat_in_line = (int*) malloc(m*n*sizeof(int));
  count = 0;
  for (i=0; i<m; i++)
    for (j=0; j<n; j++)
      {
	Mat_in_line[count] = Mat[i][j];
	count++;
      }
  display_array(Mat_in_line, m*n);
  printf("\n");


  for (i=0; i<n; i++)
    free(Mat[i]);
  free(Mat);
  free(Mat_in_line);

}

/* ====================================================

         Exercice 4 : Advanced algorithm concepts

   ==================================================== */

// 4.1 : Permutations and combinations

void init_permutation(int* T, int n)
{
  int i, k, permut;
  for (i=0; i<n; i++)
    {
      k = random_number(i, n);
  
      permut = T[i];
      T[i] = T[k];
      T[k] = permut;
    }
}

void two_permutations(int* T, int a, int b)
{
  int k;
  k = T[a];
  T[a] = T[b];
  T[b] = k;
}

void single_permutations(int* T, int n)
{
  int i, j, k;
  for (i=0; i<n; i++)
    for(j=i; j<n; j++)
      {
	if (j != i)
	  {
	    two_permutations(T, i, j);
	    display_array(T, n);
	    two_permutations(T, i, j);
	  }
      }
  printf("\n");
} 


// 4.2 : Recursion

unsigned long factorial(int n)
{
  if (n<=1)
    return 1;
  else
    return n*factorial(n-1);
       /* We could also give this as the factorial function :
          "return n<=1 ? 1 : n*factorial(n-1);"               */
}

unsigned long factorial2(int n, unsigned long result)
{
  if (n<=1)
    return result;
  else
    {
      result = result*n;
      return factorial2(n-1, result);
    }
}

// 4.3 : All permutations

void permutation(int *T, int n, int i)
{
  int k;

  if (i == n-1)
    display_array(T, n);
  else
    {
      for (k=i; k<n; k++)
        {
          two_permutations(T, k, i);
          permutation(T, n, i+1);
          two_permutations(T, k, i);
        }
    }
}


// Procedure of the exercice 4 in order to use functions and procedures
// we create.

void exercice4(int n)
{
  int *T;
  int i;
  unsigned long k;
  unsigned long result=1;


  T = (int*) malloc(n*sizeof(int));
  for (i=0; i<n; i++)
    T[i] = i;

  printf("\nT = ");
  display_array(T, n);
  printf("\n");
  init_permutation(T, n);
  printf("With the init_permutation procedure for T, we obtain randomly :\n");
  display_array(T, n);
  printf("\n");

  for (i=0; i<n; i++) // we put that in order to re-initialize T
    T[i] = i;
  printf("With the procedure single_permutations, we print all the possibity after doing just one permutation on T, we obtain:\n");
  single_permutations(T, n);
  


  printf("Thanks to the recursive factorial and factorial2 functions, we can compute %d! :\n", n);

  k = factorial(n);
  printf("   * With factorial  : %d! = %lu\n", n, k);

  k = factorial2(n, result);
  printf("   * With factorial2 : %d! = %lu\n", n, k);

  printf("\nWith the recursive permutation procedure, we can display the %lu permutations of T, we obtain :\n", k);
  for (i=0; i<n; i++)
    T[i] = i;
  permutation(T, n, 0);
  printf("\n");

  free(T);
  


}

/* ====================================================

                           Main

   ==================================================== */

int main (int argc, char* argv[])
{
  int choice, m, n;
  
  srand(time(NULL));
  
  printf("Chose the procedure of the exercise you want to see :\n\n");
  printf("For the exercise 1, enter 1.\n");
  printf("For the exercise 2, enter 2.\n");
  printf("For the exercise 3, enter 3.\n");
  printf("For the exercise 4, enter 4.\n");
  printf("If you enter anything else, you'll quit this program.\n\n");
  printf("Enter your choice : ");
  scanf("%d", &choice);
  
  
  switch (choice)
    {
    case 1 :
      exercice1();
      break;

    case 2 :
      if (argc < 2)
	{
	  printf("\nGive at least one argument in the command line please.\n\n");
	  exit(1);
	}
      n = atoi(argv[1]);
      exercice2(n); 
      break;

    case 3 :
      if (argc < 3)
	{
	  printf("\nGive at least two arguments in the command line please.\n\n");
	  exit(1);
	}
      m = atoi(argv[1]);
      n = atoi(argv[2]);
      exercice3(m, n);
      break;

    case 4 :
      if (argc < 2)
	{
	  printf("\nGive at least one argument in the command line please.\n\n");
	  exit(1);
	}
      n = atoi(argv[1]);
      exercice4(n);
      break;

    default :
      printf("\n           Good bye !\n\n");
      return 0;
      break;
    }
  







  

}  

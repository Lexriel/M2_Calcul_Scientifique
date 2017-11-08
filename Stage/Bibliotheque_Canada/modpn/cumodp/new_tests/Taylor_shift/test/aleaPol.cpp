# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <time.h>
# include <math.h>
# include <unistd.h>
# include <iostream>
# include <fstream>
using namespace std;


// error message if there is a lack of arguments to make the program
void error_message(int m)
{
  if (m < 3)
    {
      printf("********** ERROR, not enough arguments ! **********\nThe program works with the following parameters:\n\n");
      printf("1st parameter : number e of the maximal length 2^e of the polynomial we want to build.\n");
      printf("2nd parameter : m modulo for the random.\n");
      
      exit(1);
    }
}


// stockes an array in a file
void stock_array_in_file(const char *name_file, int size, int m)
{
  int i, a;
  FILE* file = NULL;

  file = fopen(name_file, "w+");
  if (file == NULL)
    {
      printf("error when opening the file !\n");
      exit(1);
    }
 
  // writting the file 
  a = rand() % m;
  fprintf(file, "%d", a);
  for (i=1; i<size-1; i++)
    {
      a = rand() % m;
      fprintf(file, "\n%d", a);
    }
  fprintf(file, "\n%d", 1);

  fclose(file);
}


void stock_array_in_file2(const char *name_file, int size)
{
  int i, a;
  FILE* file = NULL;

  file = fopen(name_file, "w+");
  if (file == NULL)
    {
      printf("error when opening the file !\n");
      exit(1);
    }
 
  // writting the file 
  fprintf(file, "1");
  for (i=1; i<size; i++)
    fprintf(file, "\n0");

  fclose(file);
}


// create a random polynomial
void random_poly(int power2, int e, int m)
{

  int i;
  char file[20];
  
  sprintf(file, "Pol%d.dat\0", e);
  // printf("file = %s\n", file);

  stock_array_in_file(file, power2, m);
}


void poly1(int power2, int e)
{

  int i;
  char file[24];
  
  sprintf(file, "Pol%d.dat\0", e);
  // printf("file = %s\n", file);

  stock_array_in_file2(file, power2);
}


// main
int main(int argc, char* argv[])
{
  int j, m, elimit;
  int power2 = 4;

  srand ( time(NULL) );

  elimit = atoi(argv[1]);
  m = atoi(argv[2]);      // modulo

  for (j=3; j<elimit; j++)
    {
      power2 *= 2;
      random_poly(power2, j, m);
      //    poly1(power2, j);
    }

  //  printf("\naleat_pol.cpp done\n\n");

  return 0;
}

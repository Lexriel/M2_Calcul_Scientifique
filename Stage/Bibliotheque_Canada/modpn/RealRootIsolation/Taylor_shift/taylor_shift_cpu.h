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


// computes the nomber of blocks
int number_of_blocks(int n)
{
  int res;
  res = n/NB_THREADS;
  if ( n % NB_THREADS != 0)
    res++;
  return res;
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


// addition of two arrays
void add_arrays(int *res, int *T1, int *T2, int size, int p)
{
  int i;
  for (i=0; i<size; i++)
    res[i] = (T1[i] + T2[i]) % p;
}


// Horner's method to compute g(x) = f(x+1) (equivalent to Shaw & Traub's method for a=1)
void horner_shift_CPU(int *Polynomial, int *Polynomial_shift, int n, int p)
{
  int i;
  int *temp;
  temp = (int*) calloc (n, sizeof(int));

  Polynomial_shift[0] = Polynomial[n-1];

  for (i=1; i<n; i++)
  {
    memcpy(temp+1, Polynomial_shift, i*sizeof(int));
    add_arrays(Polynomial_shift, Polynomial_shift, temp, n, p);
    Polynomial_shift[0] = (Polynomial_shift[0] + Polynomial[n-1-i]) % p;
  }
  
  free(temp);
}

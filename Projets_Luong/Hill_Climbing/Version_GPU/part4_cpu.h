void loadInstances(char* filename, int& n, int* & a, int* &b)
{
  ifstream data_file;
  int i, j;
  data_file.open(filename);
  if (! data_file.is_open())
    {
      printf("\n Error while reading the file %s. Please check if it exists !\n", filename);
      exit(1);
    }

  data_file >> n;

  // ***************** dynamic memory allocation *******************//
  a = (int*) malloc (sizeof(int)*n*n);
  b = (int*) malloc (sizeof(int)*n*n);

  //*************** read flows and distance matrices ****************//
  for (i=0;i<n;i++)
    for (j=0;j<n;j++)
      data_file >> a[i*n+j];

  for (i=0;i<n;i++)
    for (j=0;j<n;j++)
      data_file >> b[i*n+j];

  data_file.close();
}

void create(int* solution, int n)
{
  int random, temp;
  for (int i=0; i<n; i++)
    solution[i] = i;
  
  // we want a random permutation so we shuffle

  for (int i=0; i<n; i++)
    {
      random = rand()%(n-i) + i;
      temp = solution[i];
      solution[i] = solution[random];
      solution[random] = temp;
    }
}
    
int evaluation (int* a, int* b, int* solution, int n)
{
  int cost = 0;
  for (int i=0; i<n; i++)
    for (int j=0; j<n; j++)
      cost += a[i*n+j]*b[ solution[i]*n +solution[j] ];
	  
  return cost;
}

int compute_delta_cpu(int* a, int* b, int* p, int i, int j, int n)
{
  int d; int k;
  
  d = ( a[i*n+i] - a[j*n+j] ) * ( b[ p[j]*n + p[j] ] - b[ p[i]*n + p[i] ] ) +
    ( a[i*n+j] - a[j*n+i] ) * ( b[ p[j]*n + p[i] ] - b[ p[i]*n + p[j] ] );
  
  for (k=0; k<n; k++)
    if (k != i && k != j)
      d = d +( a[k*n+i] - a[k*n+j] ) * ( b[ p[k]*n + p[j] ] - b[ p[k]*n + p[i] ] ) +
	( a[i*n+k] - a[j*n+k] ) * ( b[ p[j]*n + p[k] ] - b[ p[i]*n + p[k] ] );
  
  return d;
}



// fonction cherchant le minimum d'un tableau
void min_tab(int *ij, int *T, int size, int n, int &condition)
{
  int p, I;
  int k = 0;
  int temp = 0;
  int prec = 0;

  I = -1; // ce -1 ne signifie rien mais permettra d'affecter 1 au booléen condition si besoin
  for (p=0; p<size; p++)
    {
      if (T[p] < ij[2])
	{
	  ij[2] = T[p];
	  I = p;
	}
    }

  if (I != -1)
    {
      k = 0;
      while ( I >= temp )
        {
          k++;
          prec = temp;
          temp = temp + (n-k);
        }
      k--;

      ij[0] = k; // i
      ij[1] = I - prec + k + 1; // j
    }
  else
    condition = 1;

}

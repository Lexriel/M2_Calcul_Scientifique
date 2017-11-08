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


int f(int k, int n)
{
  int i, M;
  M = 0;
  for (i=0; i <k; i++)
    M = M + (n-i-1);
  return M;
}


// fonction cherchant le minimum d'un tableau
void min_tab(int *ij, int *T, int size, int n)
{
  int m, k, coord; // coord = I
  ij[2] = 0;
  coord = -1;
  for (m=0; m<size; m++)
    {
      if (T[m] < ij[2])
        {
          ij[2] = T[m];
          coord = m;
        }
    }

  if (coord != -1)
    {
      k = 0;
      while ( coord >= f(k,n) )
        k++;
      k--;
  
      ij[0] = k; // i
      ij[1] = coord - f(k,n) + k + 1; // j
    }
  else
    {
      printf("pas de meilleur voisin\n");
      exit(1);
    }
}




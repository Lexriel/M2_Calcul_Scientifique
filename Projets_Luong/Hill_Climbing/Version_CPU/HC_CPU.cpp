#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
using namespace std;


// ouvre un fichier QAP et crée les matrices a et b ainsi que leur taille n
void loadInstances(char* filename, int& n, int* & a, int* &b)
{
  ifstream data_file;
  int i, j;
  data_file.open(filename);
  if (! data_file.is_open())
    {
      cout << "\n Error while reading the file " << filename << ". Please check if it exists !" << endl;
      exit(1);
    }

  data_file >> n;

  // ***************** dynamic memory allocation *******************//
  a=(int*) malloc (sizeof(int)*n*n);
  b=(int*) malloc (sizeof(int)*n*n);

  //*************** read flows and distance matrices ****************//
  for (i=0;i<n;i++)
    for (j=0;j<n;j++)
      data_file >> a[i*n+j];

  for (i=0;i<n;i++)
    for (j=0;j<n;j++)
      data_file >> b[i*n+j];

  data_file.close();
}


// creates a random permutation
void create(int* solution, int n)
{
  int random, temp;
  for (int i=0; i<n; i++)
    solution[i] = i;
  
  for (int i=0; i<n; i++)
    {
      random = rand()%(n-i) + i;
      temp = solution[i];
      solution[i] = solution[random];
      solution[random] = temp;
    }
}


// computes z(solution)
int evaluation (int* a, int* b, int* solution, int n)
{
  int cost = 0;
  for (int i=0; i<n; i++)
    for (int j=0; j<n; j++)
      cost += a[i*n+j]*b[ solution[i]*n +solution[j] ];

  return cost;
}


/* computes the decalage between z(solution) and z(p)
   ( i and j are the two indices to exchange (neighbor) ) */
int compute_delta(int* a, int* b, int* p, int i, int j, int n)
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


/* in this case, a neighbor evaluation does not make any copy
   or modification of the candidate solution */
int incremental_evaluation(int* a, int* b, int* candidate_solution, int i, int j, int n, int score)
{
  return score + compute_delta(a,b,candidate_solution,i,j,n);
}


/* =============================================================
   =                                                           =
   =                           MAIN                            =
   =                                                           =
   ============================================================= */

int main(int argc, char** argv)
{
  int *a, *b,n,*solution;
  int test[3];
  int i,j;
  int temp, score, former_score;


  srand(time(NULL));

  if (argc<2)
    {
      cout << "Please give a dat file" << endl;
      exit(1);
    }

  //lecture donnees
  loadInstances(argv[1],n,a,b);

  solution =(int *) malloc (n * sizeof(int));


  //solution de départ (au hasard) et calcul de son cout
  create(solution,n);
  score = evaluation (a,b,solution,n);
  former_score = score+1; // juste pour s'assurer de faire le while au moins une fois

  printf("Solution initiale : ");
  for (i=0; i<n; i++)
    printf ("%d ", solution[i]);
  printf("\nScore initial = %d\n\n", score);


  // tant que le nouveau score est inférieur au score précédent on peut en chercher un nouveau sinon on arrête
  while ( score < former_score )
    {
      test[0] = 0; // calcule le décalage d
      test[1] = 1; 
      test[2] = 1; // test[1] et test[2] donnent les 2 éléments i et j finaux qui seront choisis comme permutation
      
      for (i=0; i< n-1; i++)
	{
	  for (j = i+1; j< n; j++)
	    {
	      temp = compute_delta(a,b,solution,i,j,n); // on calcule le décalage des voisins
	      if ( test[0] > temp ) // si le nouveau décalage est plus petit que l'ancien, on le prend
		{
		  test[0] = temp;
		  test[1] = i;
		  test[2] = j;
		}
	    }
	}
      
      former_score = score;
      score = score + test[0];
      
      temp = solution[ test[1] ];
      solution[ test[1] ] = solution[ test[2] ];
      solution[ test[2] ] = temp;

      printf("Score temporaire = %d \n", score); 
      printf("Solution temporaire = ");
      for (i=0; i<n; i++)
	printf("%d ", solution[i]);
      printf("\n");
      
    }

  printf("\nSolution finale : ");
  for (i=0; i<n; i++)
    printf ("%d ", solution[i]);
  printf("\nScore finale = %d\n", score);

  free(a);
  free(b);
}

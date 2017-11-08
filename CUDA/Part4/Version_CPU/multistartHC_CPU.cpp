#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
using namespace std;


void loadInstances(char* filename, int& n, int* & a, int* &b)
{
  ifstream data_file;
  int i,j;
  data_file.open(filename);
  if (! data_file.is_open())
    {
      cout << "\n Error while reading the file " << filename << ". Please check if it exists !" << endl;
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


// i and j are the two indices to exchange (neighbor)

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


// in this case, a neighbor evaluation does not make any copy or modification of the candidate solution

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
  int *a, *b, n, *solution;
  int test[3];
  int i, j, k;
  int temp, score, former_score;

  int nb_solution = 5;
  int top_score;
  int *top_solution;
  

  srand(time(NULL));

  if (argc<2)
    {
      cout << "Please give a dat file" << endl;
      exit(1);
    }

  //lecture donnees
  loadInstances(argv[1],n,a,b);

  solution = (int *) malloc (n * sizeof(int));
  top_solution = (int *) malloc (n * sizeof(int));

// nb_solution itérations du Hill-Climbing
  for (k=0; k<nb_solution; k++)

    {
      //solution de depart (au hasard) et calcul de son cout

      create(solution,n);
      score = evaluation(a,b,solution,n);
      former_score = score + 1;

      // tant que le nouveau score est inférieur au score précédent on peut en chercher un nouveau sinon on arrête
      while ( score < former_score )
	{
	  test[0] = 0; // calcule le décalage d
	  test[1] = 1;
	  test[2] = 1;
      
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
	  score = score + test[0]; // incremental_evaluation(a,b, solution ,test[1],test[2],n, score);     
      
	  temp = solution[ test[1] ];
	  solution[ test[1] ] = solution[ test[2] ];
	  solution[ test[2] ] = temp;
      
	}

      printf("Solution du %d -ieme Hill Climbing : ", k+1);
      for (i=0; i<n; i++)
	printf("%d ", solution[i]);
      printf("\n");
      printf("score = %d \n\n", score);


      if ( k == 0)  //Le premier resultat de hill climbing est le meilleur score (de référence)
	{
	  top_score = score;
	  for (i=0; i<n; i++)
	    top_solution[i] = solution[i];
	}

      if ( k!=0 && score < top_score)
	{
	  top_score = score;
	  for (i=0; i<n; i++)
	    top_solution[i] = solution[i];
	}
	

    }

  //desallocation des tableaux et affichage de la meilleure solution trouvee

  free(a);
  free(b);


  printf("Solution du meilleur Hill Climbing : ");
  for (i=0; i<n; i++)
    printf("%d ", top_solution[i]);
  printf("\n");
  printf("Meilleur score : %d \n\n", top_score);

}

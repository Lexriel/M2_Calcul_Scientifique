#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>

/* Bornes de l'ensemble de Mandelbrot */
#define X_MIN -1.78
#define X_MAX 0.78
#define Y_MIN -0.961
#define Y_MAX 0.961


typedef struct {

  int nb_lig, nb_col; /* Dimensions */
  char * pixels; /* Matrice linearisee de pixels */

} Image;

static void erreur_options () {

  fprintf (stderr, "usage : ./mandel [options]\n\n");
  fprintf (stderr, "Options \t Signification \t\t Val. defaut\n\n");
  fprintf (stderr, "-n \t\t Nbre iter. \t\t 100\n");
  fprintf (stderr, "-b \t\t Bornes \t\t -1.78 0.78 -0.961 0.961\n");
  fprintf (stderr, "-d \t\t Dimensions \t\t 1024 768\n");
  fprintf (stderr, "-f \t\t Fichier \t\t /tmp/mandel.ppm\n");
  exit (1);
}

static void analyser (int argc, char * * argv, int * nb_iter, double * x_min, double * x_max, double * y_min, double * y_max, int * larg, int * haut) {

  const char * opt = "b:d:n:f:" ;
  int c ;
  /* Valeurs par defaut */
    
  * nb_iter = 100;
  * x_min = X_MIN;
  * x_max = X_MAX;
  * y_min = Y_MIN;
  * y_max = Y_MAX;
  * larg = 1024;
  * haut = 768;

  /* Analyse arguments */
  while ((c = getopt (argc, argv, opt)) != EOF) {
    
    switch (c) {
      
    case 'b':
      sscanf (optarg, "%lf", x_min);
      sscanf (argv [optind ++], "%lf", x_max);
      sscanf (argv [optind ++], "%lf", y_min);
      sscanf (argv [optind ++], "%lf", y_max);
      break ;
    case 'd': /* Largeur */
      sscanf (optarg, "%d", larg);
      sscanf (argv [optind ++], "%d", haut);
      break;
    case 'n': /* Nombre d'iterations */
      * nb_iter = atoi (optarg);
      break;
    default :
      erreur_options ();
    };
  }  
}


static void initialiser (Image * im, int nb_col, int nb_lig) {
  
  im -> nb_lig = nb_lig;
  im -> nb_col = nb_col;
  im -> pixels = (char *) malloc (sizeof (char) * nb_lig * nb_col); /* Allocation espace memoire */
} 

static void sauvegarder (const Image * im, const char * chemin) {
  
  /* Enregistrement de l'image au format ASCII '.PPM' */
  unsigned i;
  FILE * f = fopen (chemin, "w");  
  fprintf (f, "P6\n%d %d\n255\n", im -> nb_col, im -> nb_lig); 
  for (i = 0; i < im -> nb_col * im -> nb_lig; i ++) {
    char c = im -> pixels [i];
    fprintf (f, "%c%c%c", c, c, c); /* Monochrome blanc */
  }
  fclose (f);
}

static void calculer (Image * im, int nb_iter, double x_min, double x_max, double y_min, double y_max) {
  
  int pos = 0;

  int l, c, i = 0;
  
  double pasx = (x_max - x_min) / im -> nb_col, pasy = (y_max - y_min) / im -> nb_lig; /* Discretisation */

  for (l = 0; l < im -> nb_lig; l ++) {
    
    for (c = 0; c < im -> nb_col; c ++) {  

      /* Calcul en chaque point de l'image */

      double a = x_min + c * pasx, b = y_max - l * pasy, x = 0, y = 0;      
      i=0;
      while (i < nb_iter) {
	double tmp = x;
	x = x * x - y * y + a;
	y = 2 * tmp * y + b;
	if (x * x + y * y > 4) /* Divergence ! */
	  break; 
	else
	  i++;
      }
      
      im -> pixels [pos ++] = (double) i / nb_iter * 255;    
    }
  }
}



int main (int argc, char * * argv) {

  int nb_iter, larg, haut; /* Degre de nettete & dimensions de l'image */  
  double x_min, x_max, y_min, y_max; /* Bornes de la representation */
  Image im;
  int rank, P;
  int num_fichier, unite, dizaine;
  char chemin[100];
  double y_min_proc, y_max_proc, h;


  MPI_Init(NULL,NULL);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &P);


  num_fichier = (P-1) - rank;
  unite = num_fichier % 10;
  dizaine = num_fichier/10;

// Chaque processeur crée son fichier image
  sprintf(chemin, "/home/temperville/mandel%d%d.ppm", dizaine, unite);

// Analyse des paramètre
  analyser(argc, argv, & nb_iter, & x_min, & x_max, & y_min, & y_max, & larg, & haut);

// Paramètres des processeurs pour les calculs qui vont suivre
  haut = haut/P;
  h = (y_max-y_min)/P; // pas
  y_min_proc = y_min + rank*h;
  y_max_proc = y_min + (rank+1)*h;

// On initialise l'image avec sa largeur et sa hauteur (cette dernière étant modifiée lorsque l'on utilise plusieurs processeurs)
  initialiser (& im, larg, haut);

// On lance les calcule et on définit le contenu de l'image im de chaque processeur
  calculer (& im, nb_iter, x_min, x_max, y_min_proc, y_max_proc);

// On sauvegarde l'image de chaque processeur dans le fichier chemin qu'il a créé
  sauvegarder (& im, chemin);

  MPI_Finalize();

  return 0 ;
}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Fonction qui retourne le maximum de 3 paramètres
int maximum(int a, int b, int c)
{
  int k = a;

  if (b > k)
    k = b;
  if (c > k)
    k = c;

  return k;
}

// Procédure d'erreur de paramètres d'entrées
void erreur(int n)
{
  if (n < 3)
    {
      printf("Il faut 2 mots en paramètres pour faire fonctionner ce programme.\n");
      exit(1);
    }
}

// Procédure d'inversion de tableau
void reverse_array(int *T, int n)
{
  int i;
  int *T_temp;
  T_temp = (int*) malloc(n*sizeof(int));

  for (i=0; i<n; i++)
    T_temp[i] = T[n-1-i];
  for (i=0; i<n; i++)
    T[i] = T_temp[i];

  free(T_temp);
}


  /* ================================================
                          MAIN
     ================================================ */

int main(int argc, char* argv[])
{
  // Vérification de la présence de 2 mots
  erreur(argc);

  // Variables locales
  int i, j, compteur;
  int compteur1, compteur2;    // compteurs pour les mots mot1 et mot2
  int I, J;                    // taille de la matrice score
  int taille1, taille2;        // taille des 2 mots
  int match_subs, ins, del;    // variables temporaires
  int** score;                 // score
  const char* mot1 = argv[1];  // 1er mot
  const char* mot2 = argv[2];  // 2nd mot
  int* chemin;

  // Paramètres (qu'on peut éventuellement modifier)
  int cout_debut_indel = -6;   // début d'indel
  int cout_subs  = -4;         // substitution
  int cout_match = +5;         // correspondance

  // Longueur des mots étudiés
  taille1 = strlen(argv[1]);
  taille2 = strlen(argv[2]);

  // Dimensions de la matrice score
  I = taille1 + 1;
  J = taille2 + 1;

  // Allocation de la matrice des scores
  score = (int**) malloc( I * sizeof(int*) );
  for (i=0; i < I; i++)
    score[i] = (int*) malloc( J * sizeof(int) );

  // Allocation du tableau chemin
  chemin = (int*) calloc( (taille1 + taille2), sizeof(int) );


  /* ================================================
                    Mapping du score
     ================================================ */

  // Initialisation des bords
  score[0][0] = 0;
  for (i=1; i<I; i++)
    score[i][0] = i * cout_debut_indel;
  for (j=1; j<J; j++)
    score[0][j] = j * cout_debut_indel;

  // Calcul du score (mapping)
  for (j=1; j<J; j++)
    {
      for (i=1; i<I; i++)
        {
          if (mot1[i-1] == mot2[j-1])
            match_subs = score[i-1][j-1] + cout_match;  // match
          else
            match_subs = score[i-1][j-1] + cout_subs;   // substitution

          ins = score[i][j-1] + cout_debut_indel;   // insertion
          del = score[i-1][j] + cout_debut_indel;   // délétion
 
          score[i][j] = maximum(match_subs, ins, del);  // calcul du score en (i,j)
        }
    }


  /* ================================================
                       Backtracing
     ================================================ */

  typedef enum {NONE=0, INS, DEL, MATCH, MISMATCH} backtrace;

  // Boucle remontant le mapping
  i = taille1;
  j = taille2;
  compteur = 0;

  while ( (i > 0) || (j > 0) )
    {

      // Chemin diagonal (match ou substitution)
      if ( (i > 0) && (j > 0) )
        {
          if (mot1[i-1] == mot2[j-1])
            match_subs = cout_match;
          else
            match_subs = cout_subs;

          if ( (score[i-1][j-1] + cout_match == score[i][j]) && (match_subs == cout_match) )
            {
              chemin[compteur++] = MATCH;
              i--;
              j--;
              continue;
            }

          if ( (score[i-1][j-1] + cout_subs == score[i][j]) && (match_subs == cout_subs) )
            {
              chemin[compteur++] = MISMATCH;
              i--;
              j--;
              continue;
            }
        }

      // Chemin vertical (insertion)
      if (i > 0)
	{
	  if ( score[i-1][j] + cout_debut_indel == score[i][j] )
	    {
	      chemin[compteur++] = INS;
	      i--;
	      continue;
	    }
	}

      // Chemin horizontal (délétion)
      if (j > 0)
	{
	  if ( score[i][j-1] + cout_debut_indel == score[i][j] )
	    {
	      chemin[compteur++] = DEL;
	      j--;
	      continue;
	    }
	}

      // Message d'erreur si aucun 'if' n'a été pris
      printf("erreur, revoir le code !\n i = %d, j = %d, score = %d\n", i, j, score[i][j]);
      exit(1);
    }


  /* ================================================
          Affichage du score et du chemin
     ================================================ */

  // On met le tableau chemin dans l'autre sens
  reverse_array(chemin, compteur);

  // Affichage du score et du chemin
  printf("\nScore = %d\n", score[taille1][taille2]);
  printf("Le chemin est : ");
  for (i=0; i < compteur; i++)
    printf("%d ", chemin[i]);
  printf("\n\n");


  /* ================================================
         Affichage de la comparaison des 2 mots
     ================================================ */

  // Affichage du premier mot
  compteur1 = 0;
  compteur2 = 0;
  for (i=0; i < compteur; i++)
    {
      if (chemin[i] == DEL)
	printf("-");
      else
	printf("%c", mot1[compteur1++]);
    }
  printf("\n");

  // Affichage des correspondances
  for (i=0; i < compteur; i++)
    {
      if (chemin[i] == DEL)
        printf(" ");
      if (chemin[i] == INS)
        printf(" ");
      if (chemin[i] == MATCH)
        printf("|");
      if (chemin[i] == MISMATCH)
        printf("x");
    }
  printf("\n");

  // Affichage du deuxième mot
  compteur1 = 0;
  compteur2 = 0;
  for (i=0; i < compteur; i++)
    {
      if (chemin[i] == INS) // comme il s'agit du second mot, on inverse ce paramètre
	printf("-");
      else
	printf("%c", mot2[compteur2++]);
    }
  printf("\n\n");


  // Désallocations
  for (i=0; i<I; i++)
    free(score[i]);
  free(score);
  free(chemin);

}

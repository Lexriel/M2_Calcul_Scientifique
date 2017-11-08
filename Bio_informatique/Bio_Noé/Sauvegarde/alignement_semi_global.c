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
  int i, j, k, compteur, nb_indel;
  int compteur1, compteur2;    // compteurs pour les mots mot1 et mot2
  int I, J;                    // taille de la matrice score
  int taille1, taille2;        // taille des 2 mots
  int match_subs, ins, del;    // variables temporaires
  int** score;                 // score
  const char* mot1 = argv[1];  // 1er mot
  const char* mot2 = argv[2];  // 2nd mot
  int* chemin;
  int** M, ID;                 // matrices des modes "Match-mismatch" et "InDel"

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
    score[i][0] = 0;
  for (j=1; j<J; j++)
    score[0][j] = 0;

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


  // Définition des coordonnées de départ du backtracing
  int i_fin = taille1;
  int j_fin = taille2;
  int top_score = score[taille1][taille2];

  for(k=0; k < I; k++)
    if (score[k][taille2] > top_score)
      {
        top_score = score[k][taille2];
        i_fin = k;
        j_fin = taille2;
      }

  for(k=0; k < J; k++)
    if (score[taille1][k] > top_score)
      {
        top_score = score[taille1][k];
        i_fin = taille1;
        j_fin = k;
      }


  // Boucle remontant le mapping
  i = i_fin;
  j = j_fin;
  compteur = 0;

  while ( (i > 0) && (j > 0) )
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

  int i_debut = i;
  int j_debut = j;


  /* ================================================
          Affichage du score et du chemin
     ================================================ */

  // On met le tableau chemin dans l'autre sens
  reverse_array(chemin, compteur);

  // Affichage du score et du chemin
  printf("\nScore = %d\n", score[i_fin][j_fin]);
  printf("Le chemin est : ");
  for (k=0; k < compteur; k++)
    printf("%d ", chemin[k]);
  printf("\n\n");

  /* ================================================
         Affichage de la comparaison des 2 mots
     ================================================ */

  // Affichage du premier mot
  compteur1 = i_debut;
  nb_indel = 0;

  for (k=0; k < j_debut; k++)
    printf(" ");
  for (k=0; k < i_debut; k++)
    printf("%c", mot1[k]);

  for (k=0; k < compteur; k++)
    {
      if (chemin[k] == DEL)
        {
          printf("-");
          nb_indel++;
        }
      else
	printf("%c", mot1[compteur1++]);
    }

  for (k = i_debut + compteur - nb_indel; k < I; k++)
    printf("%c", mot1[k]); 
  printf("\n");

  // Affichage des correspondances
  for (k=0; k < i_debut + j_debut; k++)
    printf(" ");

  for (k=0; k < compteur; k++)
    {
      if (chemin[k] == DEL)
        printf(" ");
      if (chemin[k] == INS)
        printf(" ");
      if (chemin[k] == MATCH)
        printf("|");
      if (chemin[k] == MISMATCH)
        printf("x");
    }
  printf("\n");

  // Affichage du deuxième mot
  compteur2 = j_debut;
  nb_indel = 0;

  for (k=0; k < i_debut; k++)
    printf(" ");
  for (k=0; k < j_debut; k++)
    printf("%c", mot2[k]);

  for (k=0; k < compteur; k++)
    {
      if (chemin[k] == INS) // comme il s'agit du second mot, on inverse ce paramètre
        {
          printf("-");
          nb_indel++;
        }
      else
	printf("%c", mot2[compteur2++]);
    }

  for (k = j_debut + compteur - nb_indel; k < J; k++)
    printf("%c", mot2[k]);
  printf("\n");

  printf("\n\n");

  // Désallocations
  for (k=0; k<I; k++)
    free(score[k]);
  free(score);
  free(chemin);

}

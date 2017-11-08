# ifndef TP_CPU_H
# define TP_CPU_H


# include <stdlib.h>
# include <stdio.h>
# include <string.h>
# include <time.h>
# include <unistd.h>
# include <iostream>
# include <fstream>
using namespace std;


// Fonction puissance
unsigned long power(int n, int k)
{
  int i;
  unsigned long result = 1;

  for (i=0; i<k; i++)
    result = result*(unsigned long) n;

  return result;
}


// Fonction minimum
unsigned long minimum(unsigned long a, unsigned long b)
{
  if (a < b)
    return a;
  else
    return b;
}


// Fonction maximum
unsigned long maximum(unsigned long a, unsigned long b)
{
  if (a > b)
    return a;
  else
    return b;
}


// Fonction somme
unsigned long somme(unsigned long* T, unsigned long n)
{
  unsigned long i;
  unsigned long res = 0;
  for(i=0; i<n; i++)
    res += T[i];
  return res;
}


// Procédure de permutations
void permutations(unsigned long* T, int a, int b)
{
  unsigned long temp;
  temp = T[a];
  T[a] = T[b];
  T[b] = temp;
}


// Fonction code
unsigned long code(char *w, int k)
{
  unsigned long i;
  int wi = 0;
  int result = 0;
  int power_4_i = 1; // 4^i

  for (i=0; i<k; i++)
    {
      if (w[i] == 'A')
        wi = 0;
      if (w[i] == 'C')
        wi = 1;
      if (w[i] == 'T')
        wi = 2;
      if (w[i] == 'G')
        wi = 3;
      if (w[i] == 'N')
        wi = rand() % 4;

      result = result + wi * power_4_i;
      power_4_i = power_4_i * 4;
    }

  return result;
}


// Procédure qui décale un tableau 'mot' vers la gauche
void decalage(char *tab, char c, int k)
{
  int i;

  for (i=0; i<k-1; i++)
    tab[i] = tab[i+1];
  tab[k-1] = c;
}


// Procédure de stockage d'un fichier dans un tableau
void stocker(char* filename, unsigned long n, char* & a)
{
  ifstream data_file;
  unsigned long i;
  data_file.open(filename);

  if (! data_file.is_open())
    {
      printf("\n Error while reading the file %s. Please check if it exists !\n", filename);
      exit(1);
    }

  a = (char*) malloc (n*sizeof(char));

  for (i=0; i<n; i++)
    data_file >> a[i];

  data_file.close();
}


// Fonction calculant la taille d'un fichier
unsigned long taille_fichier(char* filename)
{
  unsigned long taille = 0;
  char c;
  FILE* fichier;
  fichier = fopen(filename, "r");
  c = fgetc(fichier);

  while (c != EOF)
    {
      if (c != '\n')
        taille++;
      c = fgetc(fichier);
    }

  fclose(fichier);
  return taille;
}


// Fonction qui compte le nombre de k-mers communs entre T1 et T2
unsigned long find_kmers(unsigned long* T1, unsigned long* T2, unsigned long n)
{
  unsigned long i;
  unsigned long result = 0;
  for (i=0; i<n; i++)
    result = result + min(T1[i], T2[i]);
  return result;
}


// Fonction pour passer du code à la chaîne de caractères
void traduction(unsigned long valeur_code, char* mot_code, int k)
{
  unsigned long i, reste;

  for (i=0; i<k; i++)
    {
      
      reste = valeur_code % 4;

      if (reste == 0)
        mot_code[i] = 'A';
      if (reste == 1)
        mot_code[i] = 'C';
      if (reste == 2)
        mot_code[i] = 'T';
      if (reste == 3)
        mot_code[i] = 'G';

      valeur_code = (valeur_code - reste)/4;
    }
}


// Procédure créant un tableau de 16 éléments de faible complexité par leur code que l'on enlèvera à l'affichage dans les10meilleurs
void faible_complex(unsigned long* T, int k)
{
    int i, j, m;
    unsigned long A = (power(4,k) - 1) / 3;
    unsigned long C = (power(16,k/2) - 1) / 15;
    unsigned long B;

    if ( k % 2 == 0)
      B = C;
    else
      B = (power(16,k/2+1) - 1) / 15;

    T[0] = 0;   // mot AA...A
    T[1] = A;   // mot CC...C
    T[2] = 2*A; // mot TT...T
    T[3] = 3*A; // mot GG...G
    m = 4;

    for(i=0; i<4; i++)        // mots ATAT..AT, GTGT..GT, CACA..CA
      {
        for(j=0; j<4; j++)
          {
            if (j != i)
              {
                T[m] = i*B + 4*j*C;
                m++;
              }
          }
      }   

}


// Procédure affichant les 10 k-mers les plus fréquents à l'aide d'index_kmers
void les10meilleurs(unsigned long* table_index, unsigned long nombre_index, int k, int top)
{
  int i, j, p;
  unsigned long top10[top];
  unsigned long pos10[top];
  char kmers[k+1];
  kmers[k] = '\0';
  int booleen = 1;
  unsigned long faib_comp[16];

  faible_complex(faib_comp, k);

  for (i=0; i<top; i++)
    {
      top10[i] = 0;
      pos10[i] = 0;
    }

  for (i=0; i<nombre_index; i++)
    {
      if (table_index[i] < top10[top-1])
        continue;

      else
        {
          // si c'est un mot de faible complexité, on ne fait rien
          if (k > 2)
            for(p=0; p<16; p++)
              if (i == faib_comp[p])
                booleen = 0;

          /* si on n'a rencontré aucun mot de faible complexité (booleen == 1)
             et que l'on a un kmer de meilleur valeur que ceux du top, on peut
             les décaler et insérer le nouveau meilleur kmer dans le top.      */
          if (booleen == 1)
            { 
              top10[top-1] = table_index[i];
              pos10[top-1] = i;

              for (j=1; j<top; j++)
                {
                  if (top10[top-1-j] >= top10[top-j])
                    break;
                  else
                    {
                      permutations(top10, top-1-j, top-j);
                      permutations(pos10, top-1-j, top-j);
                    }
                }
            }
          else
            booleen = 1; 
        }
    }

  for (i=0; i<minimum(9,top); i++)
    {
      traduction(pos10[i], kmers, k);
      printf("numéro %d  :    %s    (apparait %ld fois)\n", i+1, kmers, top10[i]);
    }
  for (i=9; i<top; i++)
    {
      traduction(pos10[i], kmers, k);
      printf("numéro %d :    %s    (apparait %ld fois)\n", i+1, kmers, top10[i]);
    }
}


// Procédure affichant les 10 k-mers les moins fréquents à l'aide d'index_kmers
void les10pires(unsigned long* table_index, unsigned long nombre_index, int k, int top, unsigned long size)
{
  int i, j, p;
  unsigned long top10[top];
  unsigned long pos10[top];
  char kmers[k+1];
  kmers[k] = '\0';
  int booleen = 1;
  unsigned long faib_comp[16];

  faible_complex(faib_comp, k);

  for (i=0; i<top; i++)
    {
        top10[i] = size+1;   // n'importe quel élément de table_index est plus petit que size+1, on remplira forcément top10
        pos10[i] = 0;
    }

  for (i=0; i<nombre_index; i++)
    {
      if (table_index[i] > top10[top-1])
        continue;

      else
        {
          // si c'est un mot de faible complexité, on ne fait rien
          if (k > 2)
            for(p=0; p<16; p++)
              if (i == faib_comp[p])
                booleen = 0;

          /* si on n'a rencontré aucun mot de faible complexité (booleen == 1)
             et que l'on a un kmer de meilleur valeur que ceux du top, on peut
             les décaler et insérer le nouveau meilleur kmer dans le top.      */
          if (booleen == 1)
            { 
              top10[top-1] = table_index[i];
              pos10[top-1] = i;

              for (j=1; j<top; j++)
                {
                  if (top10[top-1-j] <= top10[top-j])
                    break;
                  else
                    {
                      permutations(top10, top-1-j, top-j);
                      permutations(pos10, top-1-j, top-j);
                    }
                }
            }
          else
            booleen = 1; 
        }
    }

  for (i=0; i<minimum(9,top); i++)
    {
      traduction(pos10[i], kmers, k);
      printf("numéro %d  :    %s    (apparait %ld fois)\n", i+1, kmers, top10[i]);
    }
  for (i=9; i<top; i++)
    {
      traduction(pos10[i], kmers, k);
      printf("numéro %d :    %s    (apparait %ld fois)\n", i+1, kmers, top10[i]);
    }
}


// Procédure modifiant les N d'un tableau de caractères de façon aléatoire
void modif_tab(char* T, unsigned long n)
{
  int i, a;
  for(i=0; i<n; i++)
    {
      if (T[i] == 'N')
        {
          a = rand() % 4;
          if (a == 0)
            T[i] = 'A';
          else if (a == 1)
            T[i] = 'C';
          else if (a == 2)
            T[i] = 'T';
          else
            T[i] = 'G';
        }
    }
}
# endif /* TP_CPU_H */

# ifndef FONCTIONS_H
# define FONCTIONS_H


// Fonction puissance
int power(int n, int k)
{
  int i;
  int result = 1;

  for (i=0; i<k; i++)
    result = result*n;

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


// Fonction somme des éléments d'un tableau
unsigned long sum(unsigned long *T, unsigned n)
{
  unsigned long i;
  unsigned long s = 0;

  for (i=0; i<n; i++)
    s = s + T[i];
  return s;
}


// Fonction aléatoire
int random_number(int max)
{
  return rand() % max;
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
int code(char *w, int k)
{
  int i;
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
        wi = random_number(4);

      result = result + wi * power_4_i;
      power_4_i = power_4_i * 4;
    }

  return result;
}


// Procédure d'affichage des éléments au dessus de la moyenne d'un tableau
void display_array(unsigned long* T, unsigned long n)
{
  unsigned long i;
  unsigned long moy = 0;

  for (i=0; i<n; i++)
    moy = moy + T[i];
  moy = moy/n;

  for (i=0; i<n; i++)
    if (T[i] > maximum(moy, 5))
      printf("index_kmers[%ld] = %ld\n", i, T[i]);
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
unsigned long find_kmers_CPU(unsigned long* T1, unsigned long* T2, unsigned long n)
{
  unsigned long i;
  unsigned long result = 0;
  for (i=0; i<n; i++)
    result = result + min(T1[i], T2[i]);
  return result;
}


// Fonction pour passer du code à la chaîne de caractères
void traduction(int valeur_code, char* mot_code, int k)
{
  int i, reste;

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


// Procédure affichant les 10 k-mers les plus fréquents à l'aide d'index_kmers
void les10meilleurs(unsigned long* table_index, unsigned long nombre_index, int k)
{
  unsigned long i, j;
  unsigned long top10[10];
  unsigned long pos10[10];
  char kmers[k+1];
  kmers[k] = '\0';

  for (i=0; i<10; i++)
    {
      top10[i] = 0;
      pos10[i] = 0;
    }

  for (i=0; i<nombre_index; i++)
    {
      if (table_index[i] < top10[9])
        continue;

      else
        {
          top10[9] = table_index[i];
          pos10[9] = i;

          for (j=1; j<10; j++)
            {
              if (top10[9-j] >= top10[10-j])
                break;
              else
                {
                  permutations(top10, 9-j, 10-j);
                  permutations(pos10, 9-j, 10-j);
                }
            }
        }
    }

  printf("Les 10 %d-mers les plus fréquents sont :\n", k);
  for (i=0; i<9; i++)
    {
      traduction(pos10[i], kmers, k);
      printf("numéro %ld  :    %s    (apparait %ld fois)\n", i+1, kmers, top10[i]);
    }
  traduction(pos10[9], kmers, k);
  printf("numéro %d :    %s    (apparait %ld fois)\n", 10, kmers, top10[9]);
}



# endif /* FONCTIONS_H */

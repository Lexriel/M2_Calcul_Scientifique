/* ========================================================

                Fonctions propres au CPU 

   ======================================================== */

// Procédure d'affichage de tableau :
void display_tab(int* T, int n)
{
  int i;
  for (i=0; i<n; i++)
    printf("%d ", T[i]);
  printf("]\n");
}


// Fonction de test d'égalité de 2 tableaux de taille n :
int is_that_equal(int *T1, int *T2, int n)
{
  int i;
  for (i=0; i<n; i++)
    if ( abs(T1[i] - T2[i]) > 0.0001f )
      {
        printf("A = A_aux is false.\n");
        printf("There is an error on the element i=%d.\n", i);
        return 0;
      }

  printf("A = A_aux is true.\n");
  return 0;
}

// Procédure d'initialisation de tableau :
void init_tab(int *T, int n)
{
  int i;
  for (i=0; i<n; i++)
    T[i] = i;

}

// Procédure d'inversion de tableau :
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



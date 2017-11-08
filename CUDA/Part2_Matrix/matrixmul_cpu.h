/* ========================================================

                Fonctions propres au CPU 

   ======================================================== */


// Effectue le produit matriciel de A par B pour obtenir C :
void matmul(float *A, float *B, float *C)
{
  int i, j, k, l;

  // C(i][j] = sum( A[i][k]*B[k][j] , k=0..WC-1 );
  // l = i*WC + j;
  // C[l] = sum( A[i*WC+k]*B[k*WC+j] , k=0..WC-1 );

  for (i=0; i<HC; i++)
    {
      for (j=0; j<WC; j++)
        {
          l = i*WC + j;
          C[l] = 0;
          for (k=0; k<WC; k++)
            C[l] = C[l] + A[i*WC+k]*B[k*WC+j];
        }
    }
}


// Affiche un tableau :
void display_tab(float* T, int n)
{
  int i;
  for (i=0; i<n; i++)
    printf("%f ", T[i]);
  printf("]\n");
}


// Teste l'égalité de 2 tableaux de taille n :
int is_that_equal(float *T1, float *T2, int n)
{
  int i;
  for (i=0; i<n; i++)
    if ( abs(T1[i] - T2[i]) > 0.0001f )
      {
        printf("C = C_aux is false.\n");
        printf("There is an error on the element i=%d.\n", i);
        return 0;
      }

  printf("C = C_aux is true.\n");
  return 0;
}

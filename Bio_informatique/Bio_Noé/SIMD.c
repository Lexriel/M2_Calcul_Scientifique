#include <stdio.h>
#include <stdlib.h>
#include <xmmintrin.h>


// Procédure d'erreur de paramètres d'entrées
void erreur(int n)
{
  if (n < 2)
    {
      printf("Ce programme calcule la somme de 2 vecteurs à la con, puis effectue l'algorithme des babyloniens pour déterminer la racine carrée de certains nombres, le nombre d'itération 'n' doit être entré en paramètre.\n");
      printf("Il faut un paramètre (n) pour faire fonctionner ce programme.\n");
      exit(1);
    }
}

// union pour "conversion facile"
typedef union
  {
    // vecteur 128 bits SSE : un seul entier
    __m128 m128;
    // vecteur 128 bits SSE : 4 flottants
    __v4sf v4sf;
    // vecteur 128 bits SSE : 4 flottants (tableau)
    float v4sf_tab[4];
    // vecteur 128 bits SSE : 4 flottants (structure)
    struct
      {
        float v4sf1, v4sf2, v4sf3, v4sf4;
      };
  } __attribute__ ((aligned (16))) vector128;



int main(int argc, char* argv[])
{
  erreur(argc);

  // a) un test set, add, print
  __v4sf v1 = _mm_set_ps(1.0, 2.0, 3.0, 4.0);
  __v4sf v2 = _mm_set_ps(5.0, 6.0, 7.0, 8.0);
  __v4sf v3 = _mm_add_ps(v1, v2);

  printf("{%f, %f, %f, %f}\n",
    ((vector128)v3).v4sf_tab[0],
    ((vector128)v3).v4sf_tab[1],
    ((vector128)v3).v4sf_tab[2],
    ((vector128)v3).v4sf_tab[3]
    );
  printf("Ces données sont calculées à l'envers à cause de l'endian inversé.\n");

  // b) algorithme des babyloniens pour calculer la racine carrée d'un entier S
  __v4sf x0 = _mm_set_ps(16.0, 30.0 , 50.0, 100.0);
  __v4sf xn = _mm_set_ps(16.0, 30.0, 50.0, 100.0);
  __v4sf a;
  __v4sf demi = _mm_set_ps(0.5, 0.5, 0.5, 0.5);
  int i, n;


  n = atoi(argv[1]);
  for (i=0; i<n; i++)
    {
      a = _mm_rcp_ps(xn);        // 1/xn
      a = _mm_mul_ps(x0, a);     // x0/xn
      a = _mm_add_ps(xn, a);     // xn + x0/xn
      a = _mm_mul_ps(demi, a);   // 1/2*(xn + x0/xn)
      xn = a;
    }

  printf("Les racines carrées de {%f, %f, %f, %f} après %d itérations sont :\n{%f, %f, %f, %f}.\n",
    ((vector128)x0).v4sf_tab[0],
    ((vector128)x0).v4sf_tab[1],
    ((vector128)x0).v4sf_tab[2],
    ((vector128)x0).v4sf_tab[3],
    n,
    ((vector128)xn).v4sf_tab[0],
    ((vector128)xn).v4sf_tab[1],
    ((vector128)xn).v4sf_tab[2],
    ((vector128)xn).v4sf_tab[3]
    );

  return 0;
}

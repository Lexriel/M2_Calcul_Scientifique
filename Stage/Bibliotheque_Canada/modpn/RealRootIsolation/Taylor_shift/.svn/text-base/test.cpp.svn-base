// creates Polynomial_shift(x) = Polynomial(x+1)
void create_polynomial_shift_CPU(int *Polynomial, int *T1, int *T2, int *Monomial_shift, int n, int p, int param, int position)
{
  int i, s;
  int n2 = n;
  int *Temp1, *Temp2, *res;
  int *Calc;
  int *T1bis, *T2bis;

  // recursive call to transform terms in (x+1)^i of Polynomial(x+1)
  if (n2 != 1)
  {
    n2 /= 2;
    create_polynomial_shift_CPU(Polynomial, T1, T2, Monomial_shift, n2, p, 0, position);
    create_polynomial_shift_CPU(Polynomial, T1, T2, Monomial_shift, n2, p, n2, position+n2);
  }

  if (n != 1)
    {
      if (param == 0)
      {
//        for (i=n/2; i<n; i++)
//          T1[i] = 0;
Calc = (int*) calloc(n, sizeof(int));
for (i=0; i<n/2; i++)
  Calc[i] = T1[i+position];
        add_arrays(T1 + position, Calc, T2 + position, n);
free(Calc);
        // nothing to modify in T1
      }
      else // (param != 0)
      {
        s = (int) pow(2, param-1);
        Temp1 = (int*) calloc(2*n, sizeof(int));
        Temp2 = (int*) calloc(2*n, sizeof(int));
        res   = (int*) calloc(2*n, sizeof(int));
        /*for (i=0; i<n/2; i++)
          Temp1[i] = T2[i+position];*/
        add_arrays(T2 + position, T1 + position, T2 + position, n);
        memcpy(Temp1, T2 + position, n * sizeof(int));
        /*for (i=0; i<n/2+1; i++)
          Temp2[i] = Monomial_shift[s+i];*/
        memcpy(Temp2, Monomial_shift + s, (n+1) * sizeof(int));

        conv_prod(res, Temp1, Temp2, 2*n, p);
        memcpy(T2, res, 2*n* sizeof(int));

printf("conv : \n");
display_array(res, 8);
printf("Temp1 = Pol_shift2 : \n");
display_array(Temp1, 8);
printf("Temp2 = Monomial : \n");
display_array(Temp2, 8);
        free(Temp1);
        free(Temp2);
        free(res);
      }
    }

  else // (n==1)
  {
    if (param == 0)
      T1[position] = Polynomial[position];
    else // (param != 0)
    {
      T2[position-1] = Polynomial[position];
      T2[position]   = Polynomial[position];
    }
  }

  // addition
  if (n != 1) 
  {
    if (param == 0)
    {
//      add_arrays(T1 + position, T1 + position, T2 + position, n);
//      memcpy(T2 + position, T1 + position, n*sizeof(int));
    }
    else // (param != 0)
    {
//      add_arrays(T2 + position, T1 + position, T2 + position, n);
//      memcpy(T1 + position, T2 + position, n*sizeof(int));
    }
  }
printf("n = % d || param = %d || position = %d \n", n, param, position);
printf("T 1 and 2 :\n");
display_array(T1, 8);
display_array(T2, 8);
printf("\n");
}


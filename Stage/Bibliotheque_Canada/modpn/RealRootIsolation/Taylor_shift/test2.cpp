// creates Polynomial_shift(x) = Polynomial(x+1)
void create_polynomial_shift_CPU(int *Polynomial, int *T1, int *T2, int *Monomial_shift, int n, int p, int local_n)
{
  int i, j;
  int m;
  int *Temp1, *Temp2, *Temp3, *res;

  while (local_n != 1) // 
  {
    create_polynomial_shift_CPU(Polynomial, T1, T2, Monomial_shift, n, p, n/2);
  }


  if (local_n != 1)
  {
    Temp1 = (int*) calloc(2*local_n, sizeof(int));
    Temp2 = (int*) calloc(2*local_n, sizeof(int));
    Temp3 = (int*) calloc(2*local_n, sizeof(int));
    res   = (int*) calloc(2*local_n, sizeof(int));

    memcpy(Temp3, Monomial_shift + local_n, (local_n + 1) * sizeof(int));

    for (j=0; j<n; j+=2*local_n)
    {
      memcpy(Temp1, T1 + j, local_n*sizeof(int));
      memcpy(Temp2, T1 + local_n + j, n*sizeof(int));
      conv_prod(res, Temp3, Temp2, 2*local_n, p);
      add_arrays(res, res, Temp1);
      memcpy(T1+j, res, 2*local_n*sizeof(int));
    }

    free(Temp1);
    free(Temp2);
    free(Temp3);
    free(res);
  }

  else // (local_n == 1)
  {
    for (i=0; i<n; i+=2)
    {
      T1[i] = Polynomial[i] + Polynomial[i+1];
      T1[i+1] = Polynomial[i+1];
    }
  }

printf("\nlocal_n = %d\n", local_n);
printf("T1 : \n");
display_array(T1, n);

}

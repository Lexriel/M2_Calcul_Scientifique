/* test_cpu.cu */


// affiche tableau
void display_tab(int* T, int size)
{
  int i;

  for (i=0; i<size; i++)
    printf("%d ", T[i]);
}

//on oublie
void procedure_cpu(int size){
  int i;
  int *T, *T2;

  

  T  = (int*) malloc(size*sizeof(int));
  T2 = (int*) malloc(size*sizeof(int));

  for (i=0; i<size; i++)
    T[i] = i;

  printf("T = [ ");
  display_tab(T, size);
  printf("]\n");

  memcpy(T2, T, size*sizeof(int));

  for (i=0; i<size; i++)
    T2[i]++;
  

  printf("T2 = [ ");
  display_tab(T2, size);
  printf("]\n");

  free(T);
  free(T2);
}


// inc_cpu
void inc_cpu(int *a, int n)
{
  int i;
  for (i=0; i<n; i++)
    a[i]++;
}

//on cree une nouvelle procedure

void procedure_cpu2(int size){
 int i;
 int *T;

  T  = (int*) malloc(size*sizeof(int));
   for (i=0; i<size; i++)
    T[i] = i;

  printf("T = [ ");
  display_tab(T, size);
  printf("]\n");

  inc_cpu(T,size);

  printf("T = [ ");
  display_tab(T, size);
  printf("]\n");

  free(T);
  


}



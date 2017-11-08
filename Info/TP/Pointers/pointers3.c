# include <stdlib.h>
# include <stdio.h>



void cross_product(int *A, int *B, int *C)
{

  *A   = *(B+1) * *(C+2) - *(B+2) * *(C+1);
  *(A+1)   = *(B+2) * *C - *B * *(C+2);
  *(A+2)   = *B * *(C+1) - *(B+1) * *C;

}


/*
==================================TEST==========================================
int main()
{
  int A[]={0,0,0};
  int B[]={1,2,3};
  int C[]={2,1,1};
  cross_product(A,B,C);
  printf("%d %d %d\n", *A, *(A+1), *(A+2));
  cross_product(A,A,B);
  printf("%d %d %d\n", *A, *(A+1), *(A+2));
  return 0;
}
================================================================================
*/

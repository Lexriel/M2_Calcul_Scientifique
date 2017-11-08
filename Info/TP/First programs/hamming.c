# include <stdio.h>
# include <math.h>

main()
{
  int n, ones ;
  ones=0 ;
  printf("Give a number.\n");
  scanf("%d",&n) ;

    /*tant que n n'est pas nul, si le reste de la division euclidienne de n par 2 est nulle, alors on ajoute un à la variable ones, puis on affecte n/2 à n. */

    while (n!=0)
{
   if (n%2==1)
     ones++ ;

   n=n/2 ;
}

    printf("The Hamming weight of the number %d is %d\n\n", n, ones) ;

}

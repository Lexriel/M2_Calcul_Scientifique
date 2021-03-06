PROGRAM beuler_program

  USE beuler_module
  USE fonction_module

  IMPLICIT NONE

  REAL, DIMENSION(1:2) :: tspan
  REAL :: y0, c
  INTEGER :: Nh

  
  print*, "Donner l'intervalle tspan en rentrant successivement ces 2 coordonnées: "
  read*, tspan(1)
  read*, tspan(2)
  print*, "Donner la valeur initiale y0: "
  read*, y0
  print*, "Donner le nombre d'étapes souhaité Nh: "
  read*, Nh

  
  c = beuler(f, fprime, tspan, y0, Nh)


  print*, "   "
  print*, "Une valeur approchée de la solution à l'aide de la méthode 'Backward Euler' en", tspan(2), "est donnée par", c, "."
  print*, "Une approximation de la solution est donnee par les données rentrées dans le fichier 'beuler.dat'."
  print*, "Pour visualiser le graphique de cette approximation de la solution, utiliser plot(t,u) avec Matlab."
  print*, "   "
  
  
END PROGRAM beuler_program

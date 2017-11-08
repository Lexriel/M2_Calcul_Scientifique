PROGRAM beulerN_program

  USE broyden_module
  USE beulerN_module

  IMPLICIT NONE

  REAL, DIMENSION(1:2) :: tspan
  REAL, DIMENSION(3) :: y0
  INTEGER :: Nh

  
  print*, "Donner l'intervalle tspan en rentrant successivement ces 2 coordonnées: "
  read*, tspan(1)
  read*, tspan(2)
  y0=(/0.1,0.1,-0.1/)
  print*, "Donner le nombre d'étapes souhaité Nh: "
  read*, Nh

  
  call beulerN(odefun, tspan, y0, Nh)


  print*, "   "
  print*, "Une approximation de la solution est donnée par les valeurs (t,u) rentrées dans le fichier 'beulerN_data.dat'."
  print*, "   "
  
  
END PROGRAM beulerN_program

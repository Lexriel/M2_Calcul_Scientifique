PROGRAM newtonN_program

  USE newtonN_module
  USE fonction3_module

  IMPLICIT NONE

  REAL, DIMENSION(:), INTENT(IN) :: y0, c 

  
  print*, "Donner la valeur initiale y0: "
  read*, y0
! y0=(/0.1,0.1,-0.1/)

  
  c = newtonN(F, JacF, y0)


  print*, "   "
  print*, "Une valeur approchée de la solution à l'aide de la méthode de Newton en N dimensions en", tspan(2), "est donnée par", c, "."
  print*, "   "
  
  
END PROGRAM newtonN_program

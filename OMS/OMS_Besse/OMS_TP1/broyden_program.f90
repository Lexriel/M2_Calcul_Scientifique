PROGRAM broyden_program

  USE broyden_module
  USE fonction3_module

  IMPLICIT NONE

  REAL, DIMENSION(3) :: y0, c
  REAL :: h

  
!  print*, "Donner les coordonnées de la valeur initiale y0: "
!  read*, y0(1)
!  read*, y0(2)
!  read*, y0(3)
y0=(/0.1,0.1,-0.1/)


  print*, "Donner le pas h: "
  read*, h
  
  c = broyden(F, y0, h)


  print*, "   "
  print*, "Une valeur approchée de la solution à l'aide de la méthode de Broyden est donnée par"
  print*, c
  print*, "   "
  
  
END PROGRAM broyden_program

program exo1_program

  use exo1_module
  use lapack95
  use blas95

  implicit none

  INTEGER :: N

  print*, " "
  print*, "Ce programme détermine une approximation de l'équation de la chaleur de l'exercice 1,"
  print*, "pour un découpage de l'ensemble [-3;3]² sur maillage de N² points."
  print*, "Le résultat sera donné dans le fichier 'heat_equa1.dat'."
  print*, "Pour afficher le résultat, tracer un graphique à l'aide de matlab."
  print*, "Donner la valeur de N souhaitée : "
  read*, N

    call exo1(N)

  print*, " "

end program exo1_program

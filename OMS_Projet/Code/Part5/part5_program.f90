program part5_program

  use part5_module
  use lapack95
  use blas95

  implicit none

  call Solve_Laplacian(5, 4, rho)

  ! This subroutine creates the matrix A
  ! and solves A.u = rho .

end program part5_program

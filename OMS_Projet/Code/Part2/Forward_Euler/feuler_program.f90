program feuler_program

  use feuler_module
  use init_data_module
  
  implicit none
  
  double precision, dimension(4) :: A0, AS
  double precision, dimension(2) :: zspan
  double precision :: S = 1., eps = 10.**(-16.)
  zspan(1) = 0.
  zspan(2) = S
  
! test for  A0 = (/0.1,-0.5, 0.3, 1./) at the beginning, now we use a vector A0 found running this program :
 A0 =(/ 2.102668458493112*10.**(-2.),  3.251853964640191*10.**(-3.),  2.104295781272272*10.**(-2.), -3.252690878974059*10.**(-3.) /)

        AS = feuler(zspan, A0, 200)

	do while ( (abs(A0(1)-AS(1)) > eps).or.(abs(A0(2)-AS(2)) > eps).or.(abs(A0(3)-AS(3)) > eps).or.(abs(A0(4)-AS(4)) > eps) )
 
        AS = feuler(zspan, A0, 200)
	A0 = (A0+AS)/2.

        end do

open(unit=105, file='A0.dat', status='unknown')
write(105,*) A0(1)
write(105,*) A0(2)
write(105,*) A0(3)
write(105,*) A0(4)

print*, "A0 =", A0(1), A0(2), A0(3), A0(4)

call init_data(A0)
  
end program feuler_program

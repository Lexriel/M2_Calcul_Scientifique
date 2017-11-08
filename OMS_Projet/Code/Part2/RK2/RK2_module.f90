module RK2_module

implicit none

contains


function RK2(zspan,A0,Nh)

    ! Declarations
    double precision, dimension(4), intent(in) :: A0
    double precision, dimension(4) :: RK2
    integer,intent(in) :: Nh
    integer :: i
    double precision :: h, y
    double precision, dimension(4) :: U
    double precision, intent(in), dimension(2) :: zspan
    
    double precision :: eps_x, eps_y, K, pi, eps0, m, v_z, c, gamma_z, N0, q, Intensity, E_k
    
    ! Constants test 1
    eps_x = 2.36*10.**(-5.)
    eps_y = eps_x 
    pi = 4.*atan(1.)
    m = 1.672623*10.**(-27.)
    E_k = 5.*10.**6.*1.60217653*10.**(-19.)
    c   = 299792458.
    gamma_z = E_k/(m*c**2.) + 1.   ! as E_k = m*c**2.*(gamma_z-1)
    v_z = c*sqrt(1.-1./gamma_z**2.)
    q = 1.60217653*10.**(-19.)
    eps0 = 8.85418782*10.**(-12.)
    Intensity = 0.1
    N0 = Intensity/(q*v_z)
    K = q**2.*N0/( 2.*pi*eps0*gamma_z**3.*m*v_z**2. )

    y = zspan(1)
    U = A0
    h = (zspan(2)-zspan(1))/Nh

open(unit=101, file='a.dat', status='unknown')
open(unit=102, file='da.dat', status='unknown')
open(unit=103, file='b.dat', status='unknown')
open(unit=104, file='db.dat', status='unknown')

write(101,*) y, U(1)
write(102,*) y, U(2)
write(103,*) y, U(3)
write(104,*) y, U(4)


do i=1,Nh

	U = U + h*phi(y+h/2,U+h/2*phi(y,U))
	y = y + h

write(101,*) y, U(1)
write(102,*) y, U(2)
write(103,*) y, U(3)
write(104,*) y, U(4)

end do

  close(101)
  close(102)
  close(103)
  close(104)

RK2 = U

!!!!!!!!!!!!
  contains
    
!    FUNCTION Bprime(z) ! FODO I
 !     double precision, intent(in) :: z
  !    double precision :: eta=0.5, S=1., Bp=1., Bprime ! eta belongs to ]0;1[
   !   Bprime = Bp*( ((z>0.).and.(z<eta*S/4.)) .or. ((z>(1.-eta/4.)*S).and.(z<S)) ) - Bp*( (z>(1.-eta/2.)*S/2.).and.(z<(1.+eta/2.)*S/2.) )
    !END FUNCTION Bprime
    
  
    FUNCTION Bprime(z) ! FODO II
      double precision, intent(in) :: z
      double precision :: eta=0.5, S=1., Bp=0.4, Bprime ! eta belongs to ]0;1[
      Bprime = Bp*( ((z>(1.-eta)*S/4.).and.(z<(1.+eta)*S/4.)) ) - Bp*( ((z>(3.-eta)*S/4.).and.(z<(3.+eta)*S/4.)) )
    END FUNCTION Bprime
    
    
    FUNCTION K_x(z)     ! K_y = -K_x
      double precision, intent(in) :: z
      double precision :: K_x
      K_x = -q*Bprime(z)/(gamma_z*m*v_z)
    END FUNCTION K_x
    
    
    FUNCTION phi(z,A)
      double precision, intent(in) :: z
      double precision, dimension(4),intent(in) :: A
      double precision, dimension(4) :: phi
      phi = (/A(2), -K_x(z)*A(1) + 2.*K/(A(1)+A(3)) + eps_x**2./A(1)**3., A(4), K_x(z)*A(3) + 2.*K/(A(1)+A(3)) + eps_y**2./A(3)**3./)
    END FUNCTION phi


!!!!!!!!!!!!

END function RK2

end module RK2_module

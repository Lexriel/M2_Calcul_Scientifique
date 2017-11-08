program test
  use methode
  use blas95
  
  implicit none
  real (kind=8), dimension(:,:), allocatable :: Adiag,tmp
  real (kind=8), dimension(:), allocatable :: e,x,trvec
  integer, dimension(:), allocatable :: Diag,ibilut,jbilut
  integer :: N,N2, ndiag, j,k
  real (kind=8) :: dvar,tol
  real (kind=8), dimension(:), allocatable :: Acsr,b,bilut
  integer, dimension(:), allocatable :: ja,ia
  integer :: info,RCI_REQUEST,i,itercount,maxfil
  integer, dimension(128) :: IPAR
  real (kind=8), dimension(128) :: DPAR


  N2=4; N=8
  ndiag=5
  
  !---------------------------------------------------------------------------
  ! Allocation d'une matrice par stockage crux diagonal 
  ! (voir page 3360 de la documentation MKL)
  !---------------------------------------------------------------------------
  allocate(Adiag(N,ndiag), Diag(ndiag))
  allocate(e(N))
  e(:)=1.d0
  Adiag(:,1)=-e
  Adiag(:,2)=-e
  Adiag(:,3)=4.d0*e
  Adiag(:,4)=-e
  Adiag(:,5)=-e
  Adiag(N2+1,2)=0.d0; Adiag(N2,4)=0.d0
  diag=(/-N2,-1,0,1,N2/)


  !---------------------------------------------------------------------------
  ! Convert matrix from diagonal to csr storage
  !---------------------------------------------------------------------------
  allocate(acsr(N*ndiag),ja(N*ndiag),ia(N+1))
  call diacsr (n,0,ndiag,Adiag,ndiag,diag,Acsr,ja,ia)

  !---------------------------------------------------------------------------
  ! Affichage de la matrice pleine
  !---------------------------------------------------------------------------
  write(6,*) "Matrice A"
  call prints(n,N*ndiag,ACSR,IA,JA)
  write(6,*)


  !---------------------------------------------------------------------------
  ! Compute the ILUT preconditionner
  !---------------------------------------------------------------------------
  TOL=1.d-2
  dpar(31)=1.d0
  maxfil=5 ! Demi-largeur de bande

  allocate(BILUT((2*maxfil+1)*n-maxfil*(maxfil+1)+1),ibilut(n+1),jbilut((2*maxfil+1)*n-maxfil*(maxfil+1)+1),trvec(N))
  CALL DCSRILUT(N, ACSR, IA, JA, BILUT, IBILUT, JBILUT, TOL, 5, IPAR, DPAR, Info)

  write(6,*) "Factorisation LU incomplete de  A"
  call prints(n,(2*maxfil+1)*n-maxfil*(maxfil+1)+1,BILUT,IBILUT,JBILUT)
  write(6,*)
  write(6,*) '-----------------------------'

  
  !---------------------------------------------------------------------------
  !---------------------------------------------------------------------------
  ! On souhaite resoudre le systeme Ax=b
  !---------------------------------------------------------------------------
  !---------------------------------------------------------------------------
  allocate(x(N),b(N))
  b=1.d0
  x=0.d0


  !========--------========--------========--------========--------========--------========--------
  ! METHODE DE GRADIENT CONJUGUE : Utilisation de la fonction DCG de MKL
  !========--------========--------========--------========--------========--------========--------


  !---------------------------------------------------------------------------
  ! Initialisation des paramètres IPAR, DPAR suivant la solution initiale
  ! x et le second membre b
  !---------------------------------------------------------------------------
  allocate(TMP(N,4))
  CALL dcg_init(N,x,b,RCI_request,ipar,dpar,TMP)
  if (RCI_request.ne.0) then
     STOP 'Erreur d''initialisation'
  end if
     
  !---------------------------------------------------------------------------
  ! Positionnement de certains parametres par le programmeur (pages 2553-2557)
  !---------------------------------------------------------------------------
  ipar(9)=1   ! le test d'arret est controle par le programme
  ipar(10)=0  ! le test d'arret est controle par le programme
  !ipar(11)=1 ! Si ce parametre est different de zero, on utilise la methode
              ! de gradient conjugue preconditionne

  !---------------------------------------------------------------------------
  ! Verifie la coherence et la consistence des nouveaux parametres
  !---------------------------------------------------------------------------
  CALL dcg_check(N,x,b,RCI_request,ipar,dpar,TMP)
  if (RCI_request.ne.0) then
     STOP 'Erreur avec les nouveaux parametres'
  end if

  !---------------------------------------------------------------------------
  ! Compute the solution by RCI (P)CG solver 
  ! Reverse Communications starts here
  !---------------------------------------------------------------------------
  RCI_REQUEST=1
  do while (rci_request.ne.0)
     CALL dcg(N,x,e,RCI_request,ipar,dpar,TMP)

     if (rci_request.eq.1) then
        ! Il est necessaire de refaire une etape de gradient conjugue
        ! soit une multiplication matrice vecteur (voir p 2553)
        ! TMP(:,2)=A*TMP(:,1)
        CALL MKL_DCSRSYMV('U', N, ACSR, IA, JA, TMP(1:N,1), TMP(1:N,2))
     else if (rci_request.eq.3) then
        ! Il est necessaire de preconditionner le systeme (voir p 2554)
        call mkl_dcsrtrsv('L','N','U', n, bilut, ibilut, jbilut, tmp(1:N,3),trvec)
        call mkl_dcsrtrsv('U','N','N', n, bilut, ibilut, jbilut, trvec,tmp(1:N,4))
     end if
  end do

  !---------------------------------------------------------------------------
  ! Reverse Communication ends here
  ! Get the current iteration number
  !---------------------------------------------------------------------------
  CALL dcg_get(N,x,b,RCI_request,ipar,dpar,TMP,itercount)

  !---------------------------------------------------------------------------
  ! Print solution vector: x and number of iterations: itercount
  !---------------------------------------------------------------------------
  WRITE(*, *) ' The system has been solved '
  WRITE(*, *) ' The following solution obtained '
  WRITE(*,800) (x(i),i =1,N)
800 FORMAT(8(F10.3))
  WRITE(*,900)(itercount)
900 FORMAT(' Number of iterations: ',1(I2))

  open(150,file='toto.dat',status='unknown')
  do i=1,N
     write(150,*) x(i)
  end do
  close(150)


  !========--------========--------========--------========--------========--------========--------
  ! METHODE DE GRADIENT CONJUGUE : Mise en oeuvre par C. Besse
  !========--------========--------========--------========--------========--------========--------

  ! Gradient conjugue 
  x=cbcg(ACSR,IA,JA,b)
  WRITE(*, *) ' Solution obtenue'
  WRITE(*,800) (x(i),i =1,N)
  ! Gradient conjugue avec precision 1e-6 et 50 iter max
  ! Utilisation de la variable de DEBUG pour voir l'evolution de l'erreur
  x=cbcg(ACSR,IA,JA,b,1.d-6,50,DEBUG=1)
  WRITE(*, *) ' Solution obtenue'
  WRITE(*,800) (x(i),i =1,N)
  ! Gradient conjugue preconditionne avec precision 1e-6 et 50 iter max
  x=cbcg(ACSR,IA,JA,b,1.d-6,50,BILUT,ibilut,jbilut,1)
  WRITE(*, *) ' Solution obtenue'
  WRITE(*,800) (x(i),i =1,N)

  !========--------========--------========--------========--------========--------========--------
  ! METHODE DE BIGRADIENT CONJUGUE : Mise en oeuvre par C. Besse
  !========--------========--------========--------========--------========--------========--------
  ! Les options sont similaires a celle du gradient conjugue

  ! Bigradient conjugue 
  x=bcgr(ACSR,IA,JA,b)
  WRITE(*, *) ' Solution obtenue par bigradient conjugue'
  WRITE(*,800) (x(i),i =1,N)

  !========--------========--------========--------========--------========--------========--------
  ! METHODE DE CONJUGATE GRADIENT SQUARED : Mise en oeuvre par C. Besse
  !========--------========--------========--------========--------========--------========--------
  ! Les options sont similaires a celle du gradient conjugue

  ! Conjugate gradient squared
  x=cgs(ACSR,IA,JA,b)
  WRITE(*, *) ' Solution obtenue par Conjugate Gradient Squared'
  WRITE(*,800) (x(i),i =1,N)


  !========--------========--------========--------========--------========--------========--------
  ! METHODE DE BIGRADIENT CONJUGUE STABILISE: Mise en oeuvre par C. Besse
  !========--------========--------========--------========--------========--------========--------
  ! Les options sont similaires a celle du gradient conjugue

  ! Bigradient conjugue stabilise
  x=bicgstab(ACSR,IA,JA,b)
  WRITE(*, *) ' Solution obtenue par bigradient conjugue stabilise'
  WRITE(*,800) (x(i),i =1,N)





  !========--------========--------========--------========--------========--------========--------
  ! METHODE GMRES : Mise en oeuvre par C. Besse
  !========--------========--------========--------========--------========--------========--------
  ! Les options sont similaires a celle du gradient conjugue

  ! GMRES
  x=gmres(ACSR,IA,JA,b,4,1.d-10,50,BILUT,ibilut,jbilut)
  WRITE(*, *) ' Solution obtenue par Gmres'
  WRITE(*,800) (x(i),i =1,N)


  deallocate(acsr,ja,ia,e,x,b,TMP,trvec)
  deallocate(Adiag,diag)
  deallocate(BILUT, IBILUT, JBILUT)

end program test

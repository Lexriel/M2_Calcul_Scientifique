MODULE feuler_module
  
  IMPLICIT NONE
  
CONTAINS

  
  REAL FUNCTION feuler(f, tspan, y0, Nh)
     IMPLICIT NONE

! Declaration des arguments    
    REAL, INTENT(IN) :: y0 
    REAL, DIMENSION(1:2), INTENT(IN) :: tspan
    INTEGER, INTENT(IN) :: Nh


! Utiliser une fonction definie dans un autre module
  INTERFACE
    REAL FUNCTION f(x)
      REAL, INTENT(IN) :: x
    END FUNCTION f
  END INTERFACE


! Déclaration des variables locales et création de tableaux dynamiques
    INTEGER :: i
    REAL :: h, feuler
    REAL, DIMENSION(:), ALLOCATABLE :: t,u


! Définition du pas
    h=(tspan(2)-tspan(1))/Nh


! Allocation de mémoire pour les tableaux dynamiques
    allocate(t(Nh+1))
    allocate(u(Nh+1))


! Premiers éléments des tableaux t et u
    t(1)=tspan(1)
    u(1)=y0


! Ouvrir un fichier avec une unité numérotée suffisamment grande (>50) nommé feuler_data.dat de statut unknown
  open(unit=100, file='feuler_data.dat', status='unknown')
  write(100,*) t(1), u(1)

    
! Remplissage itératif des tableaux t et u
! Ecrit dans le fichier d'unité 100 les tableaux t et u    
    DO i=1,Nh       
       t(i+1)=t(i)+h
       u(i+1)=u(i)+h*f(u(i))

       write(100,*) t(i+1), u(i+1)

    END DO


! ============================================== REMARQUE ================================================
! On pourrait aussi ne pas allouer de mémoire pour des tableaux t et u et considérer t et u juste en tant
! que réels dont on change les valeurs à chaque boucle, vu que de toute façon les données sont stockées
! dans un fichier de données.
! ========================================================================================================


    feuler = u(Nh+1)
    
    deallocate(t)
    deallocate(u)
    close(100)

  END FUNCTION feuler


END MODULE feuler_module

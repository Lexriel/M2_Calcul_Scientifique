MODULE crni_module

IMPLICIT NONE

CONTAINS

REAL FUNCTION crni(f, fprime, tspan, y0, Nh)

  IMPLICIT NONE

! Déclaration des arguments    
    REAL, INTENT(IN) :: y0 
    REAL, DIMENSION(1:2), INTENT(IN) :: tspan
    INTEGER, INTENT(IN) :: Nh


! Utiliser deux fonctions définies dans un autre module
  INTERFACE
    REAL FUNCTION f(x)
      REAL, INTENT(IN) :: x
    END FUNCTION f
  END INTERFACE

  INTERFACE
    REAL FUNCTION fprime(x)
      REAL, INTENT(IN) :: x
    END FUNCTION fprime
  END INTERFACE


! Déclaration des variables locales
    INTEGER :: i
    REAL :: h, crni,r,r1,u,t,g,gprime


! Définition du pas
    h=(tspan(2)-tspan(1))/Nh


! Premiers éléments t et u
    t=tspan(1)
    u=y0


! Ouvrir un fichier avec une unité numérotée suffisamment grande (>50) nommé beuler_data.dat de statut unknown
! Ecriture des premiers éléments t et u
    open(unit=102, file="crni_data.dat", status='unknown')
    write(102,*) t,u

    
! Calcul itératif des éléments t et u (on aurait également pu les rentrer dans un tableau).
! Ecrit dans le fichier d'unité 100 les éléments t et u à chaque itération.    
DO i=1,Nh
    r=u
    r1=u+10
    
    DO WHILE (abs(r-r1) > 1e-8)
        g = r-h/2*f(r)-u-h/2*f(u)
        gprime = 1-h/2*fprime(r)
        r1=r
        r = r - g / gprime
    END DO
    t=t+h
    u=r
    write(102,*) t,u
    
END DO


! ============================================== REMARQUE ================================================
! On pourrait aussi ne pas allouer de mémoire pour des tableaux t et u et considérer t et u juste en tant
! que réels dont on change les valeurs à chaque boucle, vu que de toute façon les données sont stockées
! dans un fichier de données.
! ========================================================================================================


    crni = u
    
    close(102)

  END FUNCTION crni


END MODULE crni_module
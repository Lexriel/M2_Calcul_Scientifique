!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!               Module InitSolver                            !!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
Module InitSolver
  Use F90_Kind
  Implicit None
  ! Krylov Method : Sparsekit
  Integer, Parameter :: Iter = 1
  ! Krylov Mathod : mkl Iterative Sparse Solver
  Integer, Parameter :: mkl_iss = 11
  ! Direct Method : MUMPS
  Integer, Parameter :: Direct = 2 
  Integer, Parameter :: Direct_OneStep = 202 
  ! Direct Method : mkl Direct Sparse Solver 
  Integer, Parameter :: mkl_dss = 21
  ! Direct Method : mkl pardiso
  Integer, Parameter :: mkl_pardiso = 22
  ! Direct Method : Pastix
  Integer, Parameter :: Pastix = 23
  !
  Integer, Save :: Resolution = Iter
  Real(Double), Save :: TheTolA, TheTolR
  Integer, Save :: TheMaxItS
  Character(Len=10), Save :: TheSolver
  Integer, Save :: PrecondS
  ! ILUT (MSR matrix format from Sparsekit package)
  Integer, Parameter :: FacIlut = 1
  ! ILUT (CSR matric format from MKL package)
  Integer, Parameter :: mkl_Ilut = 11
  ! ILUT (Sparsekit) with Pivoting stategy
  Integer, Parameter :: FacIlutP = 2
  ! LIUD (MSR matricx format from Sparsekit package)
  Integer, Parameter :: FacIlud = 3
  ! IluD (MSR converted in CSR)
  Integer, Parameter :: FacIludCSR = 31
  ! IluD (MSR matrix format)
  Integer, Parameter :: FacIludP = 4
  ! Algebric Multi-Grid
  Integer, Parameter :: Agmg = 5
  !
  Real(Double), Save :: SDropTol, SPermTol, SLfil
  Integer, Save :: PrecUpD, LastPrecUpD
Contains 
!
! Lecture des parametres pour le solver de Poisson.
!
  Subroutine InitTheSolver()
!            ^^^^^^^^^^^^^
    Implicit None
    Character(Len=20) :: Com, Sprecond
!
    Open(Unit=10,File='solver.in',Status='Old',Err=200)
    Read(10,*) Com
    Read(10,*) Com
    Read(10,*) Com, TheSolver 
    Select Case (Trim(TheSolver))
    Case( 'QMR' )
       Resolution = Iter
    Case ( 'Direct' )
       Resolution = Direct
    Case Default
       Print*, "Erreur dans le choix du solver"
       Print*, Trim(TheSolver)," n'est pas propose"
       Stop "InitTheSolver"
    End Select

    Read(10,*) Com
    Read(10,*) Com, TheMaxItS
    Read(10,*) Com
    Read(10,*) Com, TheTolR
    Read(10,*) Com
    Read(10,*) Com, TheTolA
    Print*,' ## Solver de Systeme Lineaire ## '
    Print*,'    Solver Utilise : ',Trim(TheSolver)
    Print*,"    Nombre Maximum d'iterations : ", TheMaxItS
    Print*,'    Tolerances(R/A) : ',Real(TheTolR),Real(TheTolA)
    Read(10,*) Com
    Read(10,*) Com, SPrecond
    Read(10,*) Com, SDroptol
    Read(10,*) Com, SPermTol
    Read(10,*) Com, SLFil
    Read(10,*) Com
    Read(10,*) Com, PrecUpD
! Itilialisation de la derniere remise a jour du preconditionneur
    LastPrecUpD = PrecUpD + 1
    Close(10)
    Select Case ( Trim(SPrecond) )
    Case ('Ilut')
       Print*, " Precond selectionne : 'Ilut'"
       PrecondS = FacIlut
    Case ('IlutP')
       Print*, " Precond selectionne : 'IlutP'"
       PrecondS = FacIlutP
    Case ('Ilud')
       Print*, " Precond selectionne : 'Ilud'"
       PrecondS = FacIlud
    Case ('IludP')
       Print*, " Precond selectionne : 'IludP'"
       PrecondS = FacIludP
    Case Default
       Print*,'Erreur  De preconditionneur'
       Print*, Trim(Sprecond), " non diponible"
       Print*,"Valeurs possibles : 'Ilut', 'IlutP', 'Ilud', 'IludP'"
       Stop "Erreur InitTheSolver"
    End Select
    Print*, '    Droptol', SDropTol
    Print*, '    PermTol', SPermTol
    Print*, '    LFil   ', SLFil
    Return
200 Stop "Erreur a l'ouverture de `solver.in'"
  End Subroutine InitTheSolver

End Module InitSolver
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
Module Allocation
  Integer :: nmax, nzmax, maxits, lwk
  Save :: MaxitS
  Parameter (nmax=5000,nzmax=260000,Lwk=nmax*100)
  Integer :: DimIPar, DimFpar
  Parameter (DimIpar = 16, DimFpar = 16)
End Module Allocation
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!                 - Init Solver Param -                    !!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Module TuneSolver
  Use f90_kind
Contains
  Subroutine InitPar(Fpar,Ipar,NbLine)
    !            ^^^^^^^
    Use InitSolver, Only : TheMaxIts, TheTolA, TheTolR
    Use Allocation, Only : Lwk
    Implicit None
    Integer, Dimension(:), Intent(InOut) :: Ipar
    Real(Double), Dimension(:), Intent(InOut) :: Fpar
    Integer :: NbLine
    Intent(In) :: NbLine
    Ipar(:) = 0
    Fpar(:) = 0.d0
    ! Ipar(1) = 0 : Commencer le Solveur Iteratif.
    Ipar(1) = 0
    ! Ipar(2)     : Preconditionneur
    !         = 0 : pas de Preconditionneur
    !         = 1 : Preconditionneur a Gauche
    !         = 2 : Preconditionneur a Droite      
    Ipar(2) = 2
    !Ipar(2) = 0
    ! Ipar(3)     : Choix du Test de Convergence 
    !         = 2 : Test sur la difference de deux iterations successives
    Ipar(3) = 1
    ! Ipar(4)     : Taille des tableaux de Travaille (Work)
    !         = Lwk
    Ipar(4) = 11*NbLine  ! 11*n pour TfQmr
    !Ipar(4) = (NbLine+3)*(10+2) + (10+1)*10/2  !  Pour FOM
    ! Ipar(5)     : Specifique a Gmres, taille des Sous espaces de Krylov
    !         =10 : Utilise Gmres(10)
    Ipar(5) = 10
    ! Ipar(6)     : Nombre Maxi d'Iterations du Solveur
    Ipar(6) = 500
    !
    ! Fpar(1)     : Tolerance Relative
    Fpar(1) = 1.d-6
    ! Fpar(2)     : Tolerance Absolue
    Fpar(2) = 1.d-10
    Return
  End Subroutine InitPar
End Module TuneSolver


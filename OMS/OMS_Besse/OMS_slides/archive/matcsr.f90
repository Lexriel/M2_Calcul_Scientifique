Module SysLinCSR
  USE F90_kind
  USE GestionErr, Only : ErrAlloc
Implicit None

TYPE MatriceCSR
   INTEGER :: NbLine ! Nombre de lignes de la matrice
   INTEGER :: Nnz ! Nombre d'elements non nuls
   INTEGER :: Dim ! Taille des tableaux destines au stockage des elements
                  ! non nuls. Dim >= Nnz

   ! Coefficients non nuls de la matrice
   REAL*8, DIMENSION(:), POINTER :: Coeff ! Valeur des coefficients non nuls
   INTEGER, DIMENSION(:), POINTER :: JA  ! Indice des col des coeff non nuls
   INTEGER, DIMENSION(:), POINTER :: IA  ! Debut des lignes dans Coeff, JA
   ! Exemple : Les coefficients non nuls de la lignes l son stoques dans
   ! Coeff(k) et JA(k) pour k = IA(l),IA(l+1)-1

   ! Tools for matrix construction
   INTEGER :: Current_Line
   INTEGER :: Nnz_cpt
   ! Preconditionneur
   REAL*8, DIMENSION(:), POINTER :: PCoeff
   INTEGER, DIMENSION(:), POINTER :: PIA, PJA
   ! Condifugration du preconditionneur
   INTEGER :: Precond ! Nom du precondtionneur (InitSolver)
   INTEGER :: LFill ! Remplissage / matrice initiale
   REAL*8 :: DropTol ! Dropping Tolerance
   REAL*8 :: PermTol ! Permutation Tolerance   
   ! Estimation de la solution (pour initier les iterations)   
   REAL*8, DIMENSION(:), POINTER :: Guess
   ! Choix de la methode de resolution
   INTEGER :: Resolution
   ! Convergence Tol
   REAL(DOUBLE) :: TolA
   REAL(DOUBLE) :: TolR

END TYPE MatriceCSR

INTERFACE Display
   MODULE PROCEDURE Display_Full
   MODULE PROCEDURE Display_CSR
END INTERFACE

Contains
  SUBROUTINE Display_Full(A,FileName)
    REAL(Kind=8), DIMENSION(:,:), INTENT(In) :: A
    CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: FileName

    INTEGER :: IOut, i,j

    IOut = 6
    IF (PRESENT(FileName)) THEN
       IOut = 44
       OPEN(Unit=IOut,File=TRIM(FileName),ACTION='WRITE')
       
    END IF
    DO i = 1,SIZE(A,1)
       DO j = 1,SIZE(A,2)
          IF ( ABS(A(i,j)) > 1.D-15 ) &
               WRITE(IOut,*) i,j, REAL(A(i,j))
       END DO
    END DO
    IF (PRESENT(FileName)) CLOSE(IOut)
  END SUBROUTINE Display_Full

  
  
  SUBROUTINE Display_CSR(A,FileName)
    Type(MatriceCSR), INTENT(In) :: A
    CHARACTER(LEN=*), INTENT(IN), OPTIONAL :: FileName

    INTEGER :: IOut, i,j

    IOut = 6
    IF (PRESENT(FileName)) THEN
       IOut = 44
       OPEN(Unit=IOut,File=TRIM(FileName),ACTION='WRITE')
       
    END IF
    DO i = 1,A%NbLine
       DO j = A%IA(i),A%IA(i+1)-1
          WRITE(IOut,*) i,A%JA(j), REAL(A%Coeff(j))
       END DO
    END DO
    IF (PRESENT(FileName)) CLOSE(IOut)
  END SUBROUTINE Display_CSR

  ! TRANSPOSE A CSR MATRIX
  SUBROUTINE TRANSPOSE_CSR(A,B)
    ! Intent
    TYPE(MatriceCSR), INTENT(IN) :: A
    TYPE(MatriceCSR), INTENT(INOUT) :: B
    
    ! Local
    TYPE(MatriceCSR) :: Aux
    INTEGER :: nrow, ncol, Nnz
    INTEGER, DIMENSION(:), ALLOCATABLE :: iwk
    INTEGER :: Ierr
    INTEGER :: i,j

    
    Ierr = 0
    ! Get Matrix DimensionS (Matrix are NOT necessarily SQUARED)
    nrow = A%NbLine ; ncol = B%NbLine
    ! Number of Non Zero elements
    Nnz = A%IA(A%NbLine+1)-1
    
    ! Create a Copy of Input Matrix into a Local 
    CALL Create(Aux,NLine=A%NbLine,Dim=A%Nnz,TAG='Copy in TRANSPOSE_CSR')
    Aux%IA(1:A%NbLine+1) = A%IA(1:A%NbLine+1)
    Aux%JA(1:Nnz) = A%JA(1:Nnz) ; Aux%Coeff(1:Nnz) = A%Coeff(1:Nnz) 
    ! Transpose the Local Matrix
    ALLOCATE ( iwk(Nnz) )
    CALL transp (nrow,ncol,Aux%Coeff,Aux%JA,Aux%IA,iwk,ierr)
    IF ( Ierr /= 0 ) CALL Error()
    DEALLOCATE ( iwk )
    
    ! Copy the Local Transposee into the Output Matrix
    B%IA(1:B%NbLine+1) = Aux%IA(1:B%NbLine+1)
    B%JA(1:Nnz) = Aux%JA(1:Nnz) ; B%Coeff(1:Nnz) = Aux%Coeff(1:Nnz) 

    ! Destroy Local Transposee
    CALL Free(Aux)
    RETURN
  CONTAINS
    SUBROUTINE Error()
      WRITE(*,*) "Error computing the TRANSPOSE matrix", Ierr
      STOP "TRANSPOSE_CSR [matcsr.f90]"
    END SUBROUTINE Error
  END SUBROUTINE TRANSPOSE_CSR



  ! COPY MATSCR A INTO B
  ! ALLOCATIONS MUST HAVE BEEN DONE BEFORE
  SUBROUTINE COPY_Struct(A,B)
    IMPLICIT NONE
    TYPE(MatriceCSR), INTENT(IN) :: A
    TYPE(MatriceCSR), INTENT(INOUT) :: B
    
    IF ( A%NBLINE /= B%NBLINE ) THEN
       WRITE(*,*) "Matrices does not have same line numbers"
       WRITE(*,*) A%NBLINE,"/=", B%NBLINE
       STOP "COPYCSR [matcsr.f90]"
    END IF

    IF ( A%IA(B%NBLINE+1)-1 > B%DIM ) THEN
       WRITE(*,*) "Not enough space in the COPY for all coefficients"
       WRITE(*,*)  A%IA(B%NBLINE+1)-1,">",B%DIM
       STOP "COPYCSR [matcsr.f90]"
    END IF

    B%Nnz = A%IA(B%NBLINE+1)-1
    B%IA = A%IA ; B%JA = A%JA ; B%Coeff = A%Coeff

  END SUBROUTINE COPY_STRUCT

  SUBROUTINE Create(Mat,Nline,Dim,Nnz,TAG)
    IMPLICIT NONE
    ! INTENT
    ! Matrix structure
    TYPE(MatriceCSR), INTENT(INOUT) :: Mat
    ! Matrix line number
    INTEGER, INTENT(IN) :: NLine
    ! Number of Non zero coefficients
    ! Nnz : Number of non Zero elements in Matrix
    ! Dim : Estimate of Nnz or actual value of NNz if avialable 
    INTEGER, INTENT(IN) :: Dim
    ! Actual Value of Non Zero elements Number
    INTEGER, INTENT(IN), OPTIONAL :: Nnz
    ! TAG for error handling and output messages
    CHARACTER(LEN=*), OPTIONAL :: TAG
    ! LOCAL
    INTEGER :: Ierr(2)
    ! Size of IA and JA,Coeff
    Mat%NbLine = NLine ; Mat%Dim = Dim
    IF (PRESENT(Nnz)) Mat%Nnz = Nnz
    
    IF ( ASSOCIATED(Mat%IA) ) DEALLOCATE (Mat%IA)
    IF ( ASSOCIATED(Mat%Coeff) ) DEALLOCATE (Mat%Coeff)
    IF ( ASSOCIATED(Mat%JA) ) DEALLOCATE (Mat%JA)

    ALLOCATE ( Mat%IA(Mat%NbLine+1), STAT=Ierr(1) )
    ALLOCATE ( Mat%Coeff(Mat%Dim), Mat%JA(Mat%Dim), STAT=Ierr(2) )
    
    IF ( ANY(Ierr/=0) ) THEN
       WRITE(*,*) "ERROR ALLOCATE MATRIX COMPONENT"
       IF (PRESENT(TAG)) &
            WRITE(*,*) " Matrix Name :    ", TRIM(TAG)
       WRITE(*,*) " Number of LINES :", Mat%NbLine
       WRITE(*,*) " Estimate of NNz :", Mat%Dim
       IF (PRESENT(Nnz)) &
       WRITE(*,*) " Actual NNz      :", Mat%Nnz
       STOP "CREATE MATRIX [matcsr.f90]"
    END IF
    ! Init Tools for matrix construction
    Call Init_Store(Mat)
  END SUBROUTINE Create

SUBROUTINE Init_Store(Mat)
  TYPE(MatriceCSR), INTENT(INOUT) :: Mat
  Mat%Current_Line = 0
  Mat%Nnz_cpt = 0
END SUBROUTINE Init_Store

SUBROUTINE Store(Mat,Val,Col)
  IMPLICIT NONE
  ! INTENT
  ! Matrix structure
  TYPE(MatriceCSR), INTENT(INOUT) :: Mat
  ! Coefficient Value
  REAL(DOUBLE), INTENT(IN) :: Val
  ! Column of the coefficient
  INTEGER :: Col
  
  ! Increment number of Nnz stored in Matrix
  Mat%Nnz_cpt = Mat%Nnz_cpt + 1
  ! Store it
  Mat%Coeff(Mat%Nnz_cpt) = Val ; Mat%JA(Mat%Nnz_cpt) = Col
END SUBROUTINE Store

SUBROUTINE NewLine(Mat)
IMPLICIT NONE
! INTENT
! Matrix structure
TYPE(MatriceCSR), INTENT(INOUT) :: Mat

Mat%Current_Line = Mat%Current_Line + 1 
Mat%IA(Mat%Current_Line) = Mat%Nnz_Cpt + 1
END SUBROUTINE NewLine

SUBROUTINE Finalize(Mat)
! The Matrix has been stored. 
IMPLICIT NONE
! INTENT
! Matrix structure
TYPE(MatriceCSR), INTENT(INOUT) :: Mat

Mat%Nnz = Mat%Nnz_Cpt
Mat%IA(Mat%Current_Line+1) = Mat%Nnz + 1

END SUBROUTINE Finalize

SUBROUTINE Free(Mat)
  IMPLICIT NONE
  TYPE(MatriceCSR), INTENT(INOUT) :: Mat

  Mat%Dim = -1
  Mat%nnz = 0 ; Mat%Nnz_cpt = 0
  Mat%Nbline = 0 ; Mat%Current_Line = 0

  IF ( ASSOCIATED(Mat%IA) ) DEALLOCATE(Mat%IA)
  IF ( ASSOCIATED(Mat%JA) ) DEALLOCATE(Mat%JA)
  IF ( ASSOCIATED(Mat%Coeff) ) DEALLOCATE(Mat%Coeff)

END SUBROUTINE Free

SUBROUTINE Destroy(Mat)
  Use InitSolver, Only : Resolution, Iter
  IMPLICIT NONE
  TYPE(MatriceCSR), INTENT(INOUT) :: Mat

  Mat%Dim = -1
  Mat%nnz = 0 ; Mat%Nnz_cpt = 0
  Mat%Nbline = 0 ; Mat%Current_Line = 0



  IF ( ASSOCIATED(Mat%IA) ) DEALLOCATE(Mat%IA)
  IF ( ASSOCIATED(Mat%JA) ) DEALLOCATE(Mat%JA)
  IF ( ASSOCIATED(Mat%Coeff) ) DEALLOCATE(Mat%Coeff)

  IF ( ASSOCIATED(Mat%PCoeff) ) DEALLOCATE (Mat%PCoeff)

  IF ( ASSOCIATED(Mat%PJA) ) DEALLOCATE (Mat%PJA)
  IF ( ASSOCIATED(Mat%PIA) ) DEALLOCATE (Mat%PIA)
  
  IF ( ASSOCIATED(Mat%Guess) ) DEALLOCATE (Mat%Guess)


!!$  SELECT CASE ( Mat%Resolution )
!!$  CASE ( Direct ) 
!!$     !CALL mumps_driver(id=Mat%mumps_id,Job=mumps_clean)
!!$  END SELECT
END SUBROUTINE Destroy


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Subroutine CalPrecond(A)
!          ^^^^^^^^^^       
  Use InitSolver, Only : FacIlut, FacilutP, Facilud, FAcIludCSR, &
                         FaciludP, SDropTol, SPermTol, SLFil
  Use PreCond, Only : IluD, IluT, IluDP, ILutP
!  Use mkl_Interface_Mod, only : ILUT_SIZE, mkl_dcsrilut
  Implicit None
! Arguments en entree
  Type(MatriceCSR),  Intent(INOUT)  :: A
! Local
  Real(Double) :: DropTol, PermTol, Alpha
  Integer :: Nbline, Lfil, MBloc, Nwk, Ierr
  Real(Double), Dimension(:), Allocatable :: W , Wk, P
  Integer, Dimension(:), Allocatable :: JW, Iperm, Iwk, IP, JP
  Integer :: DIM_P, lnnz, maxfil
! Init
  Nbline = Size(A%IA) - 1
! Parametre pour la construction du Preconditionneur (Voir ilut.f)
!      Lfil    : Integer (>0)
!                Nb max d'elements pour chaque ligne 
!                (hors element diagonal)
!  Lfil = SLFil
  Lfil = A%LFill
!      DropTol : Double (>0)
!                Dropping Tolerance
!                Limite sous laquelle les elements ne sont
!                plus pris en comptes.
  DropTol = A%DropTol
!      PermTol : Double (>=0)
!                Limite Sous laquelle deux colonnes sont permuttees
!                Deux colonnes permuttent lorsque 
!                  A(i,j)*PermTol > a(i,i)
!                [PermTol = 0 pas de permuttation,
!                 Valeur  conseillees 1.d-1  a 1.d-2]
  PermTol = A%PermTol
!      Mbloc   : Integer
!                Raffinement pour les permuttations.
!                [MBloc = Size(IA) supprime]
  MBloc = NbLine
!      Nwk     : Integer (>0)
!                Taille des tableaux de coeff non nuls du precond
!                Si Nwk est trop petit Ierr=-2 ou -3
!  Nwk = Size(A%PCoeff) - 1 -> See Precond allocation
!      Alpha   : Double (=0 ou 1)
!                Parametre de Compensation Diagonal
!                alph*(Somme des elements negliges) est ajoute
!                a la diagonale
  Alpha = 1.d0 ! compensation Diagonale selectionnee
!      
! Allocation de Work
  Allocate(w(Nbline*2),Stat=Ierr)
  Call ErrAlloc(Ierr,'Precond: w')
  Allocate(jw(2*Nbline), Stat=Ierr)
  Call ErrAlloc(Ierr,'Precond: Jw')
  Select Case (A%Precond)
  Case (FacIlud, FaciludCSR)             
     DIM_P = 3*A%Lfill*A%Dim
     Allocate(A%PCoeff(DIM_P),A%PJA(DIM_P),A%PIA(NbLine+1))
     Call IluD                                     &
          (NBline,A%Coeff,A%JA,A%IA,Alpha,DropTol, &
                         A%PCoeff,A%PJA,A%PIA,Size(A%PCoeff)-1,W,Jw,Ierr)
     IF ( A%Precond == FaciludCSR ) THEN
        ALLOCATE(P(SIZE(A%PCoeff)), IP(SIZE(A%PIA,1)), JP(SIZE(A%PJA,1)))
        ALLOCATE( wk(NbLine) , iwk(Nbline+1) )
        P = A%Coeff; JP = A%PJA
        !      subroutine msrcsr (n,a,ja,ao,jao,iao,wk,iwk)
        CALL msrcsr (NBline,P,JP,A%PCoeff,A%PJA,A%PIA,wk,iwk)
        DEALLOCATE (wk, iwk, P, IP, JP)
     END IF
!  Case (mkl_ilut)
!     ! Transform relative to absolute fill-in
!!!$     lnnz = NINT((A%IA(Nbline+1)-1)/DBLE(Nbline)) ! Non Zero per line
!!!$     maxfil = lnnz * A%Lfill
!!!$!     DIM_P = ILUT_SIZE(Nbline,maxfil)    
!!!$!     Print*, 'DIM_P',Nbline,A%LFill,DIM_P
!!!$!     Allocate(A%PCoeff(DIM_P),A%PJA(DIM_P),A%PIA(Nbline+1))
!!!$!     Call mkl_dcsrilut                                    &
!!!$!          (Nbline,A%Coeff,A%JA,A%IA,A%PCoeff,A%PJA,A%PIA,maxfil)
  Case (FacIlut)
     DIM_P = 3*A%Lfill*A%Dim
     NWK=DIM_P
     Allocate(A%PCoeff(DIM_P),A%PJA(DIM_P),A%PIA(Nbline+1))
     Call IluT                                     &
          (Nbline,A%Coeff,A%JA,A%IA,Lfil,DropTol,  &
                        A%PCoeff,A%PJA,A%PIA,Nwk,w,jw,Ierr)
  Case (FacIludP)
     DIM_P = 3*A%Lfill*SIZE(A%Coeff,1)
     NWK=DIM_P
     Allocate(A%PCoeff(DIM_P),A%PJA(DIM_P),A%PIA(Nbline+1))
     Allocate ( Iperm(NbLine*2) )
     Call IludP                                    &
             (Nbline,A%Coeff,A%JA,A%IA,Alpha,      &
                     DropTol,PermTol,MBloc,        &
               A%PCoeff,A%PJA,A%PIA,NWK,w,jw,Iperm,ierr)
  Case (FacIlutP)
     DIM_P = 3*A%Lfill*SIZE(A%Coeff,1)
     NWK=DIM_P
     Allocate(A%PCoeff(DIM_P),A%PJA(DIM_P),A%PIA(Nbline+1))
     Allocate ( Iperm(NbLine*2) )
     Call IlutP                                    &
          (Nbline,A%Coeff,A%JA,A%IA,LFil,DropTol,  &
                   PermTol,MBloc,                  &
                  A%PCoeff,A%PJA,A%PIA,NWK,w,jw,Iperm,ierr)
     DeAllocate( Iperm )
  End Select
  Deallocate(jw,w)
  If (Ierr .ne. 0 ) then
     Print*,"Erreur Factorisation Ierr = ",Ierr
     Print*,Char(7)
     Stop "Erreur CalPrecond"
  End If
  Return
End Subroutine CalPrecond
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!                   RESOLUTION D'UN SYSTEME LINEAIRE               !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
Subroutine Resol(Mat,RhsIn,Sol)
!          ^^^^^
  Use Allocation, Only : DimIpar, DimFpar
  Use InitSolver, Only : Resolution, Iter
  Use TuneSolver, Only : InitPar
  Use ResolRun, Only : RunSolver
!  Use MKL_Interface_Mod, Only : iss_interface, dss_interface, pardiso_interface
!  Use pastix_interface_mod
  Implicit None
!
! Intent
  Type(MatriceCSR), Intent(InOut) :: Mat
  Real(Double), Dimension(:), Intent(In) :: RhsIn
  Real(Double), Dimension(:), Intent(InOut) :: Sol
!
! Variables Locales
  Integer, Dimension(DimIpar) :: Ipar
  Real(Double), Dimension(DimFpar) :: Fpar
  Integer :: Nbline
  Real(Double), Dimension(:), Allocatable :: Rhs

!
  NbLine = Size(RhsIn,1)
  Allocate(Rhs(NbLine)) ; Rhs = RhsIn
  Select Case (Mat%Resolution)       
  Case ( Iter )
     ! Krylov methods from Sparsekit package
     ! Approximation of the solution 
     Sol(:) = Mat%Guess(:)          ! Init iteration process with guess
     Call InitPar(Fpar,Ipar,NbLine) ! Set Up the Solver
     Call RunSolver(Nbline,Rhs,Sol,Ipar,Fpar,    &
          Mat%Coeff,Mat%JA,Mat%IA,               &
          Mat%PCoeff,Mat%PJA,Mat%PIA)
     Mat%Guess(:) = Sol(:)          ! Store solution
  Case Default
     WRITE(*,*) "This choice is not supported"
     STOP "Resol"
  End Select
  !
  Return
End Subroutine Resol

! Converts CSR TO CSC or CSC TO CSR 
SUBROUTINE CsXCsY(A,B) 
  IMPLICIT NONE
  ! Intent ...
  TYPE(MatriceCSR), INTENT(IN) :: A
  TYPE(MatriceCSR), INTENT(INOUT) :: B
  ! Local
  INTEGER :: NLine, job, ipos

  NLine = A%NbLine
  IF ( A%NbLine /= B%NbLine ) Call Error(1)
  IF ( A%Nnz > B%Dim) Call Error(2)

  job = 1 ; ipos = 1
  Call CSRCSC2(NLine,NLine,job,ipos,A%Coeff,A%JA,A%IA,B%Coeff,B%JA,B%IA)
  B%Nnz = A%Nnz

  RETURN
Contains
  Subroutine Error(Ierr)
    INTEGER, INTENT(IN) :: Ierr    
    WRITE(*,*) "Matrices non Conformant"
    SELECT CASE (Ierr)
    CASE (1)
       WRITE(*,*) "Number of Rows"
       WRITE(*,*) " INput Matrix  :", A%NbLine
       WRITE(*,*) " OUTput Matrix :", B%NbLine
    CASE(2)
       WRITE(*,*) "Size"
       WRITE(*,*) " INput Matrix  :", A%Nnz
       WRITE(*,*) " OUTput Matrix :", B%Dim
    END SELECT
    STOP "CsXCsY [matcsr.f90] "
  End Subroutine Error
END SUBROUTINE CsXCsY


SUBROUTINE CSRAplusB(A,B,C)
  USE BLASS, ONLY : AplB
  IMPLICIT NONE  
  ! INTENT
  TYPE(MatriceCSR), INTENT(IN) :: A,B
  TYPE(MatriceCSR), INTENT(InOUT) :: C
  ! INTENT
  INTEGER :: Nrow, NCol, NNz_max
  INTEGER :: Ierr
  INTEGER, DIMENSION(:), ALLOCATABLE :: iw
  LOGICAL :: values
  EXTERNAL :: csort
  Nrow = A%Nbline ; NCol = NRow   
  NNz_max = C%Dim

  ALLOCATE ( iw(Ncol) )
  CALL AplB (NRow,NCol,A%Coeff,A%JA,A%IA,B%Coeff,B%JA,B%IA,&
       C%Coeff,C%JA,C%IA,Nnz_max,iw,ierr)
  C%Nnz = C%IA(C%NbLine+1)-1
  DEALLOCATE(iw)
  IF (Ierr/=0) THEN
     WRITE(*,*) "Error In Summing Matrices", Ierr
     STOP "CSRAplusB [matcsr.f90]"
  END IF

  ! Sort elements in increasing column number
  !  CALL CSR_SORT(C)
END SUBROUTINE CSRAplusB


! Reorder Rows for Matrix stored accordig PERMUT order.
SUBROUTINE Order_Rows(A,IPermut,REALLOC)
  USE Tri_Mod
  IMPLICIT NONE
  ! Intent
  TYPE(MatriceCSR), INTENT(INOUT) :: A
  INTEGER, DIMENSION(:), INTENT(IN) :: IPermut
  LOGICAL, OPTIONAL :: REALLOC
  ! Local
  TYPE(MatriceCSR) :: Aux
  INTEGER, DIMENSION(:), ALLOCATABLE :: Irank
  INTEGER :: NRow, i, j, ii, Nnz
  
  NRow = A%NbLine

  ! Create a Copy of input matrix
  CALL Create(Aux,Dim=A%Nnz,Nnz=A%nnz,NLine=NRow,TAG='Aux Order_Rows')
  CALL Copy_Struct(A,Aux)

  ! Resize input matrix to the exacte space needed
  IF (PRESENT(REALLOC)) THEN
     Nnz = A%IA(NRow+1)-1 ; A%Nnz = Nnz ; A%Dim = Nnz 
     DEALLOCATE( A%Coeff ) ; ALLOCATE (A%Coeff(A%Dim))
     DEALLOCATE( A%JA)     ; ALLOCATE (A%JA   (A%DIM))
  END IF
  
  ! Compute Backward permutations
  ALLOCATE ( IRank(NRow) )
  CALL INDEXX(NRow, IPermut, Irank) 

  ! Re-order lines in input matrix
  Call Init_Store(A)
  DO i = 1,NRow
     ! Start New Line
    ii = Irank(i) ! rank of the (i)th-line in out-of-ordered matrix
     Call NewLine(A)
     DO j = Aux%IA(ii),Aux%IA(ii+1)-1 ! Copy the line
        Call Store(A,Aux%Coeff(j),Aux%JA(j))
     END DO
  END DO
  CALL Finalize(A)
  
  ! Free memory
  DEALLOCATE(IRank)
  Call Free(Aux)
END SUBROUTINE ORDER_ROWS


SUBROUTINE ORDER_COLUMNS(A)
  USE TRI_MOD
  IMPLICIT NONE
  ! Intent
  Type(MatriceCSR), INTENT(INOUT) :: A
  ! Local
  INTEGER :: NRow, Cpt, i, j
  INTEGER, DIMENSION(:), ALLOCATABLE :: Rows_Nnz, Irank, Columns
  REAL(Kind=8), DIMENSION(:), ALLOCATABLE :: Coeffs
     
  NRow = A%NbLine
  
  ! Compute Number of Non Zeros in each Row
  ALLOCATE(Rows_Nnz(NRow))
  Rows_Nnz = A%IA(2:NRow+1)-A%IA(1:NRow)
  ! Allocate SPACE for elements Column Number  
  ALLOCATE( Columns(1:MAXVAL(Rows_NNz(:))) )
  ALLOCATE( Coeffs (1:MAXVAL(Rows_NNz(:))) )
  ALLOCATE( Irank  (1:MAXVAL(Rows_NNz(:))) )
  ! Order Rows
  DO i = 1, NRow
     Columns(1:Rows_Nnz(i)) = A%JA   (A%IA(i):A%IA(i+1)-1)
     Coeffs (1:Rows_Nnz(i)) = A%Coeff(A%IA(i):A%IA(i+1)-1) 
     CALL Indexx(Rows_Nnz(i),Columns,Irank)
     ! Column(Irank) are the ordered column indexes
     Cpt = 0
     DO j = A%IA(i),A%IA(i+1)-1
        Cpt = Cpt + 1
        A%JA(j)    = Columns( Irank(Cpt) )
        A%Coeff(j) = Coeffs ( Irank(Cpt) )
     END DO
  END DO

  ! Free memory
  DEALLOCATE ( Irank, Rows_Nnz, Columns, Coeffs )
END SUBROUTINE ORDER_COLUMNS
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!                      PRODUIT MATRICE VECTEUR                     !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
Subroutine CSRAmuX(Mat,X,Res)
Use Blass, Only : AmuX
Implicit None
! Intent
Type(MatriceCSR) :: Mat
Real(Double), Dimension(:) :: X, Res
Intent(In) :: Mat, X
Intent(Out) :: Res
! Local
Integer :: NbLine
!
NbLine = Size(Mat%IA)-1
Call AmuX(NbLine,X,Res,Mat%Coeff,Mat%JA,Mat%IA)
!
Return
End Subroutine CSRAmuX




!----------------------------------------------------------------------- 
subroutine csrcsc (n,job,ipos,a,ja,ia,ao,jao,iao)
  IMPLICIT NONE
  INTEGER :: n, ipos, job
  integer :: ia(:),iao(:),ja(:),jao(:)
  real*8  :: a(:),ao(:)

!-----------------------------------------------------------------------
! Compressed Sparse Row     to      Compressed Sparse Column
!
! (transposition operation)   Not in place. 
!----------------------------------------------------------------------- 
! -- not in place --
! this subroutine transposes a matrix stored in a, ja, ia format.
! ---------------
! on entry:
!----------
! n	= dimension of A.
! job	= integer to indicate whether to fill the values (job.eq.1) of the
!         matrix ao or only the pattern., i.e.,ia, and ja (job .ne.1)
!
! ipos  = starting position in ao, jao of the transposed matrix.
!         the iao array takes this into account (thus iao(1) is set to ipos.)
!         Note: this may be useful if one needs to append the data structure
!         of the transpose to that of A. In this case use for example
!                call csrcsc (n,1,ia(n+1),a,ja,ia,a,ja,ia(n+2)) 
!	  for any other normal usage, enter ipos=1.
! a	= real array of length nnz (nnz=number of nonzero elements in input 
!         matrix) containing the nonzero elements.
! ja	= integer array of length nnz containing the column positions
! 	  of the corresponding elements in a.
! ia	= integer of size n+1. ia(k) contains the position in a, ja of
!	  the beginning of the k-th row.
!
! on return:
! ---------- 
! output arguments:
! ao	= real array of size nzz containing the "a" part of the transpose
! jao	= integer array of size nnz containing the column indices.
! iao	= integer array of size n+1 containing the "ia" index array of
!	  the transpose. 
!
!----------------------------------------------------------------------- 
      call csrcsc2 (n,n,job,ipos,a,ja,ia,ao,jao,iao)
    end subroutine csrcsc
!-----------------------------------------------------------------------
    subroutine csrcsc2 (n,n2,job,ipos,a,ja,ia,ao,jao,iao)
      IMPLICIT NONE
      INTEGER :: n, n2, job, ipos
      integer ia(n+1),iao(n2+1),ja(:),jao(:)
      real*8  :: a(:),ao(:)
      INTEGER :: i,j,k,next
!-----------------------------------------------------------------------
! Compressed Sparse Row     to      Compressed Sparse Column
!
! (transposition operation)   Not in place. 
!----------------------------------------------------------------------- 
! Rectangular version.  n is number of rows of CSR matrix,
!                       n2 (input) is number of columns of CSC matrix.
!----------------------------------------------------------------------- 
! -- not in place --
! this subroutine transposes a matrix stored in a, ja, ia format.
! ---------------
! on entry:
!----------
! n	= number of rows of CSR matrix.
! n2    = number of columns of CS! matrix.
! job	= integer to indicate whether to fill the values (job.eq.1) of the
!         matrix ao or only the pattern., i.e.,ia, and ja (job .ne.1)
!
! ipos  = starting position in ao, jao of the transposed matrix.
!         the iao array takes this into account (thus iao(1) is set to ipos.)
!         Note: this may be useful if one needs to append the data structure
!         of the transpose to that of A. In this case use for example
!                call csrcsc2 (n,n,1,ia(n+1),a,ja,ia,a,ja,ia(n+2)) 
!	  for any other normal usage, enter ipos=1.
! a	= real array of length nnz (nnz=number of nonzero elements in input 
!         matrix) containing the nonzero elements.
! ja	= integer array of length nnz containing the column positions
! 	  of the corresponding elements in a.
! ia	= integer of size n+1. ia(k) contains the position in a, ja of
!	  the beginning of the k-th row.
!
! on return:
! ---------- 
! output arguments:
! ao	= real array of size nzz containing the "a" part of the transpose
! jao	= integer array of size nnz containing the column indices.
! iao	= integer array of size n+1 containing the "ia" index array of
!	  the transpose. 
!
!----------------------------------------------------------------------- 
!----------------- compute lengths of rows of transp(A) ----------------
      do i=1,n2+1
         iao(i) = 0
      END do
      do i=1, n
         do k=ia(i), ia(i+1)-1 
            j = ja(k)+1
            iao(j) = iao(j)+1
         End do
      END do
!---------- compute pointers from lengths ------------------------------
      iao(1) = ipos 
      do  i=1,n2
         iao(i+1) = iao(i) + iao(i+1)
      END do
!--------------- now do the actual copying ----------------------------- 
      do i=1,n
         do k=ia(i),ia(i+1)-1 
            j = ja(k) 
            next = iao(j)
            if (job .eq. 1)  ao(next) = a(k)
            jao(next) = i
            iao(j) = next+1
         ENd do
      END do
!-------------------------- reshift iao and leave ---------------------- 
      do i=n2,1,-1
         iao(i+1) = iao(i)
      End do
      iao(1) = ipos
!--------------- end of csrcsc2 ---------------------------------------- 
!-----------------------------------------------------------------------
end subroutine csrcsc2

subroutine csort (n,a,ja,ia,iwork,values) 
  logical values
  integer n, ja(:), ia(:), iwork(:) 
  real*8 a(:) 
  !-----------------------------------------------------------------------
  ! This routine sorts the elements of  a matrix (stored in compressed
  ! Sparse Row Format) in increasing order of their column indices within 
  !each row. It uses a form of bucket sort with a cost of O(nnz) where
  !nnz = number of nonzero elements. 
  !requires an integer work array of length 2*nnz.  
  !-----------------------------------------------------------------------
  ! on entry:
  !--------- 
  ! n     = the row dimension of the matrix
  ! a     = the matrix A in compressed sparse row format.
  ! ja    = the array of column indices of the elements in array a.
  ! ia    = the array of pointers to the rows. 
  ! iwork = integer work array of length max ( n+1, 2*nnz ) 
  !         where nnz = (ia(n+1)-ia(1))  ) .
  ! values= logical indicating whether or not the real values a(*) must 
  !         also be permuted. if (.not. values) then the array a is not
  !         touched by csort and can be a dummy array. 
  ! 
  ! on return:
  !----------
  ! the matrix stored in the structure a, ja, ia is permuted in such a
  ! way that the column indices are in increasing order within each row.
  ! iwork(1:nnz) contains the permutation used  to rearrange the elements.
  !----------------------------------------------------------------------- 
  ! Y. Saad - Feb. 1, 1991.
  !-----------------------------------------------------------------------
  ! local variables
  integer i, k, j, ifirst, nnz, next  
  integer ko, irow
  !
  ! count the number of elements in each column
  !
  DO i=1,n+1
     iwork(i) = 0
  END DO
  DO i=1, n
     DO k=ia(i), ia(i+1)-1 
        j = ja(k)+1
        iwork(j) = iwork(j)+1
     END DO
  END DO
!
! compute pointers from lengths. 
!
  iwork(1) = 1
  DO i=1,n
     iwork(i+1) = iwork(i) + iwork(i+1)
  END DO
! 
! get the positions of the nonzero elements in order of columns.
!
  ifirst = ia(1) 
  nnz = ia(n+1)-ifirst
  DO i=1,n
     DO k=ia(i),ia(i+1)-1 
        j = ja(k) 
        next = iwork(j) 
        iwork(nnz+next) = k
        iwork(j) = next+1
     END DO
  END DO
!
! convert to coordinate format
! 
  DO i=1, n
     DO k=ia(i), ia(i+1)-1 
        iwork(k) = i
     END DO
  END DO
!
! loop to find permutation: for each element find the correct 
! position in (sorted) arrays a, ja. Record this in iwork. 
! 
  DO k=1, nnz
     ko = iwork(nnz+k) 
     irow = iwork(ko)
     next = ia(irow)
!
! the current element should go in next position in row. iwork
! records this position. 
! 
     iwork(ko) = next
     ia(irow)  = next+1
  END DO
!
! perform an in-place permutation of the  arrays.
! 
  call ivperm (nnz, ja(ifirst), iwork) 
  if (values) call dvperm (nnz, a(ifirst), iwork) 
!
! reshift the pointers of the original matrix back.
! 
  DO i=n,1,-1
     ia(i+1) = ia(i)
  END DO
  ia(1) = ifirst 
!
return 
!---------------end-of-csort-------------------------------------------- 
!-----------------------------------------------------------------------
end subroutine csort



End Module SysLinCSR


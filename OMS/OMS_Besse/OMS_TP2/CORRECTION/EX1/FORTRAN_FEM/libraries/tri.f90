MODULE TRI_MOD


  INTERFACE INDEXX
     MODULE PROCEDURE INDEXR
     MODULE PROCEDURE INDEXI
  END INTERFACE

CONTAINS
SUBROUTINE DSVRGP(N, RA, RB, IPERM) 
Implicit None
INTEGER :: N, IPERM(:)
DOUBLE PRECISION :: RA(:), RB(:)
INTENT(IN) :: N
DOUBLE PRECISION, Dimension(:), Allocatable :: Tmp
!
! RA tableau a trier, RB tableau Trie, IPerm ensemble des permutations
! RB = RA(IPerm)
!
!Stop "Appel au tri [tri.f90]"
Call INDEXX(N, RA, IPERM)
!
! RA et RB peuvent partager le meme adressage memoire
! 
Allocate ( Tmp(N) )
Tmp (1:N) = RA(IPERM(1:N))
RB(1:N) = Tmp(1:N)
DeAllocate( Tmp )
END SUBROUTINE DSVRGP

SUBROUTINE INDEXR(N,ARR,INDX)
! 
!.. IMPLICITS .. 
IMPLICIT NONE
! 
!.. PARAMETERS .. 
INTEGER M
INTEGER NSTACK
!************************************************************
!                                                           *
!  SORT ARRAYS USING THE QUICKSORT ALGORITHM (PRESS 2ND)    *
!                                                           *
!************************************************************
!
      PARAMETER (M = 7, NSTACK = 200)
! 
!.. FORMAL ARGUMENTS .. 
      INTEGER INDX(:),N
      DOUBLE PRECISION ARR(:)
      INTENT(IN) :: N, ARR
      INTENT(OUT) :: INDX
! 
!.. LOCAL SCALARS .. 
      INTEGER I,INDXT,IR,ITEMP,J,JSTACK,K,L
      DOUBLE PRECISION A
! 
!.. LOCAL ARRAYS .. 
      INTEGER ISTACK(NSTACK)
! 
! ... EXECUTABLE STATEMENTS ...
! 
!
      DO 100 J = 1,N
         INDX(J) = J
 100  CONTINUE
!
      JSTACK = 0
      L = 1
      IR = N
!
 400  CONTINUE
      IF (IR-L .LT. M) THEN
         DO 200 J = L+1,IR
            INDXT = INDX(J)
            A = ARR(INDXT)
            DO 300 I = J-1,1,-1
               IF (ARR(INDX(I)) .LE. A) GOTO 800
               INDX(I+1) = INDX(I)
 300        CONTINUE
            I = 0
 800        CONTINUE
            INDX(I+1) = INDXT
 200     CONTINUE
         IF (JSTACK .EQ. 0) THEN
            RETURN
         ELSE
            IR = ISTACK(JSTACK)
            L = ISTACK(JSTACK-1)
            JSTACK = JSTACK - 2
         ENDIF
      ELSE
         K = (L+IR) / 2
         ITEMP = INDX(K)
         INDX(K) = INDX(L+1)
         INDX(L+1) = ITEMP
         IF (ARR(INDX(L+1)) .GT. ARR(INDX(IR))) THEN
            ITEMP = INDX(L+1)
            INDX(L+1) = INDX(IR)
            INDX(IR) = ITEMP
         ENDIF
         IF (ARR(INDX(L)) .GT. ARR(INDX(IR))) THEN
            ITEMP = INDX(L)
            INDX(L) = INDX(IR)
            INDX(IR) = ITEMP
         ENDIF
         IF (ARR(INDX(L+1)) .GT. ARR(INDX(L))) THEN
            ITEMP = INDX(L+1)
            INDX(L+1) = INDX(L)
            INDX(L) = ITEMP
         ENDIF
!
         I = L + 1
         J = IR
         INDXT = INDX(L)
         A = ARR(INDXT)
!
 500     CONTINUE
         I = I + 1
         IF (ARR(INDX(I)) .GE. A) THEN
 600        CONTINUE
            J = J - 1
            IF (ARR(INDX(J)) .GT. A) GOTO 600
            IF (J .LT. I) GOTO 700
!
            ITEMP = INDX(I)
            INDX(I) = INDX(J)
            INDX(J) = ITEMP
         ENDIF
         GOTO 500
!
 700     CONTINUE
         INDX(L) = INDX(J)
         INDX(J) = INDXT
         JSTACK = JSTACK + 2
         IF (JSTACK .GT. NSTACK) THEN
            GOTO 900
!
         ELSEIF (IR-I+1 .GE. J-L) THEN
            ISTACK(JSTACK) = IR
            ISTACK(JSTACK-1) = I
            IR = J - 1
         ELSE
            ISTACK(JSTACK) = J - 1
            ISTACK(JSTACK-1) = L
            L = I
         ENDIF
      ENDIF
      GOTO 400
 900  CONTINUE
      WRITE (*,*) 'NSTACK TOO SMALL IN INDEXX!'
      STOP
!
      END SUBROUTINE INDEXR

SUBROUTINE INDEXI(N,ARR,INDX)
! 
!.. IMPLICITS .. 
IMPLICIT NONE
! 
!.. PARAMETERS .. 
INTEGER M
INTEGER NSTACK
!************************************************************
!                                                           *
!  SORT ARRAYS USING THE QUICKSORT ALGORITHM (PRESS 2ND)    *
!                                                           *
!************************************************************
!
      PARAMETER (M = 7, NSTACK = 200)
! 
!.. FORMAL ARGUMENTS .. 
      INTEGER INDX(:),N
      INTEGER ARR(:)
      INTENT(IN) :: N, ARR
      INTENT(OUT) :: INDX
! 
!.. LOCAL SCALARS .. 
      INTEGER I,INDXT,IR,ITEMP,J,JSTACK,K,L
      INTEGER A
! 
!.. LOCAL ARRAYS .. 
      INTEGER ISTACK(NSTACK)
! 
! ... EXECUTABLE STATEMENTS ...
! 
!
      DO 100 J = 1,N
         INDX(J) = J
 100  CONTINUE
!
      JSTACK = 0
      L = 1
      IR = N
!
 400  CONTINUE
      IF (IR-L .LT. M) THEN
         DO 200 J = L+1,IR
            INDXT = INDX(J)
            A = ARR(INDXT)
            DO 300 I = J-1,1,-1
               IF (ARR(INDX(I)) .LE. A) GOTO 800
               INDX(I+1) = INDX(I)
 300        CONTINUE
            I = 0
 800        CONTINUE
            INDX(I+1) = INDXT
 200     CONTINUE
         IF (JSTACK .EQ. 0) THEN
            RETURN
         ELSE
            IR = ISTACK(JSTACK)
            L = ISTACK(JSTACK-1)
            JSTACK = JSTACK - 2
         ENDIF
      ELSE
         K = (L+IR) / 2
         ITEMP = INDX(K)
         INDX(K) = INDX(L+1)
         INDX(L+1) = ITEMP
         IF (ARR(INDX(L+1)) .GT. ARR(INDX(IR))) THEN
            ITEMP = INDX(L+1)
            INDX(L+1) = INDX(IR)
            INDX(IR) = ITEMP
         ENDIF
         IF (ARR(INDX(L)) .GT. ARR(INDX(IR))) THEN
            ITEMP = INDX(L)
            INDX(L) = INDX(IR)
            INDX(IR) = ITEMP
         ENDIF
         IF (ARR(INDX(L+1)) .GT. ARR(INDX(L))) THEN
            ITEMP = INDX(L+1)
            INDX(L+1) = INDX(L)
            INDX(L) = ITEMP
         ENDIF
!
         I = L + 1
         J = IR
         INDXT = INDX(L)
         A = ARR(INDXT)
!
 500     CONTINUE
         I = I + 1
         IF (ARR(INDX(I)) .GE. A) THEN
 600        CONTINUE
            J = J - 1
            IF (ARR(INDX(J)) .GT. A) GOTO 600
            IF (J .LT. I) GOTO 700
!
            ITEMP = INDX(I)
            INDX(I) = INDX(J)
            INDX(J) = ITEMP
         ENDIF
         GOTO 500
!
 700     CONTINUE
         INDX(L) = INDX(J)
         INDX(J) = INDXT
         JSTACK = JSTACK + 2
         IF (JSTACK .GT. NSTACK) THEN
            GOTO 900
!
         ELSEIF (IR-I+1 .GE. J-L) THEN
            ISTACK(JSTACK) = IR
            ISTACK(JSTACK-1) = I
            IR = J - 1
         ELSE
            ISTACK(JSTACK) = J - 1
            ISTACK(JSTACK-1) = L
            L = I
         ENDIF
      ENDIF
      GOTO 400
 900  CONTINUE
      WRITE (*,*) 'NSTACK TOO SMALL IN INDEXX!'
      STOP
!
      END SUBROUTINE INDEXI


!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      SUBROUTINE RANK(N,INDX,IRANK)
      IMPLICIT NONE
! 
!.. FORMAL ARGUMENTS .. 
      INTEGER IRANK(*),INDX(*),N
! 
!.. LOCAL SCALARS .. 
      INTEGER J
! 
! ... EXECUTABLE STATEMENTS ... 
!
      DO 100 J = 1,N
         IRANK(INDX(J)) = J
 100  CONTINUE
      END SUBROUTINE RANK

END MODULE TRI_MOD





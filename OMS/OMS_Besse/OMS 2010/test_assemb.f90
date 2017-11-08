program test_assemb
  implicit none
  integer :: NN,NT,NNE
  integer, dimension(30) :: LT
  REAL, dimension(22) :: LP
  integer, dimension(10,3) :: T
  real, dimension(11,2) :: P
  real, dimension(2,11) :: TP
  integer, dimension(3) :: TRI
  real, dimension(3,3) :: S1,S2,S3
  real, dimension(2) :: P1,P2,P3,B1,B2
  real :: detB,G1,G2,G3
  integer, dimension(:,:), allocatable :: IND
  real, dimension(:,:), allocatable :: VAL
  real, dimension(:), allocatable :: A
  integer, dimension(:), allocatable :: NBE,IA,JA
  integer :: k, l, lig, i, j
  LT=(/ 1, 5, 4, &
        1, 2, 5, &
        2, 6, 5, &
        2, 3, 6, &
        4, 8, 7, &
        4, 5, 8, &
        5, 9, 8, &
        5, 6, 9, &
        7,11,10, &
        7, 8,11  /)
  i=1
  do j=1,10
     do k=1,3
        T(j,k)=LT(i)
        i=i+1
     end do
  end do
  !T=transpose(reshape(LT,(/3,10/)))

  LP=(/ 0.0, 0.   , &
        0.5, 0.   , &
        1.0, 0.   , &
        0.0, 0.333, &
        0.5, 0.333, &
        1.0, 0.333, &
        0.0, 0.666, &
        0.5, 0.666, &
        1.0, 0.666, &
        0.0, 1.   , &
        0.5, 1.   /)

  LP=(/ 0.0, 0., &
        1.0, 0., &
        2.0, 0., &
        0.0, 1., &
        1.0, 1., &
        2.0, 1., &
        0.0, 2., &
        1.0, 2., &
        2.0, 2., &
        0.0, 3., &
        1.0, 3./)

  i=1
  do j=1,11
     do k=1,2
        P(j,k)=LP(i)
        i=i+1
     end do
     !write(6,'(2F6.3)') P(j,1),P(j,2)
  end do
  !P=transpose(reshape(LP,(/2,11/)))

  
  S1=0.5*reshape((/1,-1,0,-1,1,0,0,0,0/),(/3,3/))
  S2=0.5*reshape((/2,-1,-1,-1,0,1,-1,1,0/),(/3,3/))
  S3=0.5*reshape((/1,0,-1,0,0,0,-1,0,1/),(/3,3/))


  NT=10
  NN=11
  ! La valeur 10 suivante est le nombre maximal de voisins autorises pour un sommet
  allocate(IND(NN,10),VAL(NN,10),NBE(NN))
  
  ! Parcours du maillage et assemblage
  do k=1,NT
     !write(6,*) 'Triangle ',k
     tri=T(k,:)
     ! Points du triangle tri
     P1=P(tri(1),:);     P2=P(tri(2),:);     P3=P(tri(3),:)

     ! Definition des termes de geometrie
     ! Matrice elementaire locale
     ! G1*S1+G2*S2+G3*S3
     B1=P2-P1;    B2=P3-P1
     detB=B1(1)*B2(2)-B1(2)*B2(1)
     G1= (B2(1)*B2(1)+B2(2)*B2(2))/detB
     G2=-(B1(1)*B2(1)+B1(2)*B2(2))/detB
     G3= (B1(1)*B1(1)+B1(2)*B1(2))/detB

     ! Boucle sur les elements du triangle
     do l=1,3
        lig=tri(l) ! numero du noeud courant
        if (nbe(lig).eq.0) then
           ! Si le noeud n'a jamais ete parcouru
           ! On met dans la liste de connectivite l'ensemble
           ! des noeuds du triangle
           ind(lig,1:3)=tri 
           ! Le nombre d'element de la ligne correspondant
           ! au noeud courant devient 3
           nbe(lig)=3
           ! Assemblage
!!$           val(lig,1:3)=1.0
           val(lig,1:3)=G1*S1(l,1:3)+G2*S2(l,1:3)+G3*S3(l,1:3)
        else
           ! Le noeud a deja ete visite dans un triangle precedant
           ! Il faut alors parcourir l'ensemble de ses voisins
           ! (y compris lui même)
           out: do i=1,3
              ! Cette boucle sert à eviter de rajouter des doublons
              ! dans la liste de connectivite. Si une connectivite
              ! existe déjà, on rajoute a la composante dans la matrice
              ! l'element de la matrice locale
              do j=1,nbe(lig)
                 if (ind(lig,j).eq.tri(i)) then
!!$                    val(lig,j)=val(lig,j)+1.0
                    val(lig,j)=val(lig,j)+G1*S1(l,i)+G2*S2(l,i)+G3*S3(l,i)
                    cycle out
                 end if
              end do
              ! Si aucune connection n'existait, on la cree.
              ! On assemble l'element a partir de la matrice locale
              nbe(lig)=nbe(lig)+1
              ind(lig,nbe(lig))=tri(i)
!!$              val(lig,nbe(lig))=1.0
              val(lig,nbe(lig))=G1*S1(l,i)+G2*S2(l,i)+G3*S3(l,i)
           end do out
        end if
     end do
  end do
  
  ! Mise au format CSR de la matrice du systeme lineaire
  NNE=sum(nbe)
  write(6,*) 'Nombre d elements non nuls : ',NNE
  allocate(A(NNE),JA(NNE),IA(NN+1))
  j=1
  do lig=1,NN
     A(j:j+nbe(lig)-1)=val(lig,1:nbe(lig))
     JA(j:j+nbe(lig)-1)=ind(lig,1:nbe(lig))
     if (lig==1) then
        ia(lig)=1
     else
        IA(lig)=ia(lig-1)+nbe(lig-1)
     end if
     j=j+nbe(lig)
  end do
  ia(NN+1)=ia(NN)+nbe(NN)

  deallocate(IND,NBE,VAL)
  
  call prints(NN,NNE,A,IA,JA)
  deallocate(A,JA,IA)


contains

  SUBROUTINE csr_to_full(n,iwk,A,IA,JA,F)
    ! Christophe Besse 2010
    integer, intent(in) :: n,iwk
    real (kind=4), dimension(iwk), intent(in) :: a
    integer ,dimension(iwk), intent(in) :: ja
    integer ,dimension(n+1), intent(in) :: ia
    
    real (kind=4), INTENT(out), DIMENSION(n,n) :: F
    
    INTEGER :: i,k,j
    
    F=0.0

    DO i=1,n
       DO k=ia(i),ia(i+1)-1
          j = ja(k) 
          F(i,j) = A(k)
       END DO
    END DO

    RETURN
  END SUBROUTINE csr_to_full

  SUBROUTINE prints(n,iwk,A,IA,JA)
    ! Christophe Besse 2010
    integer, intent(in) :: n,iwk
    real (kind=4), dimension(iwk), intent(in) :: a
    integer ,dimension(iwk), intent(in) :: ja
    integer ,dimension(n+1), intent(in) :: ia
    REAL (kind=4), DIMENSION(:,:), ALLOCATABLE :: B
    INTEGER :: ncol,j,k
    CHARACTER (len=40) :: forma

    ncol=n
    ALLOCATE(B(n,n)) 
    CALL csr_to_full(n,iwk,A,IA,JA,B)

    WRITE(FORMA,*) "(",ncol,"f9.4)"
    WRITE(6,'(a1,1x)',ADVANCE='NO') "/"
    DO k=1,ncol
       IF (B(1,k).EQ.0.0) THEN
          WRITE(6,'(8x,a1)',ADVANCE='NO') "0"
       ELSE
          WRITE(6,'(f9.4)',ADVANCE='NO')  B(1,k)
       END IF
    END DO
    WRITE(6,'(1x,a1,1x)') "\\"
    DO j=2,n-1
       WRITE(6,'(a1,1x)',ADVANCE='NO') "|"
       DO k=1,ncol
          IF (B(j,k).EQ.0.0d0) THEN
             WRITE(6,'(8x,a1)',ADVANCE='NO') "0"
          ELSE
             WRITE(6,'(f9.4)',ADVANCE='NO')  B(j,k)
          END IF
       END DO
       WRITE(6,'(1x,a1,1x)') "|"
    END DO
    WRITE(6,'(a1,1x)',ADVANCE='NO') "\\"
    DO k=1,ncol
       IF (B(n,k).EQ.0.0d0) THEN
          WRITE(6,'(8x,a1)',ADVANCE='NO') "0"
       ELSE
          WRITE(6,'(f9.4)',ADVANCE='NO')  B(n,k)
       END IF
    END DO
    WRITE(6,'(1x,a1,1x)') "/"
    DEALLOCATE(B)

    RETURN
  END SUBROUTINE prints

end program test_assemb

program test_assemb
!  use some_fun_saad
  use mod_lec_fic
  USE SysLinCSR
  USE Precond
  USE BLASS
  implicit none
  integer :: NN,NT,NNE
  integer, dimension(3) :: TRI
  real (kind=8), dimension(3,3) :: S1,S2,S3,M1,Mloc
  real (kind=8), dimension(2) :: P1,P2,P3,B1,B2
  real :: detB,G1,G2,G3
  integer, dimension(:,:), allocatable :: IND
  real (kind=8), dimension(:,:), allocatable :: VAL
  real (kind=8), dimension(:), allocatable :: A,RHS,V1,V2,SOL
  integer, dimension(:), allocatable :: NBE,IA,JA,iwork,vu
  integer, dimension(:,:), allocatable :: noeud_bord
  integer :: k, l, lig, i, j, iadd
  real(kind=8) :: cl, val_cl
  integer :: reft,k1,k2,k3,pp1,pp2,type,Zone_Phys_Bord
  real (kind=8) :: x1,x2,x3,y1,y2,y3,F1,F2,F3

  type(msh) :: mesh
  TYPE(MatriceCSR) :: Mat2D

  call load_gmsh('ex1.msh',mesh)
  
  print *,"Nombre de noeud : ",mesh%nbnod
  print *,"Nombre de triangle : ",mesh%nbtriangles
  
  S1=0.5*reshape((/1,-1,0,-1,1,0,0,0,0/),(/3,3/))
  S2=0.5*reshape((/2,-1,-1,-1,0,1,-1,1,0/),(/3,3/))
  S3=0.5*reshape((/1,0,-1,0,0,0,-1,0,1/),(/3,3/))

  M1=reshape((/2,1,1,1,2,1,1,1,2/),(/3,3/))

  NT=mesh%nbTriangles
  NN=mesh%nbNod
  ! La valeur 10 suivante est le nombre maximal de voisins autorises pour un sommet
  allocate(IND(NN,10),VAL(NN,10),NBE(NN))
  allocate(rhs(NN),V1(NN),V2(NN))
  ind=0; val=0; nbe=0;
  rhs=0.0; v1=0.; v2=0.
  
  ! Parcours du maillage et assemblage
  do k=1,NT
     !write(6,*) 'Triangle ',k
     tri=mesh%triangles(k,1:3)
     ! Points du triangle tri
     k1=tri(1); k2=tri(2); k3=tri(3);
     P1=mesh%POS(k1,1:2);     P2=mesh%POS(k2,1:2);     P3=mesh%POS(k3,1:2)

     x1=P1(1); y1=P1(2);
     x2=P2(1); y2=P2(2);
     x3=P3(1); y3=P3(2);
     
     ! Definition des termes de geometrie
     ! Matrice elementaire locale
     ! G1*S1+G2*S2+G3*S3
     B1=P2-P1;    B2=P3-P1
     detB=B1(1)*B2(2)-B1(2)*B2(1)
     G1= (B2(1)*B2(1)+B2(2)*B2(2))/detB
     G2=-(B1(1)*B2(1)+B1(2)*B2(2))/detB
     G3= (B1(1)*B1(1)+B1(2)*B1(2))/detB
     
     Mloc=(detB/24.)*M1;

     F1=fun_s(x1,y1)
     F2=fun_s(x2,y2)
     F3=fun_s(x3,y3)
!     print *,k,F1,F2,F3
!     print *,k,k1,F1,Mloc(1,1)*F1+Mloc(1,2)*F2+Mloc(1,3)*F3
  
     ! Boucle sur les elements du triangle
     do l=1,3
        lig=tri(l) ! numero du noeud courant
        rhs(lig)=rhs(lig)+Mloc(l,1)*F1+Mloc(l,2)*F2+Mloc(l,3)*F3
        if (nbe(lig).eq.0) then
           ! Si le noeud n'a jamais ete parcouru
           ! On met dans la liste de connectivite l'ensemble
           ! des noeuds du triangle
           ind(lig,1:3)=tri 
           ! Le nombre d'element de la ligne correspondant
           ! au noeud courant devient 3
           nbe(lig)=3
           ! Assemblage
           val(lig,1:3)=G1*S1(l,1:3)+G2*S2(l,1:3)+G3*S3(l,1:3)
           !val(lig,1:3)=Aloc(l,1:3)
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
                    val(lig,j)=val(lig,j)+G1*S1(l,i)+G2*S2(l,i)+G3*S3(l,i)
                    !val(lig,j)=val(lig,j)+Aloc(l,i)
                    cycle out
                 end if
              end do
              ! Si aucune connection n'existait, on la cree.
              ! On assemble l'element a partir de la matrice locale
              nbe(lig)=nbe(lig)+1
              ind(lig,nbe(lig))=tri(i)
              val(lig,nbe(lig))=G1*S1(l,i)+G2*S2(l,i)+G3*S3(l,i)
              !val(lig,nbe(lig))=Aloc(l,i)
           end do out
        end if
     end do
  end do


  ! Mise au format CSR de la matrice du systeme lineaire
  NNE=sum(nbe)
  write(6,*) 'Nombre d elements non nuls dans la matrice : ',NNE
  allocate(A(NNE),JA(NNE),IA(NN+1))
  A=0.; ja=0; ia=0;
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
  
  allocate(iwork(max(nn+1,2*nne)))
  call csort(nn,a,ja,ia,iwork,.true.) 
  deallocate(iwork)


!! Prise en compte de CL de Dirichlet
!! Identification des noeuds
!! Pour un noeud I sur le bord
!! On doit mettre la ligne I corresp. a ce noeud a 0 sauf la colonne I a 1 
!! On doit mettre la colonne I corresp. a ce noeud a 0 sauf la ligne I a 1 
!! WHERE(JA.EQ.I) A=0.0
!! A(IA(I):IA(I+1)-1)=0.0
!! GETELM (I,I,a,ja,ia,iadd,.TRUE.)
!! A(IADD)=1.0
!! - On calcule la matrice de raideur complete Aij et les forces nodales Bi. 
!! - On multiplie la k-ieme colonne de la matrice de raideur par la valeur Tk, et on la
!!   soustrait au vecteur des forces nodales.
!!   B = B - A*(0,...,0,Tk,0,...0)
!!   call amux(n,x,y,a,ja,ia)
!! - La k-ieme ligne et la k-ieme colonne de la matrice sont remplacees par une ligne 
!!   et une colonne de zeros.
!!   WHERE(JA.EQ.I) A=0.0
!!   A(IA(I):IA(I+1)-1)=0.0
!! - Le terme Akk est remplacee par 1.
!!   GETELM (I,I,a,ja,ia,iadd,.TRUE.)
!!   A(IADD)=1.0
!! - La composante Bk est remplacee par Tk.
!!   B(k)=Tk



  allocate(noeud_bord(mesh%nbLines,2))
  allocate(vu(mesh%nbNod))
  noeud_bord=0
  vu=0
  k=0
  do j=1,mesh%nbLines
     pp1=mesh%LINES(j,1)
     pp2=mesh%LINES(j,2)
     type=mesh%LINES(j,3)
     if (vu(pp1).eq.0) then
        k=k+1
        noeud_bord(k,1)=pp1
        noeud_bord(k,2)=type
        vu(pp1)=k
     end if
     if (vu(pp2).eq.0) then
        k=k+1
        noeud_bord(k,1)=pp2
        noeud_bord(k,2)=type
        vu(pp2)=k
     end if
  end do


  V1=0.0
  cl=20.
  Zone_Phys_Bord=1001
  Zone_Phys_Bord2=1002
  do j=1,mesh%nbLines
     !! Identification du noeud du bord a traiter
     lig=noeud_bord(j,1)
     type=noeud_bord(j,2)
     if (type.eq.Zone_Phys_Bord) then
        val_CL=CL
     end if
     !! - On multiplie la k-ieme colonne de la matrice de raideur par la valeur Tk, et on la
     !!   soustrait au vecteur des forces nodales.
     !!   B = B - A*(0,...,0,Tk,0,...0)
     V1(lig)=Val_CL
     call amux(nn,v1,v2,a,ja,ia)
     !print *,maxval(v2)
     rhs=rhs-v2
     v1(lig)=0.0
     !! - La k-ieme ligne et la k-ieme colonne de la matrice sont remplacees par une ligne 
     !!   et une colonne de zeros.
     WHERE(JA.EQ.LIG) A=0.0
     A(IA(LIG):IA(LIG+1)-1)=0.0
     !! - Le terme Akk est remplacee par 1.
     g1=GETELM (lig,lig,a,ja,ia,iadd,.true.)
     A(IADD)=1.0
     !! - La composante Bk est remplacee par Tk.
     RHS(LIG)=Val_CL
  end do

!!$  open(100,file='test_ass3.ps',status='unknown')
!!$  call pspltm(nn,nn,0,ja,ia,'Ma matrice',0,21.,'cm',0,(/1/),100)
!!$  close(100)


  open(unit=100,file='mat_fish.dat',status='unknown')
  do i=1,nn
     do j=ia(i),ia(i+1)-1
        write(100,*) i,ja(j),a(j)
     end do
  end do
  close(100)
  open(100,file='rhs_fish.dat',status='unknown')
  do i=1,nn
     write(100,*) rhs(i)
  end do
  close(100)

  open(100,file='nodes_fish.dat',status='unknown')
  do j=1,nn
     write(100,*) mesh%pos(j,:)
  end do
  close(100)


  CALL CREATE(Mat2D,NN,NNE)
  ALLOCATE (Mat2D%Guess(Mat2d%NbLine)) ! Necessaire pour methode iterative
  ! Copie des donnees dans la matrice Mat2D
  Mat2D%Resolution = 1
  Mat2D%Coeff(:) = a(:)
  Mat2D%IA(:) = ia(:)
  Mat2D%JA(:) = ja(:)
  Mat2D%Guess(:) = 0.

  !
  !     set-up the preconditioner ILUT(15, 1E-4)  ! new definition of lfil
  !
  Mat2D%Precond=3        ! Type de preconditioneur ILUT
  Mat2D%LFill=3          ! Niveau de remplissage
  Mat2D%DropTol=1.0D-4   ! Tolerance
  CALL CalPrecond(Mat2D)

  allocate(sol(NN))
  
  CALL RESOL(Mat2D,RHS(1:NN),SOL(1:NN))


  CALL DESTROY(Mat2D)

  
  open(unit=100,file='solution.dat',status='unknown')
  do i=1,nn
        write(100,*) sol(i)
  end do
  close(100)



  deallocate(A,JA,IA,SOL)
  deallocate(v1,v2,rhs)
  deallocate(noeud_bord,vu)
  
  contains
    function fun_s(x,y) result(z)
      implicit none
      real(kind=8), intent(in) :: x,y
      real(kind=8) :: r,z
      r=sqrt(x**2.+y**2.)
      z=0.
      if ((r>1.5).and.(r<2)) then
         z=z+20*((r-1.5)**2.)*(r-2)**2.
      end if
      if (r<1.) then
         z=z-0.5*(r**2.)*((r-1)**2.)
      end if
      return
    end function fun_s
end program test_assemb
